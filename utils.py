import torch
import numpy as np
import torch.nn as nn
import re
import gymnasium as gym

# Environment name mapping to Gymnasium identifiers
ENV_MAP = {
    'FetchReach': 'FetchReach-v2',
    'FetchPush': 'FetchPush-v2',
    'FetchSlide': 'FetchSlide-v2',
    'FetchPick': 'FetchPickAndPlace-v2',
    'HandManipulateBlockRotateZ': 'HandManipulateBlockRotateZ-v1',
    'HandManipulateBlockRotateParallel': 'HandManipulateBlockRotateParallel-v1',
    'HandManipulateBlockRotateXYZ': 'HandManipulateBlockRotateXYZ-v1',
    'HandManipulateBlockFull': 'HandManipulateBlockFull-v1',
    'HandManipulateEggRotate': 'HandManipulateEggRotate-v1',
    'HandManipulateEggFull': 'HandManipulateEggFull-v1',
    'HandManipulatePenRotate': 'HandManipulatePenRotate-v1',
    'HandManipulatePenFull': 'HandManipulatePenFull-v1',
}

def parse_seed_from_filename(filename):
    """Extract seed value from formatted filename"""
    match = re.search(r'_sd(\d+)_', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1))


# ----------------------- Core Network Architectures -----------------------
class Actor(nn.Module):
    """Policy network for generating actions"""

    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_action),
            nn.Tanh()
        )

    def forward(self, s, g):
        """Forward pass with state and goal inputs"""
        x = torch.cat([s, g], -1)
        return self.max_action * self.net(x)


# ----------------------- Data Normalization Classes -----------------------
class Normalizer(object):
    """Online normalizer for observations and goals (MPI dependency removed)"""

    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.mean = np.zeros(self.size, dtype=np.float32)
        self.std = np.ones(self.size, dtype=np.float32)

    def normalize(self, v, clip_range=None):
        """Normalize input data with clipping"""
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std + 1e-8), -clip_range, clip_range)


# ----------------------- Agent Class (Simplified) -----------------------
class Agent(object):
    """Evaluation-focused agent with essential methods only"""

    def __init__(self, args, env=None):
        self.args = args
        self.actor = Actor(args)
        self.s_norm = Normalizer(size=args.dim_state, default_clip_range=args.clip_range)
        self.g_norm = Normalizer(size=args.dim_goal, default_clip_range=args.clip_range)

    def _preproc_inputs(self, s=None, g=None):
        """Preprocess state and goal inputs"""
        s_ = self.s_norm.normalize(np.clip(s, -self.args.clip_obs, self.args.clip_obs)) if s is not None else None
        g_ = self.g_norm.normalize(np.clip(g, -self.args.clip_obs, self.args.clip_obs)) if g is not None else None
        s_tensor = torch.FloatTensor(s_) if s is not None else None
        g_tensor = torch.FloatTensor(g_) if g is not None else None
        return s_tensor, g_tensor

    def select_action(self, observation_dict, stochastic=False):
        """Select action based on observation (deterministic during evaluation)"""
        s = observation_dict['observation'].astype(np.float32)
        g = observation_dict['desired_goal'].astype(np.float32)
        s_tensor, g_tensor = self._preproc_inputs(s, g)
        with torch.no_grad():
            action = self.actor(s_tensor, g_tensor).cpu().numpy().squeeze()
        return action


def load_policy_model(filepath):
    """Load policy model from checkpoint file"""
    policy = torch.load(filepath, map_location='cpu', weights_only=False)
    agent = Agent(args=policy["args"])
    agent.actor.load_state_dict(policy["actor"])
    agent.s_norm.mean = policy["s_mean"]
    agent.s_norm.std = policy["s_std"]
    agent.g_norm.mean = policy["g_mean"]
    agent.g_norm.std = policy["g_std"]
    return agent


def create_environment(env_name):
    """Create and wrap specified environment"""
    if env_name not in ENV_MAP:
        raise ValueError(f"Unsupported environment: {env_name}")
    return gym.make(ENV_MAP[env_name], max_episode_steps=1000)


def numpy2torch(v, unsqueeze=False):
    """Convert numpy array to PyTorch tensor"""
    v_tensor = torch.FloatTensor(v) if v.dtype in (np.float32, np.float64) else torch.LongTensor(v)
    return v_tensor.unsqueeze(0) if unsqueeze else v_tensor


def q_multiply(q1, q2):
    """Multiply two qs (w, x, y, z format)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def normalize_q(q):
    """Normalize q to unit length"""
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-6 else q  # Prevent division by zero


def q_inverse(q):
    """Compute inverse of a q"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


