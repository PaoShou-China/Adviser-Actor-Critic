from network import Actor, Critic
from utils import convert_to_tensor
from replay_buffers.hindsight_replay_buffer import HindsightReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class SAC(nn.Module):
    def __init__(self, writer, device, env, args):
        super(SAC, self).__init__()
        self.args = args
        self.env = env
        obs_dim = (env.observation_space['observation'].shape[0] +
                   env.observation_space['achieved_goal'].shape[0] +
                   env.observation_space['desired_goal'].shape[0])
        act_dim = env.action_space.shape[0]
        self.actor = Actor(args.layer_num, obs_dim, act_dim, args.hidden_dim,
                           args.activation_function, args.last_activation, args.trainable_std)

        self.q_1 = Critic(args.layer_num, obs_dim + act_dim, 1, args.hidden_dim,
                          args.activation_function, args.last_activation)
        self.q_2 = Critic(args.layer_num, obs_dim + act_dim, 1, args.hidden_dim,
                          args.activation_function, args.last_activation)

        self.target_q_1 = Critic(args.layer_num, obs_dim + act_dim, 1, args.hidden_dim,
                                 args.activation_function, self.args.last_activation)
        self.target_q_2 = Critic(args.layer_num, obs_dim + act_dim, 1, args.hidden_dim,
                                 args.activation_function, args.last_activation)

        self.soft_update(self.q_1, self.target_q_1, 1.)
        self.soft_update(self.q_2, self.target_q_2, 1.)

        self.alpha = nn.Parameter(torch.tensor(args.alpha_init))

        self.data = HindsightReplayBuffer(env, int(args.memory_size), args.n_sampled_goals, args.goal_strategy)
        self.target_entropy = - torch.tensor(act_dim)

        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=args.q_lr)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=args.q_lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=args.alpha_lr)

        self.device = device
        self.writer = writer

    def put_data(self, transition, done):
        self.data.put_data(transition, done)

    @staticmethod
    def soft_update(network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def get_action(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs, device=self.device).unsqueeze(dim=0)
        mu, std = self.actor(obs)
        if deterministic: return torch.tanh(mu)
        dist = Normal(mu, std)
        u = dist.rsample()
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) + 1e-3)
        return a, a_log_prob.sum(-1, keepdim=True)

    def q_update(self, Q, q_optimizer, obss, actions, rewards, next_obss, dones):
        # target
        with torch.no_grad():
            next_actions, next_action_log_prob = self.get_action(next_obss)
            q_1 = self.target_q_1(next_obss, next_actions)
            q_2 = self.target_q_2(next_obss, next_actions)
            q = torch.min(q_1, q_2)
            v = (1 - dones) * (q - self.alpha * next_action_log_prob)
            targets = rewards + self.args.gamma * v

        q = Q(obss, actions)
        loss = F.smooth_l1_loss(q, targets)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        return loss

    def actor_update(self, obss):
        now_actions, now_action_log_prob = self.get_action(obss)
        q_1 = self.q_1(obss, now_actions)
        q_2 = self.q_2(obss, now_actions)
        q = torch.min(q_1, q_2)

        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss, now_action_log_prob

    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        return loss

    def train_net(self, batch_size, n_epi):
        data = self.data.random_batch(batch_size=batch_size)
        states, actions, rewards, next_states, dones = \
            convert_to_tensor(self.device, data['observations'], data['actions'], data['rewards'],
                              data['next_observations'], data['terminals'])

        ###q update
        q_1_loss = self.q_update(self.q_1, self.q_1_optimizer, states, actions, rewards, next_states, dones)
        q_2_loss = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, dones)

        ### actor update
        actor_loss, prob = self.actor_update(states)

        ###alpha update
        alpha_loss = self.alpha_update(prob)

        self.soft_update(self.q_1, self.target_q_1, self.args.soft_update_rate)
        self.soft_update(self.q_2, self.target_q_2, self.args.soft_update_rate)

        if self.writer is not None:
            self.writer.add_scalar("loss/q_1", q_1_loss, n_epi)
            self.writer.add_scalar("loss/q_2", q_2_loss, n_epi)
            self.writer.add_scalar("loss/actor", actor_loss, n_epi)
            self.writer.add_scalar("loss/alpha", alpha_loss, n_epi)
