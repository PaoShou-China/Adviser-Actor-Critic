import copy
import numpy as np
from utils import q_multiply, q_inverse


def create_adviser(dt=0.02, pos_params=None, rot_params=None):
    """Factory function to create appropriate controller based on parameters.

    Args:
        dt (float): Control step duration in seconds
        pos_params (tuple, optional): Position control parameters (kp, ki, sigma)
        rot_params (tuple, optional): Attitude control parameters (kp, ki, sigma)

    Returns:
        Controller: Instance of PositionController, AttitudeController, or CombinedController

    Raises:
        ValueError: If invalid parameter combination or count is provided
    """
    # Create appropriate controller
    if rot_params and pos_params:
        return CombinedController(dt, pos_params, rot_params)
    elif pos_params:
        return PositionController(dt, *pos_params)
    elif rot_params:
        return AttitudeController(dt, *rot_params)
    else:
        raise ValueError("At least one parameter group must be provided")


class PositionController:
    """3D position PI controller with anti-windup mechanism.

    Attributes:
        dt (float): Control step duration
        kp (float): Proportional gain
        ki (float): Integral gain
        sigma (float): Anti-windup threshold
        integral (np.ndarray): Integral term storage
        time (float): Internal timer
    """

    def __init__(self, dt, kp, ki, sigma):
        """Initialize position controller with specified parameters."""
        self.dt = dt
        self.kp, self.ki, self.sigma = kp, ki, sigma
        self.integral = np.zeros(3)
        self.time = 0.0

    def reset(self):
        """Reset controller state (integral term and timer)."""
        self.integral = np.zeros(3)
        self.time = 0.0

    def __call__(self, obs_dict):
        """Compute adjusted goal using PI control law.

        Args:
            obs_dict (dict): Observation containing 'achieved_goal' and 'desired_goal'

        Returns:
            dict: Modified observation with adjusted desired_goal
        """
        # Validate input dimensions
        achieved = obs_dict['achieved_goal']
        desired = obs_dict['desired_goal']
        assert achieved.shape == (3,), f"Invalid achieved_goal shape: {achieved.shape}"
        assert desired.shape == (3,), f"Invalid desired_goal shape: {desired.shape}"

        self.time += self.dt
        err = achieved - desired

        # Anti-windup integration
        if np.linalg.norm(err) < self.sigma:
            self.integral = np.clip(
                self.integral + err * self.dt,
                -self.sigma,
                self.sigma
            )

        goal_diff = self.kp * err + self.ki * self.integral
        new_obs = copy.deepcopy(obs_dict)
        new_obs['desired_goal'] = achieved - goal_diff
        return new_obs


class AttitudeController:
    """Quaternion-based rotitude PI controller with anti-windup.

    Attributes:
        dt (float): Control step duration
        kp (float): Proportional gain
        ki (float): Integral gain
        sigma (float): Anti-windup threshold
        integral (np.ndarray): Integral term storage
        time (float): Internal timer
    """

    def __init__(self, dt, kp, ki, sigma):
        """Initialize rotitude controller with specified parameters."""
        self.dt = dt
        self.kp, self.ki, self.sigma = kp, ki, sigma
        self.integral = np.zeros(3)
        self.time = 0.0

    def reset(self):
        """Reset controller state (integral term and timer)."""
        self.integral = np.zeros(3)
        self.time = 0.0

    def __call__(self, obs_dict):
        """Compute adjusted quaternion goal using PI control law.

        Args:
            obs_dict (dict): Observation containing 'achieved_goal' and 'desired_goal'

        Returns:
            dict: Modified observation with adjusted desired_goal
        """
        # Validate input dimensions
        achieved = obs_dict['achieved_goal']
        desired = obs_dict['desired_goal']
        assert achieved.shape == (4,), f"Invalid achieved_goal shape: {achieved.shape}"
        assert desired.shape == (4,), f"Invalid desired_goal shape: {desired.shape}"

        self.time += self.dt

        # Calculate quaternion error
        err_q = q_multiply(achieved, q_inverse(desired))
        q_imag = err_q[1:]  # Use imaginary components for control

        # Anti-windup integration
        if np.linalg.norm(q_imag) < self.sigma / 2.0:
            self.integral = np.clip(
                self.integral + q_imag * self.dt,
                -self.sigma / 2.0,
                self.sigma / 2.0
            )

        goal_diff = -self.kp * q_imag - self.ki * self.integral
        w = np.sqrt(max(1 - np.sum(goal_diff ** 2), 0))
        new_obs = copy.deepcopy(obs_dict)
        new_obs['desired_goal'] = q_multiply([w, *goal_diff], achieved)
        return new_obs


class CombinedController:
    """Combined 7D controller for position and rotitude tracking.

    Attributes:
        dt (float): Control step duration
        pos_ctrl (PositionController): Position sub-controller
        rot_ctrl (AttitudeController): Attitude sub-controller
        time (float): Internal timer
    """

    def __init__(self, dt, pos_params, rot_params):
        """Initialize combined controller with sub-controllers."""
        self.dt = dt
        self.pos_ctrl = PositionController(dt, *pos_params)
        self.rot_ctrl = AttitudeController(dt, *rot_params)
        self.time = 0.0

    def reset(self):
        """Reset both sub-controllers and internal timer."""
        self.pos_ctrl.reset()
        self.rot_ctrl.reset()
        self.time = 0.0

    def __call__(self, obs_dict):
        """Compute combined adjustment for 7D pose.

        Args:
            obs_dict (dict): Observation containing 7D 'achieved_goal' and 'desired_goal'

        Returns:
            dict: Modified observation with adjusted desired_goal
        """
        # Validate input dimensions
        achieved = obs_dict['achieved_goal']
        desired = obs_dict['desired_goal']
        assert achieved.shape == (7,), f"Invalid achieved_goal shape: {achieved.shape}"
        assert desired.shape == (7,), f"Invalid desired_goal shape: {desired.shape}"

        self.time += self.dt

        # Process position and rotitude separately
        pos_obs = {
            'achieved_goal': achieved[:3],
            'desired_goal': desired[:3]
        }
        rot_obs = {
            'achieved_goal': achieved[3:],
            'desired_goal': desired[3:]
        }

        fake_pos = self.pos_ctrl(pos_obs)
        fake_rot = self.rot_ctrl(rot_obs)

        # Combine results
        new_obs = copy.deepcopy(obs_dict)
        fake_goal = np.concatenate([
            fake_pos['desired_goal'],
            fake_rot['desired_goal']
        ])
        new_obs['desired_goal'] = fake_goal
        return new_obs
