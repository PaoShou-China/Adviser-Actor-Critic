import numpy as np
import pygame
from numpy.random import uniform
from .core import Simulator, RobotEnv, RobotTask


class RoamingSimulator(Simulator):
    def derivative_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        fx, fy = action[0], action[1]
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        ax, ay = (fx - 0.25 * vx - 0.25 * x) / 0.5, (fy - 0.25 * vy - 0.25 * y) / 0.5
        return np.array([vx, vy, ax, ay])

    def get_action_dim(self) -> int:
        return 2


class RoamingTask(RobotTask):
    def get_observation(self) -> np.ndarray:
        observation = self.sim.state.copy()
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        achieved_goal = self.sim.state.copy()
        return achieved_goal

    def get_desired_goal(self) -> np.ndarray:
        desired_goal = self.sim.goal.copy()
        return desired_goal

    def reset(self, options: dict):
        """
        Reset the environment state and goal to either predefined values or
        random values within specified ranges.

        :param options: A dictionary that can contain 'init_state' and 'goal'
                        keys with corresponding initial state and goal arrays.
        """
        if options is not None:
            # If options are provided, use them to initialize state and goal
            if 'init_state' in options:
                # Set the initial state from the options if provided
                self.sim.state = options['init_state']
            else:
                # Otherwise, set the initial state to random values within the specified range
                self.sim.state = np.array([
                    uniform(-2.4, 2.4),  # Initial x-position
                    uniform(-2.4, 2.4),  # Initial y-position
                    uniform(-0.1, 0.1),  # Initial x-velocity
                    uniform(-0.1, 0.1),  # Initial y-velocity
                ])

            if 'goal' in options:
                # Set the goal from the options if provided
                self.sim.goal = options['goal']
            else:
                # Otherwise, set the goal to random values within the specified range
                self.sim.goal = np.array([
                    uniform(-2.4, 2.4),  # Goal x-position
                    uniform(-2.4, 2.4),  # Goal y-position
                    0.0,  # Goal x-velocity (set to 0.0)
                    0.0,  # Goal y-velocity (set to 0.0)
                ])
        else:
            # If no options are provided, initialize both state and goal randomly
            self.sim.state = np.array([
                uniform(-2.4, 2.4),  # Initial x-position
                uniform(-2.4, 2.4),  # Initial y-position
                uniform(-0.1, 0.1),  # Initial x-velocity
                uniform(-0.1, 0.1),  # Initial y-velocity
            ])
            self.sim.goal = np.array([
                uniform(-2.4, 2.4),  # Goal x-position
                uniform(-2.4, 2.4),  # Goal y-position
                0.0,  # Goal x-velocity (set to 0.0)
                0.0,  # Goal y-velocity (set to 0.0)
            ])

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        velocity = np.linalg.norm(achieved_goal[2:] - desired_goal[2:])
        success = distance < 0.01 and velocity < 0.01
        return success

    def is_failure(self, observation: np.ndarray, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        x, y = observation[0], observation[1]
        return abs(x) > 4.8 or abs(y) > 4.8

    def compute_reward(self, action: np.ndarray, observation: np.ndarray, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, terminated: bool) -> float:

        distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        velocity = np.linalg.norm(achieved_goal[2:] - desired_goal[2:])

        # Define reward coefficients
        distance_cost_coef = -1.0
        velocity_cost_coef = -0.5
        action_cost_coef = -0.1

        # Reward formula
        reward = (distance_cost_coef * distance ** 2 +
                  velocity_cost_coef * velocity ** 2 +
                  action_cost_coef * np.linalg.norm(action) - 1.0) * self.sim.dt

        # If the distance is less than a threshold, consider it a success
        if self.is_success(achieved_goal, desired_goal):
            reward += 50  # Add success reward
        if self.is_failure(observation, achieved_goal, desired_goal):
            reward += -50  # Add failure cost

        return reward


class RoamingEnv(RobotEnv):
    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((480, 480))
                self.clock = pygame.time.Clock()
            self.screen.fill((255, 255, 255))

            pygame.draw.circle(self.screen, (255, 0, 0), 240 + self.sim.state[:2] * 50, 10)
            pygame.draw.circle(self.screen, (0, 0, 0), 240 + self.sim.goal[:2] * 50, 10)

            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.Surface((480, 480))

            self.screen.fill((255, 255, 255))
            pygame.draw.circle(self.screen, (255, 0, 0), 240 + self.sim.state[:2] * 50, 10)
            pygame.draw.circle(self.screen, (0, 0, 0), 240 + self.sim.goal[:2] * 50, 10)

            return np.array(pygame.surfarray.pixels3d(self.screen))


def make_env(render_mode: str = None) -> RoamingEnv:
    simulator = RoamingSimulator(dt=0.05)
    task = RoamingTask(simulator)
    action_low = np.array([-1.0, -1.0])
    action_high = np.array([1.0, 1.0])
    env = RoamingEnv(task, action_low, action_high, max_episode_steps=1000, render_mode=render_mode)
    return env
