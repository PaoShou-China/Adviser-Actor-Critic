import numpy as np
import pygame
from numpy.random import uniform
from .core import Simulator, RobotEnv, RobotTask


class ArmSimulator(Simulator):
    l1, l2, l3 = 0.5, 1.0, 1.0
    d1, d2, d3 = 1.0, 1.0, 1.0

    def derivative_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot = state
        theta1_ddot, theta2_ddot, theta3_ddot = action[0] - self.d1 * theta1_dot, \
                                                action[1] - self.d2 * theta2_dot, action[2] - self.d3 * theta3_dot
        return np.array([theta1_dot, theta2_dot, theta3_dot, theta1_ddot, theta2_ddot, theta3_ddot])

    def get_action_dim(self) -> int:
        return 3


class ArmTask(RobotTask):
    def get_observation(self) -> np.ndarray:
        theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot = self.sim.state.copy()
        observation = np.array([np.cos(theta1), np.cos(theta2), np.cos(theta3),
                                np.sin(theta1), np.sin(theta2), np.sin(theta3),
                                theta1_dot, theta2_dot, theta3_dot])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot = self.sim.state.copy()
        x = (self.sim.l2 * np.cos(theta2) + self.sim.l3 * np.cos(theta2 + theta3)) * np.cos(theta1)
        y = (self.sim.l2 * np.cos(theta2) + self.sim.l3 * np.cos(theta2 + theta3)) * np.sin(theta1)
        z = self.sim.l1 + self.sim.l2 * np.sin(theta2) + self.sim.l3 * np.sin(theta2 + theta3)
        achieved_goal = np.array([x, y, z])
        return achieved_goal

    def get_desired_goal(self) -> np.ndarray:
        desired_goal = self.sim.goal.copy()
        return desired_goal

    def reset(self, options: dict):
        if options is not None:
            if 'init_state' in options:
                self.sim.state = options['init_state']
            else:
                # Initialize the joint angles and velocities randomly
                self.sim.state = np.array([
                    uniform(-np.pi, np.pi),  # Joint 1 angle
                    uniform(-np.pi, np.pi),  # Joint 2 angle
                    uniform(-np.pi, np.pi),  # Joint 3 angle
                    uniform(-0.005, 0.005),  # Joint 1 velocity
                    uniform(-0.005, 0.005),  # Joint 2 velocity
                    uniform(-0.005, 0.005),  # Joint 3 velocity

                ])

            if 'goal' in options:
                self.sim.goal = options['goal']
            else:
                # Initialize the goal position randomly within a specified range
                self.sim.goal = np.array([
                    uniform(0.5, 1.0),  # Goal x-coordinate
                    uniform(0.5, 1.0),  # Goal y-coordinate
                    uniform(0.0, 0.3),  # Goal z-coordinate
                ])
        else:
            # Initialize both state and goal randomly if no options are provided
            self.sim.state = np.array([
                uniform(-np.pi, np.pi),  # Joint 1 angle
                uniform(-np.pi, np.pi),  # Joint 2 angle
                uniform(-np.pi, np.pi),  # Joint 3 angle
                uniform(-0.005, 0.005),  # Joint 1 velocity
                uniform(-0.005, 0.005),  # Joint 2 velocity
                uniform(-0.005, 0.005),  # Joint 3 velocity

            ])
            self.sim.goal = np.array([
                uniform(0.5, 1.0),  # Goal x-coordinate
                uniform(0.5, 1.0),  # Goal y-coordinate
                uniform(0.0, 0.3),  # Goal z-coordinate
            ])

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = distance < 0.01
        return success

    def is_failure(self, observation: np.ndarray, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        theta1_dot, theta2_dot, theta3_dot = observation[6], observation[7], observation[8]
        return abs(theta1_dot) > 3.0 or abs(theta2_dot) > 3.0 or abs(theta3_dot) > 3.0

    def compute_reward(self, action: np.ndarray, observation: np.ndarray, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, terminated: bool) -> float:

        distance = np.linalg.norm(achieved_goal - desired_goal)
        velocity = np.linalg.norm(observation[6:9])

        # Define reward coefficients
        distance_cost_coef = -1.0
        velocity_cost_coef = -0.1
        action_cost_coef = -0.1

        # Reward formula
        reward = (distance_cost_coef * distance +
                  velocity_cost_coef * velocity +
                  action_cost_coef * np.linalg.norm(action)) * self.sim.dt

        # If the distance is less than a threshold, consider it a success
        if self.is_success(achieved_goal, desired_goal):
            reward += 50  # Add success reward
        if self.is_failure(observation, achieved_goal, desired_goal):
            reward += -50  # Add failure cost

        return reward


class ArmEnv(RobotEnv):
    @staticmethod
    def project(x_0, y_0, z_0):
        t = (x_0 + y_0 + z_0) / 3
        x_p = x_0 - t
        y_p = y_0 - t
        z_p = z_0 - t
        # 在新坐标系中的坐标
        a = -y_p * 100 + 240
        b = -z_p * 100 + 240
        return (a, b)

    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((480, 480))
                self.clock = pygame.time.Clock()
            self.screen.fill((255, 255, 255))
            theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot = self.sim.state.copy()
            px1, py1, pz1 = 0.0, 0.0, self.sim.l1
            px2 = (self.sim.l2 * np.cos(theta2)) * np.cos(theta1)
            py2 = (self.sim.l2 * np.cos(theta2)) * np.sin(theta1)
            pz2 = self.sim.l1 + self.sim.l2 * np.sin(theta2)
            px3 = (self.sim.l2 * np.cos(theta2) + self.sim.l3 * np.cos(theta2 + theta3)) * np.cos(theta1)
            py3 = (self.sim.l2 * np.cos(theta2) + self.sim.l3 * np.cos(theta2 + theta3)) * np.sin(theta1)
            pz3 = self.sim.l1 + self.sim.l2 * np.sin(theta2) + self.sim.l3 * np.sin(theta2 + theta3)
            gx, gy, gz = self.sim.goal

            pygame.draw.circle(self.screen, (0, 0, 0), self.project(0, 0, 0), 10)
            pygame.draw.line(self.screen, (255, 0, 0), self.project(0, 0, 0), self.project(px1, py1, pz1), 5)
            pygame.draw.circle(self.screen, (255, 0, 0), self.project(px1, py1, pz1), 5)
            pygame.draw.line(self.screen, (0, 255, 0), self.project(px1, py1, pz1), self.project(px2, py2, pz2), 5)
            pygame.draw.circle(self.screen, (0, 255, 0), self.project(px2, py2, pz2), 5)
            pygame.draw.line(self.screen, (0, 0, 255), self.project(px2, py2, pz2), self.project(px3, py3, pz3), 5)
            pygame.draw.circle(self.screen, (0, 0, 255), self.project(px3, py3, pz3), 5)
            pygame.draw.circle(self.screen, (255, 0, 255), self.project(gx, gy, gz), 5)

            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.Surface((480, 480))

            self.screen.fill((255, 255, 255))
            theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot = self.sim.state.copy()
            px1, py1, pz1 = 0.0, 0.0, self.sim.l1
            px2 = (self.sim.l2 * np.cos(theta2)) * np.cos(theta1)
            py2 = (self.sim.l2 * np.cos(theta2)) * np.sin(theta1)
            pz2 = self.sim.l1 + self.sim.l2 * np.sin(theta2)
            px3 = (self.sim.l2 * np.cos(theta2) + self.sim.l3 * np.cos(theta2 + theta3)) * np.cos(theta1)
            py3 = (self.sim.l2 * np.cos(theta2) + self.sim.l3 * np.cos(theta2 + theta3)) * np.sin(theta1)
            pz3 = self.sim.l1 + self.sim.l2 * np.sin(theta2) + self.sim.l3 * np.sin(theta2 + theta3)
            gx, gy, gz = self.sim.goal

            pygame.draw.circle(self.screen, (0, 0, 0), self.project(0, 0, 0), 10)
            pygame.draw.line(self.screen, (255, 0, 0), self.project(0, 0, 0), self.project(px1, py1, pz1), 5)
            pygame.draw.circle(self.screen, (255, 0, 0), self.project(px1, py1, pz1), 5)
            pygame.draw.line(self.screen, (0, 255, 0), self.project(px1, py1, pz1), self.project(px2, py2, pz2), 5)
            pygame.draw.circle(self.screen, (0, 255, 0), self.project(px2, py2, pz2), 5)
            pygame.draw.line(self.screen, (0, 0, 255), self.project(px2, py2, pz2), self.project(px3, py3, pz3), 5)
            pygame.draw.circle(self.screen, (0, 0, 255), self.project(px3, py3, pz3), 5)
            pygame.draw.circle(self.screen, (255, 0, 255), self.project(gx, gy, gz), 5)

        return np.array(pygame.surfarray.pixels3d(self.screen))


def make_env(render_mode: str = None) -> ArmEnv:
    simulator = ArmSimulator(dt=0.1)
    task = ArmTask(simulator)
    action_low = np.array([-1.0, -1.0, -1.0])
    action_high = np.array([1.0, 1.0, 1.0])
    env = ArmEnv(task, action_low, action_high, max_episode_steps=1000, render_mode=render_mode)
    return env
