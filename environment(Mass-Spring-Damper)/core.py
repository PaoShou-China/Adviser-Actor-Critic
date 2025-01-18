from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod


class Simulator(ABC):
    """A simulator class that provides state transition based on a given derivative function.

    This class supports manual single-step RK34 numerical integration, state saving, restoration,
    and cleanup. It also tracks task goals associated with each saved state.

    Args:
        dt (float, optional): Time step size. Defaults to 0.1.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.saved_states = {}
        self.saved_goals = {}
        self.state_counter = 0
        self.state = None
        self.goal = None

    def reset(self, initial_state: np.ndarray = None) -> np.ndarray:
        """Reset the simulator to an initial state.

        Args:
            initial_state (np.ndarray, optional): The initial state vector. Defaults to None.

        Returns:
            np.ndarray: The initial state after reset.
        """
        if initial_state is not None:
            self.state = initial_state
        return self.state

    def save_state(self) -> int:
        """Save the current state of the simulator and the associated task goal.

        Returns:
            int: State unique identifier.
        """
        state_id = self.state_counter
        self.saved_states[state_id] = self.state.copy()
        self.saved_goals[state_id] = self.goal
        self.state_counter += 1
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state and the associated task goal from the saved state.

        Args:
            state_id (int): State unique identifier.
        """
        if state_id in self.saved_states:
            self.state = self.saved_states[state_id].copy()
            self.goal = self.saved_goals[state_id]
        else:
            raise ValueError(f"No saved state found with ID {state_id}")

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state and its associated task goal.

        Args:
            state_id (int): State unique identifier.
        """
        if state_id in self.saved_states:
            del self.saved_states[state_id]
            self.saved_goals.pop(state_id, None)
        else:
            raise ValueError(f"No saved state found with ID {state_id}")

    def close(self) -> None:
        """Clean up any resources held by the simulator.
        """
        self.saved_states.clear()
        self.saved_goals.clear()
        self.state_counter = 0
        self.goal = None

    @abstractmethod
    def derivative_func(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """The dynamic model of system.
        Args:
            state (np.ndarray): The state vector.
            action (np.ndarray): The action vector.
        Returns:
            np.ndarray: The derivative of state vector.
        """
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        """The dynamic model of system.
        Returns:
            int: The dimension of action vector.
        """
        pass

    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance the state one time step forward using RK34 numerical integration.

        Args:
            action (np.ndarray): The control input or action.

        Returns:
            np.ndarray: The new state after applying the action.
        """
        # Calculate the k values
        k1 = self.derivative_func(self.state, action)
        k2 = self.derivative_func(self.state + self.dt * k1 / 2, action)
        k3 = self.derivative_func(self.state + self.dt * k2 / 2, action)
        k4 = self.derivative_func(self.state + self.dt * k3, action)

        # Update the state using the weighted average of the k values
        self.state += self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return self.state


class RobotTask(ABC):
    """Base class for goal-oriented environments.

    Args:
        sim (Simulator): Simulation instance.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, sim: 'Simulator') -> None:
        super().__init__()
        self.sim = sim

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Returns the observation associated with the current state of the task."""
        pass

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Returns the achieved goal."""
        pass

    def get_desired_goal(self) -> np.ndarray:
        """Returns the current goal."""
        if self.sim.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.sim.goal.copy()

    def step(self, action: np.ndarray):
        """Executes one step in the environment with the given action."""
        self.sim.step(action)

    @abstractmethod
    def reset(self, options: dict):
        """Resets the task, samples a new goal, and returns the initial observation."""
        pass

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Determines whether the achieved goal matches the desired goal."""
        pass

    @abstractmethod
    def is_failure(self, observation: np.ndarray, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Determines whether the robot exceed the limits."""
        pass

    @abstractmethod
    def compute_reward(self, action: np.ndarray, observation: np.ndarray, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, terminated: bool) -> float:
        """Computes the reward associated with the achieved and desired goals, observation and action."""
        pass

    def render(self, mode='human'):
        """Renders the environment."""
        pass

    def close(self):
        """Closes the environment."""
        pass


class RobotEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        task (RobotTask): The combined robot and task instance.
        max_episode_steps (int, optional): Maximum number of steps per episode. Defaults to 100.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, task: 'RobotTask', action_low: np.ndarray, action_high: np.ndarray,
                 max_episode_steps: int = 100, render_mode: str = None) -> None:
        assert render_mode in self.metadata["render_modes"] or render_mode is None
        self.render_mode = render_mode
        self.sim = task.sim
        self.metadata["render_fps"] = 1.0 / self.sim.dt
        self.screen = None
        self.task = task
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.action_low, self.action_high = np.array(action_low), np.array(action_high)
        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=desired_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=achieved_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.sim.get_action_dim(),), dtype=np.float32)
        self._saved_goal = dict()  # For state saving and restoring

    def _get_obs(self) -> dict:
        robot_obs = self.task.get_observation().astype(np.float32)  # combined observation
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        desired_goal = self.task.get_desired_goal().astype(np.float32)
        return {
            "observation": robot_obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple:
        super().reset(seed=seed, options=options)
        self.task.reset(options)
        self.current_step = 0
        observation = self._get_obs()
        info = {"is_success": False, "is_failure": False}
        return observation, info

    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.sim.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.sim.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> tuple:
        mapped_action = self.action_mapping(action)
        self.task.step(mapped_action)
        self.current_step += 1
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        is_success = self.task.is_success(observation["achieved_goal"], observation["desired_goal"])
        is_failure = self.task.is_failure(observation["observation"], observation["achieved_goal"],
                                          observation["desired_goal"])
        terminated = is_success or is_failure
        truncated = self.current_step >= self.max_episode_steps
        # if terminated or truncated: print(f"final error: {self.sim.state - self.sim.goal}")
        info = {"is_success": is_success, "is_failure": is_failure}
        reward = self.task.compute_reward(action, observation['observation'], observation["achieved_goal"],
                                          observation["desired_goal"], terminated)
        return observation, reward, terminated, truncated, info

    def action_mapping(self, action: np.ndarray) -> np.ndarray:
        mapped_action = self.action_low + (action - (-1.0)) * (
                (self.action_high - self.action_low) / 2.0)
        mapped_action = np.clip(mapped_action, self.action_low, self.action_high)
        return mapped_action

    def close(self) -> None:
        self.sim.close()

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    @property
    def observation_dim(self):
        return self.observation_space['observation'].shape[0]

    @property
    def achieved_goal_dim(self):
        return self.observation_space['achieved_goal'].shape[0]

    @property
    def desired_goal_dim(self):
        return self.observation_space['desired_goal'].shape[0]
