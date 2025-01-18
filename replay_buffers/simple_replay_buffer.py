import numpy as np
import abc


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    An abstract base class used to save and replay data for reinforcement learning algorithms.

    This class defines the interface for replay buffers, which are essential components in
    many reinforcement learning frameworks. They allow for the storage of experiences and
    the ability to sample from them randomly to break the temporal correlation of the data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation, terminal, **kwargs):
        """
        Adds a single transition tuple to the replay buffer.

        Parameters:
            observation: The current observation/state.
            action: The action taken.
            reward: The reward received.
            next_observation: The next observation/state.
            terminal: Whether the episode has ended.
            **kwargs: Additional information such as agent info and environment info.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Notifies the replay buffer that the episode has ended.

        This method allows for any necessary bookkeeping or cleanup related to the end of an episode.
        """
        pass

    @property
    @abc.abstractmethod
    def size(self):
        """
        Returns the number of unique items that can be sampled from the replay buffer.

        Returns:
            int: The number of unique items in the buffer.
        """
        pass

    def add_path(self, path):
        """
        Adds a full path (sequence of transitions) to the replay buffer.

        This method iterates over each transition in the given path and adds it to the buffer.
        It is responsible for handling the termination of the episode.

        Parameters:
            path (dict): A dictionary containing sequences of observations, actions, rewards,
                         next observations, terminals, agent infos, and environment infos.
                         Typically structured as output by a rollout function.
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"]
        )):
            self.add_sample(
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info=agent_info,
                env_info=env_info
            )
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Returns a batch of samples from the replay buffer.

        Parameters:
            batch_size (int): The number of samples to include in the batch.

        Returns:
            dict: A batch of samples.
        """
        pass

class SimpleReplayBuffer(ReplayBuffer):
    """
    A simple replay buffer for storing transitions experienced by an agent.

    This buffer stores observations, actions, rewards, terminals, and next observations.
    It supports adding samples and retrieving random batches for training.
    """

    def __init__(self, action_dim, observation_dim, max_replay_buffer_size):
        """
        Initializes the replay buffer with specified action and observation dimensions and a maximum buffer size.

        Parameters:
            action_dim (int): The dimensionality of the action space.
            observation_dim (int): The dimensionality of the observation space.
            max_replay_buffer_size (int): The maximum capacity of the replay buffer.
        """
        super().__init__()

        self._action_dim = action_dim
        self._observation_dim = observation_dim
        self._max_buffer_size = int(max_replay_buffer_size)

        # Allocate memory for the replay buffer arrays
        self._observations = np.zeros((self._max_buffer_size, self._observation_dim))
        self._next_obs = np.zeros((self._max_buffer_size, self._observation_dim))
        self._actions = np.zeros((self._max_buffer_size, self._action_dim))
        self._rewards = np.zeros((self._max_buffer_size, 1))
        self._terminals = np.zeros((self._max_buffer_size, 1))

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
        """
        Adds a single transition to the replay buffer.

        Parameters:
            observation (np.array): The current observation.
            action (np.array): The action taken.
            reward (float): The reward received.
            terminal (bool): Whether the episode has ended.
            next_observation (np.array): The next observation.
            **kwargs: Additional arguments (not used).
        """
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        """
        Terminates the current episode in the replay buffer.
        This method does nothing in the current implementation.
        """
        pass

    def _advance(self):
        """
        Advances the internal pointer and updates the buffer size.
        """
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        """
        Returns a random batch of samples from the replay buffer.

        Parameters:
            batch_size (int): The number of samples to include in the batch.

        Returns:
            dict: A dictionary containing the batch of samples.
        """
        indices = np.random.randint(0, self._size, batch_size)
        return {
            'observations': self._observations[indices],
            'actions': self._actions[indices],
            'rewards': self._rewards[indices],
            'terminals': self._terminals[indices],
            'next_observations': self._next_obs[indices],
        }

    @property
    def size(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of unique items that can be sampled.
        """
        return self._size

