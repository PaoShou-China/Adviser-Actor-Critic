import numpy as np
import copy
from .simple_replay_buffer import SimpleReplayBuffer


class HindsightReplayBuffer(SimpleReplayBuffer):
    def __init__(self, env, max_replay_buffer_size, n_sampled_goals=4, goal_strategy='future'):
        """
        Initializes a Hindsight Replay Buffer used for implementing Hindsight Experience Replay (HER).

        Parameters:
            env (GoalEnv): The environment that provides the goal specifications.
            max_replay_buffer_size (int): The maximum size of the replay buffer.
            n_sampled_goals (int): Number of goals to be substituted per transition.
            goal_strategy (str): Strategy for sampling goals ('none','future' or 'final').
        """
        action_dim = env.action_dim
        observation_dim = (env.observation_dim +
                           env.achieved_goal_dim +
                           env.desired_goal_dim)
        max_replay_buffer_size = int(max_replay_buffer_size)
        super().__init__(action_dim, observation_dim, max_replay_buffer_size)

        self.env = env
        self.n_sampled_goals = n_sampled_goals
        self.goal_strategy = goal_strategy
        self.episode = []

    def _sample_strategy_goal(self, episode, start_idx, strategy='future'):
        """
        Samples a goal based on the specified strategy.

        Parameters:
            episode (list): List of transitions in the episode.
            start_idx (int): Current time step within the episode.
            strategy (str): Sampling strategy ('future' or 'final').

        Returns:
            dict: A new goal to substitute the original desired goal.
        """
        if strategy == 'future':
            future_indices = np.arange(start_idx + 1, len(episode))
            transition_idx = np.random.choice(future_indices)
            transition = episode[transition_idx]
        elif strategy == 'final':
            transition = episode[-1]
        else:
            raise NotImplementedError("Sampling strategy not implemented.")

        return transition[0]['achieved_goal']

    def add_hindsight_episode(self, episode):
        """
        Adds an episode to the replay buffer and applies HER.

        Parameters:
            episode (list): List of transitions (o, a, r, o2, d) where o is a dictionary.
        """
        for t, transition in enumerate(episode):
            obs, action, reward, next_obs, done = transition

            # Augment observations with desired goals
            augmented_observation = np.concatenate((obs['observation'], obs['achieved_goal'],
                                                    obs['achieved_goal'] - obs['desired_goal']))
            next_augmented_observation = np.concatenate((next_obs['observation'], next_obs['achieved_goal'],
                                                         next_obs['achieved_goal'] - next_obs['desired_goal']))

            # Add the original transition to the buffer
            self.add_sample(augmented_observation, action, reward, done, next_augmented_observation)

            # Apply HER if it is set
            if self.goal_strategy == 'none':
                continue
            elif t == len(episode) - 1 or self.goal_strategy == 'final':
                strategy = 'final'
            elif self.goal_strategy == 'future':
                strategy = 'future'
            else:
                raise "Wrong goal strategy!"

            sampled_goals = [
                self._sample_strategy_goal(episode, t, strategy)
                for _ in range(self.n_sampled_goals)
            ]

            for goal in sampled_goals:
                # Deep copy the transition to avoid mutating the original
                obs_copy, action_copy, reward_copy, next_obs_copy, done_copy = copy.deepcopy(transition)

                # Update the desired goal in both observation and next observation
                obs_copy['desired_goal'] = goal
                next_obs_copy['desired_goal'] = goal

                # Recalculate the reward with the new goal
                augmented_observation = np.concatenate((obs_copy['observation'], obs_copy['achieved_goal'],
                                                        obs_copy['achieved_goal'] - obs_copy['desired_goal']))
                next_augmented_observation = np.concatenate(
                    (next_obs_copy['observation'], next_obs_copy['achieved_goal'],
                     next_obs_copy['achieved_goal'] - next_obs_copy['desired_goal']))
                reward = self.env.task.compute_reward(action_copy, next_obs_copy['observation'],
                                                      next_obs_copy['achieved_goal'], next_obs_copy['desired_goal'],
                                                      done_copy)

                # Add the modified transition to the buffer
                self.add_sample(augmented_observation, action_copy, reward, done_copy, next_augmented_observation)

        # Terminate the episode in the buffer
        self.terminate_episode()

    def terminate_episode(self):
        self.episode.clear()

    def put_data(self, transition, done):
        self.episode.append(transition)
        if done: self.add_hindsight_episode(self.episode)
