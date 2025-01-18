import numpy as np
import torch


def convert_to_tensor(device, *values):
    """
    Converts the given values to tensors and moves them to the specified device.

    :param device: Device to move tensors to.
    :param values: Values to convert to tensors.
    :return: List of tensors.
    """
    return [torch.tensor(v, dtype=torch.float32).to(device) for v in values]


def cat_obs_dict(obs_dict):
    """
    Concatenates the observation, achieved goal, and the difference between achieved and desired goals.

    :param obs_dict: Dictionary containing observation components.
    :return: Concatenated observation array.
    """
    return np.concatenate([
        obs_dict["observation"],
        obs_dict["achieved_goal"],
        obs_dict["achieved_goal"] - obs_dict["desired_goal"]
    ])


def evaluate_agent(agent, env, evaluator):
    """
    Evaluates the performance of an agent in the given environment.

    :param agent: Agent to evaluate.
    :param env: Environment to evaluate the agent in.
    :param evaluator: Evaluator function for processing observations.
    :return: Total score obtained during the evaluation.
    """
    init_state = np.array([0.0, 0.0, 0.0, 0.0])
    goals = [np.array([1.2, 1.2, 0.0, 0.0]),
             np.array([1.2, 0.0, 0.0, 0.0]),
             np.array([0.0, 1.2, 0.0, 0.0])]
    total_score = 0.0
    total_error = 0.0
    for goal in goals:
        options = {'init_state':init_state, 'goal':goal}
        obs_dict, _ = env.reset(options=options)
        evaluator.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action, _ = agent.get_action(evaluator(cat_obs_dict(obs_dict)))
            action = action.cpu().detach().numpy().flatten()
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            obs_dict = next_obs_dict
            total_score += reward
        total_error += np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal'])

    return total_score / len(goals), total_error / len(goals)
