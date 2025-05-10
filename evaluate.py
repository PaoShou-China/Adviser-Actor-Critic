import os
import json
import multiprocessing
import numpy as np
from utils import *
from advisers import create_adviser
from tqdm import tqdm

dt = 0.04
base_results_cache = {}    # Global cache to store base controller evaluation results for each model


def calculate_success_rate(results):
    """
    Calculate the success rate from a list of dictionaries containing 'success' field.
    Returns a float formatted to two decimal places (for calculation).
    """
    success_count = sum(res['success'] for res in results)
    total = len(results)
    success_rate = (success_count / total) * 100 if total > 0 else 0.0
    return success_rate


def evaluate_single_episode(args):
    """
    Execute a single episode of evaluation.

    Args:
        args: Tuple containing (agent, env, seed, episode_idx, adviser)

    Returns:
        Dictionary with metrics including reward, distance errors, success flag, ITAE scores.
    """
    agent, env, seed, episode_idx, adviser = args
    np.random.seed(seed + episode_idx)
    obs, _ = env.reset(seed=seed)

    achieved_goals = []
    desired_goals = []
    errors = []

    time_steps = 0
    itae_pos = 0.0
    itae_rot = 0.0
    total_reward = 0.0
    terminated, truncated = False, False

    while not terminated and not truncated:
        current_ag = obs['achieved_goal'].copy().astype(float)
        current_dg = obs['desired_goal'].copy().astype(float)
        achieved_goals.append(current_ag.tolist())
        desired_goals.append(current_dg.tolist())
        try:
            d_pos, d_rot = env.unwrapped._goal_distance(obs['achieved_goal'], obs['desired_goal'])
        except:
            d_pos, d_rot = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']).astype(float), 0.0

        # Update ITAE metrics
        time_steps += 1
        itae_pos += time_steps * dt * d_pos
        itae_rot += time_steps * dt * d_rot

        action = agent.select_action(adviser(obs), stochastic=False)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward


    success_flag = bool(info.get('is_success', False))

    return {
        'total_reward': float(total_reward),
        'd_pos': float(d_pos),
        'd_rot': float(d_rot),
        'success': success_flag,
        'itae_pos': float(itae_pos),
        'itae_rot': float(itae_rot)
    }


def process_model_file(args):
    """
    Process a single model file: load model, evaluate using both base and custom controllers,
    and compute performance improvements.

    Args:
        args: Tuple containing (results_dir, filename, num_episodes, pos_params, rot_params)

    Returns:
        List of per-episode performance data along with summary statistics.
    """
    def _calc_improvement(base, current):
        if base > 0:
            return round(((base - current) / base) * 100, 2)
        return 0.0

    results_dir, filename, num_episodes, pos_params, rot_params = args
    cache_key = f"{results_dir}/{filename}"   # Unique key for caching base results

    env_name = filename.split('_')[0]
    seed = parse_seed_from_filename(filename)
    filepath = os.path.join(results_dir, filename)

    agent = load_policy_model(filepath)
    env = create_environment(env_name)

    # Evaluate base controller only once per model and cache result
    if cache_key not in base_results_cache:
        base_adviser = create_adviser(
            dt,
            pos_params=[1.0, 0.0, 0.0] if pos_params is not None else None,
            rot_params=[1.0, 0.0, 0.0] if rot_params is not None else None
        )

        with multiprocessing.Pool() as pool:
            base_tasks = [(agent, env, seed + i, i, base_adviser) for i in range(num_episodes)]
            base_results = list(tqdm(pool.imap(evaluate_single_episode, base_tasks), total=num_episodes,
                                     desc=f"{filename} (Base)"))
        base_results_cache[cache_key] = base_results

    base_results = base_results_cache[cache_key]

    # Evaluate advisor controller
    adviser = create_adviser(dt, pos_params, rot_params)
    with multiprocessing.Pool() as pool:
        adv_tasks = [(agent, env, seed + i, i, adviser) for i in range(num_episodes)]
        adv_results = list(tqdm(pool.imap(evaluate_single_episode, adv_tasks), total=num_episodes,
                                desc=f"{filename} (Adv)"))

    output_data = []
    base_success_rate = calculate_success_rate(base_results)
    adv_success_rate = calculate_success_rate(adv_results)


    # Format and collect results per episode
    for ep, (base_res, adv_res) in enumerate(zip(base_results, adv_results)):
        # Convert numpy.float32 to float
        base_res_converted = {
            'total_reward': float(base_res['total_reward']),
            'd_pos': float(base_res['d_pos']),
            'd_rot': float(base_res['d_rot']),
            'success': base_res['success'],
            'itae_pos': float(base_res['itae_pos']),
            'itae_rot': float(base_res['itae_rot'])
        }

        adv_res_converted = {
            'total_reward': float(adv_res['total_reward']),
            'd_pos': float(adv_res['d_pos']),
            'd_rot': float(adv_res['d_rot']),
            'success': adv_res['success'],
            'itae_pos': float(adv_res['itae_pos']),
            'itae_rot': float(adv_res['itae_rot'])
        }

        record = {
            'env': env_name,
            'seed': seed,
            'episode': ep + 1,
            'base': base_res_converted,
            'adv': adv_res_converted,
            'imp': {
                'dpos': _calc_improvement(base_res['d_pos'], adv_res['d_pos']),
                'drot': _calc_improvement(base_res['d_rot'], adv_res['d_rot']),
                'd_itae_pos': _calc_improvement(base_res['itae_pos'], adv_res['itae_pos']),
                'd_itae_rot': _calc_improvement(base_res['itae_rot'], adv_res['itae_rot'])
            }
        }
        output_data.append(record)

    # Add overall summary at the end
    summary = {
        'summary': {
            'base_success_rate': base_success_rate,
            'adv_success_rate': adv_success_rate,
            'd_success_rate': _calc_improvement(adv_success_rate, base_success_rate)
        }
    }

    output_data.append(summary)  # 可以选择作为最后一个元素加入

    return output_data


def run_serial_evaluation(model_dir, num_episodes=10, pos_params=None, rot_params=None):
    """
    Run evaluation serially for all models in a given directory.

    Args:
        model_dir: Directory containing policy models (files ending with '_best.pt')
        num_episodes: Number of episodes to evaluate per model
        pos_params: Position PID parameters [Kp, Ki, Kd]
        rot_params: Rotation PID parameters [Kp, Ki, Kd]
    """

    alg = ['pqe', 'wn', 'asym']
    os.makedirs('log', exist_ok=True)

    # Prepare tasks
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_best.pt')]
    tasks = [(model_dir, f, num_episodes, pos_params, rot_params) for f in model_files]

    # Serial processing
    for i, task in enumerate(tasks):
        results = process_model_file(task)
        output_file = f"log/{task[0][6:]}_{alg[i]}_{task[3]}_{task[4]}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Processed {i + 1}/{len(tasks)} models")


