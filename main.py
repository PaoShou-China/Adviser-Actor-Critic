import os
import itertools
from evaluate import run_serial_evaluation
from tqdm import tqdm
import numpy as np
from utils import ENV_MAP


def run_serial_batch(env_name, pos_params=None, rot_params=None, episodes=10):
    """Simplified batch evaluation with automatic parameter combinations"""
    # Generate parameter combinations
    combinations = []
    if env_name.startswith('Fetch'):
        combinations = [[p, None] for p in pos_params]
    elif 'Rotate' in env_name:
        combinations = [[[1.0, 0.0, 0.0], r] for r in rot_params]
    elif 'Full' in env_name:
        combinations = [[p, r] for p, r in zip(pos_params, rot_params)]

    # Run evaluations
    for idx, (pos, rot) in enumerate(combinations, 1):
        # Execute single evaluation
        run_serial_evaluation(
            model_dir=f'model/{env_name}',
            num_episodes=episodes,
            pos_params=pos,
            rot_params=rot,
        )
        print(f"[{env_name}]_{idx} completed!")

    print(f"\n[{env_name}] Batch completed!")


# Usage Examples
if __name__ == "__main__":
    # Define parameter ranges
    ki_range = [0.3, 0.5, 1.0]
    sigma_range = [0.1, 0.2, 0.3]

    # Generate all parameter combinations
    param_combinations = []
    for ki in ki_range:
        for sigma in sigma_range:
            param_combinations.append([1.0, ki, sigma])

    # Evaluate each environment
    for env_name in ENV_MAP.keys():
        print(f"\nStarting evaluation for {env_name}...")

        if env_name.startswith('Fetch'):
            # Fetch environments use position control
            run_serial_batch(env_name,
                             pos_params=param_combinations,
                             rot_params=None,
                             episodes=200)
        elif 'Rotate' in env_name:
            # Rotate environments use rotation control
            run_serial_batch(env_name,
                             pos_params=None,
                             rot_params=param_combinations,
                             episodes=200)
        elif 'Full' in env_name:
            # Full environments use both position and rotation control
            run_serial_batch(env_name,
                             pos_params=param_combinations,
                             rot_params=param_combinations,
                             episodes=200)

        print(f"Completed evaluation for {env_name}")

    print("\nAll evaluations completed!")
