import argparse
from evaluate import run_serial_evaluation
from utils import ENV_MAP


def _parse_float_list(value):
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("provide at least one number")
    return values


def run_serial_batch(env_name, pos_params=None, rot_params=None, episodes=10, model_dir=None):
    """Evaluate all policy checkpoints for an environment and adviser setting."""
    if env_name.startswith("Fetch"):
        combinations = [[p, None] for p in (pos_params or [])]
    elif "Rotate" in env_name:
        combinations = [[[1.0, 0.0, 0.0], r] for r in (rot_params or [])]
    elif "Full" in env_name:
        combinations = [[p, r] for p, r in zip(pos_params or [], rot_params or [])]
    else:
        combinations = []

    if not combinations:
        raise ValueError(f"No adviser parameter combinations for environment {env_name}")

    for idx, (pos, rot) in enumerate(combinations, 1):
        selected_model_dir = model_dir or f"model/{env_name}"
        run_serial_evaluation(
            model_dir=selected_model_dir,
            num_episodes=episodes,
            pos_params=pos,
            rot_params=rot,
        )
        print(f"[{env_name}]_{idx} completed!")

    print(f"\n[{env_name}] Batch completed!")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate saved goal-conditioned policies with and without an adviser."
    )
    parser.add_argument(
        "--env",
        nargs="+",
        choices=sorted(ENV_MAP),
        help="Environment(s) to evaluate. Defaults to all environments in utils.ENV_MAP.",
    )
    parser.add_argument(
        "--model-dir",
        help="Checkpoint directory. Defaults to model/<environment> for each environment.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Evaluation episodes per model (default: 200).",
    )
    parser.add_argument(
        "--ki-range",
        type=_parse_float_list,
        default=[0.3, 0.5, 1.0],
        help="Comma-separated adviser integral gains (default: 0.3,0.5,1.0).",
    )
    parser.add_argument(
        "--sigma-range",
        type=_parse_float_list,
        default=[0.1, 0.2, 0.3],
        help="Comma-separated adviser thresholds (default: 0.1,0.2,0.3).",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=1.0,
        help="Fixed proportional gain used in adviser parameter combinations (default: 1.0).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be a positive integer")

    params = [[args.kp, ki, sigma] for ki in args.ki_range for sigma in args.sigma_range]
    env_names = args.env or list(ENV_MAP)

    for env_name in env_names:
        print(f"\nStarting evaluation for {env_name}...")
        if env_name.startswith("Fetch"):
            run_serial_batch(
                env_name, pos_params=params, episodes=args.episodes, model_dir=args.model_dir
            )
        elif "Rotate" in env_name:
            run_serial_batch(
                env_name, rot_params=params, episodes=args.episodes, model_dir=args.model_dir
            )
        elif "Full" in env_name:
            run_serial_batch(
                env_name, pos_params=params, rot_params=params,
                episodes=args.episodes, model_dir=args.model_dir
            )
        else:
            print(f"Skipping {env_name}: environment type is not classified for adviser evaluation.")
        print(f"Completed evaluation for {env_name}")

    print("\nAll evaluations completed!")


if __name__ == "__main__":
    main()
