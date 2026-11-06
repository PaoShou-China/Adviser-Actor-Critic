# AAC: Adviser-Actor-Critic for Reinforcement Learning Control

This repository contains evaluation code for applying parameterized position and attitude advisers to saved goal-conditioned policies in Gymnasium Robotics environments. The current `main.py` is an **evaluation entry point**: it does not train policies. Checkpoints are expected under `model/<environment>/` by default.

References:
- Backbone and baseline: [Metric-Residual Network](https://github.com/Cranial-XIX/metric-residual-network)
- Environments: [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)

## What the evaluation does

For each selected environment, the script evaluates checkpoints both with a baseline adviser configuration and with each requested adviser parameter combination. It writes per-episode JSON records and summary success rates under `log/`.

The supported environment names are defined in `utils.py` (`ENV_MAP`). Checkpoint files must end in `_best.pt`; their filenames must include a seed token matching `_sd<number>_`. The environment name is inferred from the filename prefix before the first underscore.

The Adviser implementations are in `advisers.py`:
- `PositionController`: position PI correction.
- `AttitudeController`: quaternion-based attitude PI correction.
- `CombinedController`: applies position and attitude correction together.

## Installation

The repository includes a Conda environment specification:

```bash
conda env create -f aac_environment.yml
conda activate aac
```

The environment file pins the main Python packages and versions used for the evaluation code. For a new environment, review platform-specific packages before installation; the package list includes MuJoCo and rendering-related dependencies.

## Evaluate saved models

Evaluate every environment listed in `utils.ENV_MAP` with the default adviser parameter grid and 200 episodes per model:

```bash
python main.py
```

Evaluate only selected environments:

```bash
python main.py --env FetchReach HandManipulateBlockRotateZ
```

Use a custom checkpoint directory, episode count, and parameter grid:

```bash
python main.py --env FetchReach --model-dir model/FetchReach --episodes 50 \
  --ki-range 0.2,0.5 --sigma-range 0.1,0.3
```

The parameter combinations use `[Kp, Ki, sigma]`, with `Kp=1.0` by default. Set another proportional gain with `--kp`. For `Fetch*` environments, position Adviser parameters are varied; for names containing `Rotate`, attitude parameters are varied; for names containing `Full`, the same parameter grid is paired for both controllers.

Checkpoints are loaded from the environment's default directory `model/<environment>` unless `--model-dir` is supplied. When specifying one custom directory for multiple environments, ensure its checkpoints match the environments being evaluated.

Results are written beneath `log/` relative to the current working directory. Each JSON file contains per-episode baseline and Adviser metrics, including reward, terminal position/rotation errors, success, and ITAE, followed by an aggregate success-rate record.

## Repository files

- `main.py`: command-line evaluation driver and parameter sweep.
- `evaluate.py`: checkpoint loading, episode evaluation, metrics, and JSON output.
- `advisers.py`: position and attitude Adviser controllers.
- `utils.py`: environment mapping, policy loading, and shared utilities.
- `model/`: saved checkpoints used by the evaluation script.

## Scope and interpretation

The code evaluates saved policies; it does not implement a training loop in this repository's current entry point. Reported performance depends on the checkpoint, environment version, adviser parameters, and episode protocol. The README's usage commands describe the checked-in evaluation workflow; claims about physical accuracy, real-world transfer, or benchmark superiority require separate experimental evidence.
