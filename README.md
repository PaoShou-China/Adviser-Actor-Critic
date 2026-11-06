![AAC — Adviser-Actor-Critic](assets/hero.svg)

# AAC: Adviser-Actor-Critic for Reinforcement Learning Control

[Project homepage](./) · [Quick start](#quick-start) · [Code](https://github.com/PaoShou-China/Adviser-Actor-Critic)

This repository evaluates saved goal-conditioned policies using position and quaternion attitude advisers in Gymnasium Robotics environments. The current `main.py` is an **evaluation entry point**; it does not train policies.

## What it does

- `PositionController`: position PI correction.
- `AttitudeController`: quaternion-based attitude PI correction.
- `CombinedController`: applies both corrections.
- Batch comparison of baseline and adviser parameter settings across supported environments.

The environment mapping is defined in `utils.py`. References: [Metric-Residual Network](https://github.com/Cranial-XIX/metric-residual-network) and [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics).

## Installation

The Conda specification is `aac_environment.yml`:

```bash
conda env create -f aac_environment.yml
conda activate aac
```

## Quick start

Checkpoints are read from `model/<environment>/` by default. The file name must end with `_best.pt`, contain a seed token such as `_sd100_`, and begin with the environment name.

```bash
python main.py --env FetchReach --episodes 10
```

To evaluate all environments using the script defaults:

```bash
python main.py
```

Options include `--model-dir`, `--kp`, `--ki-range`, and `--sigma-range`. Use `python main.py --help` for the current arguments. Results are written under `log/` as JSON episode records and summaries.

## Repository map

- `main.py`: batch evaluation CLI.
- `evaluate.py`: episode evaluation and result aggregation.
- `advisers.py`: position, attitude, and combined adviser implementations.
- `utils.py`: supported environment map, policy loading, and helpers.
- `model/`: example saved policy checkpoints.

## Reproducibility notes

The evaluation seed is derived from checkpoint filenames. Keep the model filename convention intact when adding checkpoints. Evaluation requires the Gymnasium Robotics/MuJoCo environment and compatible installed dependencies. The saved checkpoints are examples; training code is not included in this repository.
