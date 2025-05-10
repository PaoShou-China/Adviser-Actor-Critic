# AAC: Adviser-Actor-Critic for Reinforcement Learning Control  
**Source Code Reference**:  
- Backbone & Baseline Implementation: [Metric-Residual Network](https://github.com/Cranial-XIX/metric-residual-network)  
- Benchmark Environment: [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)  

---

## Introduction  
The Adviser-Actor-Critic (AAC) framework addresses high-precision control challenges in reinforcement learning (RL) by integrating classical feedback control theory with adaptive RL methodologies. This hybrid architecture demonstrates exceptional performance in robotics applications requiring positional accuracy and robust goal-conditioned control.  

Key Contributions:  
- **Theoretical Unification**: Bridges classical control principles (e.g., error integration) with deep RL for enhanced trajectory optimization.  
- **Mentored Actor Architecture**: Introduces a parameterized Adviser module that refines actor outputs through adaptive action correction.  
- **Precision Validation**: Achieves ±0.01mm positional accuracy in physical simulations, outperforming conventional RL baselines.  
- **Domain Transfer**: Demonstrates robustness in real-world robotic systems via environment-agnostic policy learning.  

---

## Methodology  

### Control-Theoretic Adviser Design  
The Adviser module implements a parameterized control law inspired by PID controllers, defined by three tunable parameters:  
- **Proportional Gain** $K_p$: Regulates response to instantaneous state errors.  
- **Integral Gain** $K_i$: Accumulates historical errors to eliminate steady-state deviations.  
- **Smoothing Factor** $\sigma$: Modulates action update smoothness to suppress oscillations.  

This formulation operates within the Gymnasium-Robotics environment framework, enabling compatibility with robotic manipulation tasks requiring high-dimensional state observations and goal-conditioned policies.  

---

## Implementation  

### Training Protocol  
The agent is trained via the `train_agent` function in `main.py`. Critical hyperparameters include:  
- `epochs`: Total training iterations.  
- `batch_size`: Batch dimension for gradient updates.  
- `learn_start_size`: Minimum buffer size before policy optimization begins.  

Adviser behavior is decoupled between training and evaluation phases:  
```python
adviser_train = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_train_params)
adviser_eval = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_eval_params)
```

### QuickStart

To quickly get started and evaluate the performance improvement brought by the Adviser-Actor-Critic (AAC) framework, follow these simple steps:

##### Run

The main entry point is `main.py`. You can control the PI adviser parameters via `ki_range` and `sigma_range`, and adjust the number of episodes per model using the `episodes` parameter.

##### View Results

After execution, results will be saved under the `log/` directory in JSON format. Each file contains detailed metrics for every episode, comparing performance between:

- The **base controller** (no adviser)
- The **adviser-enhanced controller**

Key metrics include:

- Success rate
- Position and rotation errors (`d_pos`, `d_rot`)
- ITAE (Integral of Time-weighted Absolute Error)
- Improvement percentages

Example filenames：`log/FetchPick_pqe_[1.0, 0.1, 0.2]_None.json`; `HandManipulateBlockRotateParallel_pqe_[1.0, 0.0, 0.0]_[1.0, 0.6, 0.15].json`

To compare different adviser configurations or environments, simply modify the `parameter ranges` in `main.py` and `ENV_MAP` in `utils.py`, then re-run.

### Code Explanation

`advisers.py`: Implements controllers for robot control goal adjustment.

- Factory function `create_adviser()` to create different controllers.
- Three controller classes: `PositionController` (3D position PI control), `AttitudeController` (quaternion-based PI control), and `CombinedController` (combines both).



`evaluate.py`: Provides a framework for evaluating robot control strategies.

- Tools for calculating success rates, evaluating single episodes, and processing model files.
- `run_serial_evaluation()`: Evaluates all models in a directory using multiprocessing for speed.

  
`main.py`: Serves as the main entry point for evaluation.

- `run_serial_batch()`: Automates batch evaluation with parameter combinations.
- Main block defines parameter ranges and evaluates different environments (Fetch, Hand).



`utils.py`: Offers utility functions and classes for the system.

- Environment mapping and file utilities.
- Defines `Actor` network for action generation.
- Provides `Normalizer` for data normalization and `Agent` for evaluation.
- Functions for loading models, creating environments, and quaternion operations.

## Experimental Framework  

### Benchmarking Protocol  

### Baseline Comparison  
AAC's performance is benchmarked against the Metric-Residual Network (Cranial-XIX, 2022), a state-of-the-art RL architecture for precision control. Key differences include:  
| Feature                | AAC Framework               | Metric-Residual Network      |  
|------------------------|----------------------------|------------------------------|  
| Control Integration    | Explicit parameterized Adviser | Implicit residual learning    |  
| Action Refinement      | Hierarchical (Actor + Adviser) | Single-stage residual correction |  
---

## Installation Guide  
### Create conda environment with Python 3.10
conda create -n AAC python=3.10

conda activate AAC

### Install required packages
pip install tianshou==1.1.0

pip install gymnasium-robotics==1.2.4

---  
*This implementation extends methodologies from [Cranial-XIX/metric-residual-network](https://github.com/Cranial-XIX/metric-residual-network) and [Farama-Foundation/Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics).*
