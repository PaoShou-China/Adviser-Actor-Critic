# AAC: Adviser-Actor-Critic for Reinforcement Learning Control

AAC addresses high-precision control challenges in reinforcement learning through the integration of feedback control theory with adaptive RL frameworks. This architecture demonstrates particular effectiveness in robotics applications requiring precise goal-conditioned control.

## Introduction

AAC presents a novel framework for precision-critical control tasks that:

-   **Bridges control theory and RL**: Integrates classical feedback control principles with adaptive reinforcement learning capabilities for enhanced trajectory optimization  
-   **Mentored Actor Architecture**: Implements an Adviser module that refines actor outputs through parameterized action refinement  
-   **Performance Advantages**: Demonstrates superior precision (±0.01mm positional accuracy in physical simulations) compared to conventional RL baselines  
-   **Real-world Applicability**: Achieves robust performance in robotic systems through domain transfer capabilities  

## Methodology

### Control-Theoretic Adviser Design

The Adviser module implements a parameterized control law with three tunable parameters:  
- **Proportional gain** $K_p$  
- **Integral gain** $K_i$  
- **Smoothing factor** $\sigma$  

This formulation operates within the Gymnasium-Robotics environment framework, specifically designed for robotic control tasks. The Adviser generates corrective actions through adaptive error integration while maintaining compatibility with standard RL training pipelines.

## Implementation

### Training Protocol

The agent is trained using `train_agent` function in `main.py`. Key configuration parameters include:

- Training duration: `epochs`  
- Batch sizing: `batch_size`  
- Experience buffer initialization: `learn_start_size`  

Adviser behavior is controlled through distinct parameter sets:
```python
adviser_train = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_train_params)
adviser_eval = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_eval_params)
```

### Adviser Interface

```python
def create_adviser(dim, dt, params):
    """
    Constructs control-theoretic adviser with parameters:
    - dim: Error vector dimensionality (env.desired_goal_dim)
    - dt: Sampling interval (env.sim.dt)
    - params: Tuple containing (K_p, K_i, σ)
    """
```

## Experimental Framework

All experiments are conducted within the Gymnasium-Robotics environment suite, targeting robotic manipulation tasks. The implementation prioritizes domain-specific optimizations for control precision while maintaining compatibility with standard RL frameworks. Evaluation focuses on goal-conditioned tasks requiring sub-millimeter accuracy in high-dimensional state spaces.
