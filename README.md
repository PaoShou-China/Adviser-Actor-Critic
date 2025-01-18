# AAC: Adviser-Actor-Critic for Reinforcement Learning Control
AAC addresses the challenges of high-precision control tasks in reinforcement learning by integrating feedback control theory with adaptive RL capabilities. 

## Introduction

AAC is a high-precision reinforcement learning model that

-   integrates feedback control theory with adaptive RL capabilities for precise control tasks，
    
-   introduces an Adviser to mentor the actor, refining actions for improved goal attainment，
        
-   outperforms standard RL algorithms in precision-critical, goal-conditioned tasks，
    
-   demonstrates exceptional precision, reliability, and robustness in real-world applications like robotics.


## Experiments

### Training

To train the agent, use `train_agent`, the main function in `main.py`. Define key training parameters such as the number of epochs (`epochs`), batch size (`batch_size`), and the minimum experience buffer size (`learn_start_size`) required before training begins. Configure the Adviser by setting `adviser_train_params` and `adviser_eval_params` to guide the agent during training and evaluation. Start training by calling `train_agent(args)` with the configured parameters; the functioning will log progress, save model weights periodically, and evaluate performance at specified intervals. Monitor training metrics (e.g., scores, errors) and adjust parameters as needed to ensure efficient training and high-precision performance in goal-conditioned tasks.

### Adviser

To use the Adviser, call  `create_adviser(dim, dt, params)`  from  `advisers.py`, where:

-   `dim`  is the dimension of the error vector (e.g.,  `env.desired_goal_dim`).
    
-   `dt`  is the sampling time (e.g.,  `env.sim.dt`).
    
-   `params`  is a list of parameters (`kp`,  `ki`,  `kd`,  `T`):
    
    -    **PID controller**  is created with  `kp`,  `ki`,  `kd`, and  `T`.
                

Example:
```
adviser_train = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_train_params)  # Training Adviser
adviser_eval = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_eval_params)    # Evaluation Adviser
```
