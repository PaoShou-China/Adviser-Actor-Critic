import torch
from sac import SAC
from utils import cat_obs_dict, evaluate_agent
from advisers import create_adviser 
import os
from argparse import ArgumentParser
from environment.environment import make_env
import multiprocessing as mp
import numpy as np
import random

device = 'cpu'  # only cpu, because cuda is not available.

def get_args(goal_strategy, adviser_train_params, adviser_eval_params, seed):
    """
    Parse and return command-line arguments for training.
    """
    log_dir = f'log/{goal_strategy}/{"_".join(map(str, adviser_train_params))}_{"_".join(map(str, adviser_eval_params))}/{seed}'
    os.makedirs(log_dir, exist_ok=True)
    parser = ArgumentParser('parameters')
    parser.add_argument("--algo", type=str, default='sac', help='only sac')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--goal_strategy', type=str, default=goal_strategy,
                        help="strategy of HER: none, future or final")
    parser.add_argument("--n_sampled_goals", type=int, default=4,
                        help="Number of goals to be substituted per transition")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--epochs', type=int, default=52, help='number of epochs, (default: 100)')
    parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=100, help='save interval(default: 10)')
    parser.add_argument("--print_interval", type=int, default=1, help='print interval(default : 20)')
    parser.add_argument("--alpha_init", type=float, default=0.2, help="initial value for alpha")
    parser.add_argument("--gamma", type=float, default=0.995, help="discount factor")
    parser.add_argument("--q_lr", type=float, default=5e-4, help="learning rate for Q networks")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="learning rate for actor network")
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="learning rate for alpha")
    parser.add_argument("--soft_update_rate", type=float, default=0.005, help="tau for soft updates")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimension of hidden layers")
    parser.add_argument("--learn_start_size", type=int, default=1000,
                        help="minimum replay buffer size before learning starts")
    parser.add_argument("--memory_size", type=int, default=int(1e+6), help="maximum replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--layer_num", type=int, default=3, help="number of hidden layers")
    parser.add_argument("--activation_function", type=str, default='selu', help="activation function")
    parser.add_argument("--last_activation", type=str, default=None, help="activation function for last layer")
    parser.add_argument("--trainable_std", type=bool, default=True, help="whether the standard deviation is trainable")
    parser.add_argument('--adviser_train_params', nargs=4, type=float, default=adviser_train_params,
                        help='Kp, Ki, Kd, T for training adviser') 
    parser.add_argument('--adviser_eval_params', nargs=4, type=float, default=adviser_eval_params,
                        help='Kp, Ki, Kd, T for evaluation adviser') 
    args = parser.parse_args()
    args.log_dir = log_dir
    args.seed = seed

    return args


# Training function
def train_agent(args):
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()

    # Instantiate Soft-Actor-Critic (SAC) agent
    agent = SAC(writer, device, env, args).to(device)
    if args.load != 'no':
        agent.load_state_dict(torch.load(os.path.join(args.log_dir, "model_weights")))

    # Initialize advisers using the create_adviser function
    adviser_train = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_train_params)
    adviser_eval = create_adviser(env.desired_goal_dim, env.sim.dt, args.adviser_eval_params)

    score_lst = []

    for n_epi in range(args.epochs):
        train_score = 0.0
        obs_dict, _ = env.reset(seed=args.seed)
        adviser_train.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            if args.render: env.render()
            action, _ = agent.get_action(adviser_train(cat_obs_dict(obs_dict)))
            action = action.cpu().detach().numpy().flatten()
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            transition = [obs_dict, action, reward, next_obs_dict, terminated]
            agent.put_data(transition, terminated or truncated)
            obs_dict = next_obs_dict
            train_score += reward
            if agent.data.size > args.learn_start_size:
                agent.train_net(args.batch_size, n_epi)

        score_lst.append(train_score)

        if args.tensorboard:
            # Log the training score for each episode
            writer.add_scalar("score/train", train_score, n_epi)
            train_error = np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal'])
            writer.add_scalar("error/train", train_error, n_epi)

        if n_epi % args.print_interval == 0:
            eval_score, eval_error = evaluate_agent(agent, env, adviser_eval)
            print(f"episode: {n_epi}, avg score: {sum(score_lst) / len(score_lst):.2f}, eval score: {eval_score:.2f}")
            if args.tensorboard:
                # Log the evaluation score at each print interval
                writer.add_scalar("score/eval", eval_score, n_epi)
                writer.add_scalar("error/eval", eval_error, n_epi)
            score_lst = []

        if n_epi % args.save_interval == 0 and n_epi != 0:
            # Save model weights in the log directory
            model_weights_dir = os.path.join(args.log_dir, "model_weights")
            os.makedirs(model_weights_dir, exist_ok=True)
            torch.save(agent.state_dict(), os.path.join(model_weights_dir, f'agent_{n_epi}.pth'))

    print("Training is done!")


if __name__ == '__main__':
    adviser_train_params_list = [
        [1.0, 0.0, 0.0, 0.1],  # naive
        [1.3, 0.0, 0.0, 0.1],  # PID1
        [1.3, 0.01, 0.01, 0.1],  # PID2
    ]
    adviser_eval_params_list = [
        [1.0, 0.0, 0.0, 0.1],  # naive
        [1.3, 0.0, 0.0, 0.1],  # PID1
        [1.3, 0.1, 0.1, 0.1],  # PID2
    ]

    max_workers = mp.cpu_count()

    with mp.Pool(processes=max_workers) as pool:
        all_args = []
        for goal_strategy in ['none', 'future']:
            for adviser_train_params in adviser_train_params_list:
                for adviser_eval_params in adviser_eval_params_list:
                    for seed in [1, 2, 3, 4, 5]:
                        args = get_args(goal_strategy, adviser_train_params, adviser_eval_params, seed)
                        all_args.append(args)

        pool.map(train_agent, all_args)

    print("All training tasks have been completed!")
