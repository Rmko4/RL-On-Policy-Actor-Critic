import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training policy agent")

    parser.add_argument("--run_name", type=str, default="train",
                        help="Name of the run")
    parser.add_argument("--algorithm", type=str, default="A2C",
                        choices=["A2C", "PPO"],
                        help="Type of algorithm to use for training")
    
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of steps to train for")
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Number of steps to train for per epoch")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of environments to train on")
    parser.add_argument("--num_rollout_steps", type=int, default=5,
                        help="Number of steps to rollout policy for")
    
    parser.add_argument("--optimizer", type=str, default="RMSprop",
                        choices=["Adam", "RMSprop", "SGD"])
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=1.,
                        help="Lambda parameter for GAE")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Coefficient for value loss")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Coefficient for entropy loss")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--init_std", type=float, default=0.2,
                        help="Initial standard deviation for policy")
    

    # parser.add_argument("--log_video", action="store_true",
    #                     help="Whether to log video of agent's performance")
    # parser.add_argument("--batch_size", type=int, default=32,
    #                     help="Batch size for training")
    # parser.add_argument("--batches_per_step", type=int, default=1,
    #                     help="Number of batches to sample from replay buffer per agent step")
    # parser.add_argument("--epsilon_start", type=float, default=0.1,
    #                     help="Initial epsilon")
    # parser.add_argument("--epsilon_end", type=float, default=0.01,
    #                     help="Final epsilon")
    # parser.add_argument("--epsilon_decay_rate", type=float, default=1000,
    #                     help="Number of steps to decay epsilon over")
    # parser.add_argument("--buffer_capacity", type=int, default=1000,
    #                     help="Capacity of replay buffer")
    # parser.add_argument("--replay_warmup_steps", type=int, default=100,
    #                     help="Number of steps to warm up replay buffer")
    # parser.add_argument("--prioritized_replay", action="store_true",
    #                     help="Whether to use prioritized replay")
    # parser.add_argument("--prioritized_replay_alpha", type=float, default=None,
    #                     help="Alpha parameter for prioritized replay")
    # parser.add_argument("--prioritized_replay_beta", type=float, default=None,
    #                     help="Beta parameter for prioritized replay")
    # parser.add_argument("--target_net_update_freq", type=int, default=None,
    #                     help="Number of steps between target network updates")
    # parser.add_argument("--soft_update_tau", type=float, default=1e-3,
    #                     help="Tau for soft target network updates")
    # parser.add_argument("--double_q_learning", action="store_true",
    #                     help="Whether to use double Q-learning")
    # parser.add_argument("--hidden_size", type=int, default=128,
    #                     help="Number of hidden units in feedforward network.")
    # parser.add_argument("--n_filters", type=int, default=32,
    #                     help="Number of filters in convolutional network.")

    args = parser.parse_args()
    return args
