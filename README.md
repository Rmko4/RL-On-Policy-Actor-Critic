# On-Policy Actor-Critic methods
An implementation of the following on-policy actor-critic methods: Advantage Actor-Critic (A2C), Proximal Policy Optimization (PPO). The implementation is based on the following papers: 
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## Examples of Trained Agents
### InvertedDoublePendulum-v4 (PPO ~ 300k frames)
https://github.com/Rmko4/RL-Policy-Gradients/assets/55834815/ecdc37f3-500f-4001-9d45-e821101f01ea

### HalfCheetah-v4 (PPO ~ 200k frames)
https://github.com/Rmko4/RL-Policy-Gradients/assets/55834815/20fa1ee7-0681-4437-9fa8-6233eedc8582

## Running the code
### Installation
To install all dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### Training
To train the agent, run the following command:
```bash
python source/train_agent.py [Training Options] [PPO Options]
```

#### Training Options:
- **--run_name** (*str*): Name of the run.
- **--algorithm** (*{A2C,PPO}*) : Type of algorithm to use for training.
- **--env_id** (*str*): Id of the environment to train on.
<br><br>
- **--perform_testing**: Whether to perform testing after training.
- **--log_video**: Whether to log video of agent's performance.
<br><br>
- **--max_epochs** (*int*) (default: 3): Maximum number of steps to train for.
- **--steps_per_epoch** (*int*): Number of steps to train for per epoch.
- **--num_envs** (*int*): Number of environments to train on.
- **--num_rollout_steps** (*int*): Number of steps to rollout policy for.
<br><br>
- **--optimizer** (*{Adam,RMSprop,SGD}*): Optimizer to use for training.
- **--learning_rate** (*float*): Learning rate for training.
- **--lr_decay** (*float*): Learning rate decay for training.
- **--weight_decay** (*float*): Weight decay (L2 regularization) for training.
- **--gamma** (*float*): Discount factor.
- **--gae_lambda** (*float*): Lambda parameter for Generalized Advantage Estimation (GAE).
- **--value_coef** (*float*): Coefficient for value loss.
- **--entropy_coef** (*float*): Coefficient for entropy loss.
- **--max_grad_norm** (*float*): Maximum gradient norm for clipping.
<br><br>
- **--init_std** (*float*): Initial standard deviation for policy.
- **--hidden_size** (*int*): Hidden size for policy.
- **--shared_extractor**: Whether to use a shared feature extractor for policy.

#### PPO Options:
- **--ppo_batch_size** (*int*): Batch size for Proximal Policy Optimization (PPO).
- **--ppo_epochs** (*int*): Number of epochs to train PPO for.
- **--ppo_clip_ratio** (*float*): Clip ratio for PPO.
- **--ppo_clip_anneal**: Whether to anneal the clip ratio for PPO.
