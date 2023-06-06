#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --job-name=RL-Catch
#SBATCH --mem=200GB
#SBATCH --profile=task

module load Python
module load cuDNN
module load CUDA

source ~/envs/drlenv/bin/activate

# Copy git repo to local
cp -r ~/RL-Policy-Gradients/ $TMPDIR
# cd to working directory (repo)
cd $TMPDIR/RL-Policy-Gradients/

python source/train_agent.py \
--env_id HalfCheetah-v4 \
--run_name train0 \
--algorithm PPO \
--optimizer Adam \
--num_envs 8 \
--num_rollout_steps 128 \
--max_epochs 200 \
--steps_per_epoch 5 \
--learning_rate 0.0005 \
--lr_decay 0.98 \
--gamma 0.99 \
--gae_lambda 0.95 \
--value_coef 1.0 \
--entropy_coef 0.001 \
--init_std 0.2 \
--hidden_size 512 \
--ppo_batch_size 64 \
--ppo_epochs 10 \
--ppo_clip_ratio 0.1 \
--ppo_clip_anneal


python source/train_agent.py \
--env_id HalfCheetah-v4 \
--run_name train0 \
--algorithm A2C \
--optimizer Adam \
--num_envs 8 \
--num_rollout_steps 64 \
--max_epochs 1000 \
--steps_per_epoch 50 \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--gamma 0.99 \
--gae_lambda 0.95 \
--value_coef 1.0 \
--entropy_coef 0.001 \
--init_std 0.2 \
--hidden_size 512 \
--max_grad_norm 0.5


python source/train_agent.py \
--env_id InvertedDoublePendulum-v4 \
--run_name train0 \
--algorithm PPO \
--optimizer Adam \
--num_envs 8 \
--num_rollout_steps 128 \
--max_epochs 1000 \
--steps_per_epoch 5 \
--learning_rate 0.0003 \
--lr_decay 0.95 \
--gamma 0.99 \
--gae_lambda 0.95 \
--value_coef 1.0 \
--entropy_coef 0.001 \
--init_std 0.2 \
--hidden_size 512 \
--ppo_batch_size 64 \
--ppo_epochs 10 \
--ppo_clip_ratio 0.1 \


python source/train_agent.py \
--env_id InvertedDoublePendulum-v4 \
--run_name train0 \
--algorithm RMSprop \
--optimizer Adam \
--num_envs 8 \
--num_rollout_steps 128 \
--max_epochs 1000 \
--steps_per_epoch 5 \
--learning_rate 0.0005 \
--lr_decay 0.992 \
--gamma 0.99 \
--gae_lambda 0.95 \
--value_coef 1.0 \
--entropy_coef 0.001 \
--init_std 0.2 \
--hidden_size 512 \
--ppo_batch_size 64 \
--ppo_epochs 10 \
--ppo_clip_ratio 0.1 \


python source/train_agent.py \
--env_id InvertedDoublePendulum-v4 \
--run_name train0 \
--algorithm A2C \
--optimizer Adam \
--num_envs 8 \
--num_rollout_steps 4 \
--max_epochs 1000 \
--steps_per_epoch 50 \
--learning_rate 0.0005 \
--lr_decay 0.992 \
--gamma 0.99 \
--gae_lambda 0.95 \
--value_coef 1.0 \
--entropy_coef 0.001 \
--init_std 0.2 \
--hidden_size 512 \
--max_grad_norm 0.5 \

