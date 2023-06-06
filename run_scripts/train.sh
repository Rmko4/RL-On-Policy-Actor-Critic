source/train_agent.py `
--env_id InvertedDoublePendulum-v4 `
--run_name train0 `
--num_envs 8 `
--num_rollout_steps 16 `
--max_epochs 1000 `
--steps_per_epoch 10 `
--learning_rate 0.0003 `
--gamma 0.99 `
--gae_lambda 0.95 `
--value_coef 0.5 `
--entropy_coef 0.001 `
--init_std 0.3 `
--hidden_size 128 `
--algorithm A2C `
--ppo_batch_size 64 `
--ppo_epochs 10 `
--ppo_clip_ratio 0.1
# --max_grad_norm 0.5 `