python source/train_agent.py `
--env_id InvertedDoublePendulum-v4 `
--run_name train0 `
--algorithm A2C `
--optimizer Adam `
--num_envs 8 `
--num_rollout_steps 8 `
--max_epochs 1000 `
--steps_per_epoch 50 `
--learning_rate 0.0007 `
--lr_decay 0.99 `
--gamma 0.99 `
--gae_lambda 0.95 `
--value_coef 1.0 `
--entropy_coef 0.001 `
--init_std 0.2 `
--hidden_size 512 `
--max_grad_norm 0.5 `
# --ppo_batch_size 64 `
# --ppo_epochs 10 `
# --ppo_clip_ratio 0.1