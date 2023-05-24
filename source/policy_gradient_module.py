import time
from typing import Any, Callable, Dict, Optional, Type

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from policy_network import ActorCriticPolicy
from pytorch_lightning import LightningModule
from rollout import RolloutAgent, RolloutBufferDataset, RolloutSample
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader

OPTIMIZERS: Dict[str, Type[torch.optim.Optimizer]] = {'Adam': Adam,
                                                      'RMSprop': RMSprop,
                                                      'SGD': SGD}


class PolicyGradientModule(LightningModule):
    def __init__(self,
                 env_id: str = 'Ant-v4',
                 algorithm: str = 'A2C',
                 steps_per_epoch: int = 20,
                 num_envs: int = 8,
                 num_rollout_steps: int = 5,
                 optimizer: str = 'RMSProp',
                 learning_rate: float = 1e-3,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 gae_lambda: float = 1.,
                 init_std: float = 1.,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters()

        # Makes envs in init already
        self.env = gym.vector.make(
            env_id, num_envs=num_envs, asynchronous=True)

        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.policy = ActorCriticPolicy(state_space=self.state_space,
                                        action_space=self.action_space,
                                        hidden_size=128,
                                        init_std=init_std)

        self.rollout_agent = RolloutAgent(self.env,
                                          self.policy,
                                          num_rollout_steps=num_rollout_steps,
                                          gamma=gamma,
                                          gae_lambda=gae_lambda)

        self.mse_loss = nn.MSELoss()
        self.batch_size = num_envs * num_rollout_steps

    def training_step(self, batch: RolloutSample, batch_idx: int) -> Tensor:
        # Run for one rollout sample

        # A2C
        # Evaluate policy for state action pair pi(a_t|s_t)
        # Retrieves log pi(a_t|s_t), V(s_t), entropy: H(pi)
        log_prob, value, entropy = self.policy.evaluate(
            batch.state, batch.action)
        value = value.flatten()

        # Advantage estimate
        # A(s_t, a_t) = R_t - V(s_t)
        # Note, we use batch.value here as this one requires no grad.
        # Alternatively we could detach: advantage.detach()
        advantage = batch.return_ - batch.value

        # Optionally could normalize advantage

        # Policy loss
        # L_pi = E_t [- log pi(a_t|s_t) * A(s_t, a_t)]
        policy_loss = -(log_prob * advantage).mean()

        # Value loss
        # L_v = E_t [(R_t - V(s_t))^2]
        # Note this loss is simply advantage squared, value requires grad here.
        value_loss = self.mse_loss(batch.return_, value)

        # Entropy loss
        # L_H = E_t [H(pi)]
        entropy_loss = -entropy.mean()

        # Total loss
        # L = L_pi + L_v + L_H
        loss = policy_loss + self.hparams.value_coef * value_loss \
                           + self.hparams.entropy_coef * entropy_loss

        # TODO: Clip gradients. Can do in trainer args.

        # Log metrics
        self.log('policy_loss', policy_loss)
        self.log('value_loss', value_loss)
        self.log('entropy_loss', entropy_loss)
        self.log('loss', loss, prog_bar=True)
        self.log('std', self.policy.log_std[0].exp().item(), prog_bar=True)
        self.log('return', batch.return_.mean().item(), prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        self.rollout_agent.prepare()
        # Inform policy of new device
        self.policy.update_device()

    def on_train_epoch_end(self) -> None:
        self.test_epoch()

    def test_epoch(self):
        env = gym.make(self.hparams.env_id)#, render_mode='human')

        # %%
        # Run a few episodes
        for episode in range(1):
            done = truncated = False
            total_reward = 0.

            state, _ = env.reset()
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            while not done:
                # time.sleep(0.05)
                # Choose a random action
                action = self.policy.act(state)
                # Take action 0, as policy considers vectorized env.
                action = action[0].cpu().numpy()
                
                # # Environment has boxed action space, so clip actions.
                # if isinstance(env.action_space, spaces.Box):
                #     clipped_action = np.clip(action, env.action_space.low,
                #                             env.action_space.high)

                # Perform the action in the environment
                state, reward, done, truncated, info = env.step(action)
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

                # Update the total reward
                total_reward += float(reward)

                if done or truncated:
                    break

            # Print the total reward for the episode
            self.log('test_reward', total_reward, prog_bar=True)
            # print("Episode:", episode + 1, "Total Reward:", total_reward)

        env.close()

    def train_dataloader(self) -> DataLoader:
        self.dataset = RolloutBufferDataset(
            self.rollout_agent,
            max_steps=self.hparams.steps_per_epoch)  # type: ignore
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def configure_optimizers(self) -> Optimizer:
        return OPTIMIZERS[self.hparams.optimizer](self.policy.parameters(),  # type: ignore
                                                  lr=self.hparams.learning_rate)  # type: ignore
