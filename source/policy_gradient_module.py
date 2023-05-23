from typing import Any, Callable, Dict, Optional, Type

import gymnasium as gym
import torch
from policy_network import ActorCriticPolicy
from pytorch_lightning import LightningModule
from rollout import RolloutAgent, RolloutBuffer, RolloutBufferDataset
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
                 gamma: float = 0.99,
                 gae_lambda: float = 1.,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters()

        # Makes envs in init already
        self.env = gym.vector.make(
            env_id, num_envs=num_envs, asynchronous=True)

        self.state_size = self.env.observation_space.shape[-1]  # type: ignore
        self.n_actions = self.env.action_space.shape[-1]  # type: ignore

        self.policy = ActorCriticPolicy(state_size=self.state_size,
                                        n_actions=self.n_actions,
                                        hidden_size=128)

        self.rollout_agent = RolloutAgent(self.env,
                                          self.policy,
                                          num_rollout_steps=num_rollout_steps,
                                          gamma=gamma,
                                          gae_lambda=gae_lambda)

        self.batch_size = num_envs * num_rollout_steps

    # TODO: Typing

    def training_step(self, batch, batch_idx: int) -> Tensor:
        # Retrieve everything from rollout buffer
        pass

    def on_train_start(self) -> None:
        self.rollout_agent.prepare()
        # Inform policy of new device
        self.policy.update_device()

    def on_train_epoch_end(self) -> None:
        self.test_epoch()

    def test_epoch(self):
        pass

    def train_dataloader(self) -> DataLoader:
        self.dataset = RolloutBufferDataset(
            self.rollout_agent,
            max_steps=self.hparams.steps_per_epoch)  # type: ignore
        return DataLoader(self.dataset, batch_size=self.hparams.num_rollout_steps)

    def configure_optimizers(self) -> Optimizer:
        return OPTIMIZERS[self.hparams.optimizer](self.policy.parameters(),  # type: ignore
                                                  lr=self.hparams.learning_rate)  # type: ignore
