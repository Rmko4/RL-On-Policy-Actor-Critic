from typing import Any, Callable

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader

import gymnasium as gym


OPTIMIZERS = {'Adam': Adam,
              'RMSprop': RMSprop,
              'SGD': SGD}


class PolicyGradientModule(LightningModule):
    def __init__(self,
                 env_id: str = 'Ant-v4',
                 algorithm: str = 'A2C',
                 steps_per_epoch: int = 10,
                 num_envs: int = 8,
                 num_rollout_steps: int = 5,
                 optimizer: str = 'RMSProp',
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Does frame inspection to find parameters
        self.save_hyperparameters()

        self.env = gym.vector.make(env_id, num_envs=num_envs, asynchronous=True)
        pass

    # TODO: Typing

    def training_step(self, batch, batch_idx: int) -> Tensor:
        pass

    def on_train_epoch_end(self) -> None:
        self.test_epoch()

    def test_epoch(self):
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def configure_optimizers(self) -> Optimizer:
        pass
