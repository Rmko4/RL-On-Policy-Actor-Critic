import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import List, NamedTuple, Tuple

import numpy as np
from gymnasium.vector import VectorEnv
from torch import Tensor, nn
from torch.utils.data import IterableDataset


class RolloutSample(NamedTuple):
    # Using typing.NamedTuple
    states: np.ndarray | Tensor
    actions: np.ndarray | Tensor
    rewards: float | Tensor
    returns: float | Tensor
    log_probs: float | Tensor
    terminal: bool | Tensor


class RolloutBuffer():
    def __init__(self, buffer_size: int, num_envs: int) -> None:
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.reset()

    def __getitem__(self, index: int) -> RolloutSample:
        pass

    def collect(self) -> None:
        pass

    def reset(self):
        pass

    @property
    def size(self) -> int:
        return 1


class RolloutAgent():
    def __init__(self,
                 env: VectorEnv,
                 policy_network: nn.Module,
                 num_rollout_steps: int = 5) -> None:
        self.env = env
        self.policy_network = policy_network
        self.num_rollout_steps = num_rollout_steps

        self.rollout_buffer = RolloutBuffer(num_rollout_steps, env.num_envs)
        

    def perform_rollout(self) -> None:
        buffer = self.rollout_buffer
        buffer.reset()

        for step in range(self.num_rollout_steps):
            pass

        pass


class RolloutBufferDataset(IterableDataset):
    """ Iterable dataset for replay buffer
        Supports random sampling of dynamic replay buffer
    """

    def __init__(self,
                 rollout_agent: RolloutAgent,
                 max_steps: int = 10) -> None:
        self.rollout_agent = rollout_agent
        self.max_steps = max_steps
        self.rollout_buffer = rollout_agent.rollout_buffer

    def __iter__(self) -> Iterator:
        for epoch_step in range(self.max_steps):
            self.rollout_agent.perform_rollout()


            for i in range(self.rollout_buffer.size):
                yield self.rollout_buffer[i]


# def main():
#     from torch.utils.data import DataLoader
#     buffer = UniformReplayBuffer(10)
#     for i in range(10):
#         buffer.append(Trajectory(i, i, i, i, False))
#     dataset = RolloutBufferDataset(buffer)
#     dataloader = DataLoader(dataset, batch_size=2)
#     for batch in dataloader:
#         print(batch)
#         break


# if __name__ == "__main__":
#     main()
