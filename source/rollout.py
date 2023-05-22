import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import List, NamedTuple, Tuple

import numpy as np
from torch.distributions import Distribution
import torch
from gymnasium.vector import VectorEnv
from policy_network import ActorCriticPolicy
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

    def add(self, *items) -> None:
        pass

    def reset(self):
        pass


class RolloutAgent():
    def __init__(self,
                 env: VectorEnv,
                 policy: ActorCriticPolicy,
                 num_rollout_steps: int = 5) -> None:
        self.env = env
        self.policy = policy
        self.num_rollout_steps = num_rollout_steps

        self.rollout_buffer = RolloutBuffer(num_rollout_steps, env.num_envs)

    def prepare(self) -> None:
        self.last_state = self.env.reset()

    def perform_rollout(self) -> None:
        buffer = self.rollout_buffer
        buffer.reset()

        self.policy.train(False)

        for step in range(self.num_rollout_steps):
            # Gradient is not computed for rollout,
            # this will all be done in training_step in one go.
            with torch.no_grad():
                action_dist, value, log_prob = self.policy(self.last_state)
                action_dist: Distribution
                log_prob: Tensor

                action = action_dist.sample()

                next_state, reward, terminated, truncated, info = \
                    self.env.step(action)

                done = terminated or truncated

                buffer.add(self.last_state, action, reward, value, log_prob,
                            done)
                
                self.last_state = next_state

        self.policy.train(True)


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
