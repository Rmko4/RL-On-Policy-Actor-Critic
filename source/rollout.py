from collections.abc import Iterator
from typing import NamedTuple, Tuple

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from policy_network import ActorCriticPolicy
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import IterableDataset


class RolloutSample(NamedTuple):
    # Using typing.NamedTuple
    state: np.ndarray | Tensor
    action: np.ndarray | Tensor
    value: float | Tensor
    return_: float | Tensor
    log_prob: float | Tensor
    terminal: bool | Tensor


class RolloutBuffer():
    def __init__(self,
                 buffer_size: int,
                 num_envs: int,
                 state_space: Tuple[int, ...],
                 action_space: Tuple[int, ...],
                 gamma: float = 0.99,
                 gae_lambda: float = 1.) -> None:
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.buffer_shape = (buffer_size, num_envs)
        self.batch_size = buffer_size * num_envs
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    def __getitem__(self, index: int) -> RolloutSample:
        sample = RolloutSample(
            state=self.state_buffer[index],
            action=self.action_buffer[index],
            value=self.value_buffer[index],
            return_=self.return_buffer[index],
            log_prob=self.log_prob_buffer[index],
            terminal=self.done_buffer[index],
        )
        return sample

    def __iter__(self) -> Iterator[RolloutSample]:
        assert self.finalized, "Should only iterate over finalized buffer"
        return self

    def __next__(self) -> RolloutSample:
        if self.iter_pos < self.size * self.num_envs:
            sample = self[self.iter_pos]
            self.iter_pos += 1
            return sample
        else:
            raise StopIteration

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray,
            ) -> None:
        rollout_step = [state, action, reward, done, value, log_prob]
        for buffer, data in zip(self.add_buffer, rollout_step):
            buffer[self.head_pos] = data

        self.head_pos += 1

    def finalize(self, last_value: np.ndarray) -> None:
        assert self.size == self.buffer_size, "Should only finalize when buffer is full"
        self._compute_gae_returns(last_value)

        # Flatten the outer dims
        for buffer_name in self.get_buffer_names:
            buffer: np.ndarray = self.__dict__[buffer_name]
            self.__dict__[buffer_name] = buffer.reshape(self.batch_size, *buffer.shape[2:])

        self.finalized = True

    def reset(self):
        self.state_buffer = np.zeros(
            (self.buffer_size, *self.state_space), dtype=np.float32)
        self.action_buffer = np.zeros(
            (self.buffer_size, *self.action_space), dtype=np.float32)

        self.value_buffer = np.zeros(self.buffer_shape, dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_shape, dtype=np.float32)
        self.return_buffer = np.zeros(self.buffer_shape, dtype=np.float32)

        self.log_prob_buffer = np.zeros(self.buffer_shape, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_shape, dtype=np.float32)

        self.add_buffer = [self.state_buffer, self.action_buffer,
                           self.reward_buffer, self.done_buffer,
                           self.value_buffer, self.log_prob_buffer]

        self.get_buffer_names = ["state_buffer", "action_buffer",
                           "value_buffer", "return_buffer",
                           "log_prob_buffer", "done_buffer"]

        self.head_pos = 0
        self.iter_pos = 0
        self.finalized = False

    def _compute_gae_returns(self, last_value: np.ndarray):
        # Compute the generalized advantage estimate (GAE) of the return.
        last_advantage = 0
        value_next = last_value
        for t in reversed(range(self.buffer_size)):
            value = self.value_buffer[t]
            reward = self.reward_buffer[t]
            done = self.done_buffer[t]

            delta = reward + self.gamma * value_next * (1 - done) - value
            advantage = delta + self.gamma * self.gae_lambda * \
                last_advantage * (1 - done)
            self.return_buffer[t] = advantage + value

            value_next = value
            last_advantage = advantage

    @property
    def size(self) -> int:
        return self.head_pos


class RolloutAgent():
    def __init__(self,
                 env: VectorEnv,
                 policy: ActorCriticPolicy,
                 num_rollout_steps: int = 5,
                 gamma: float = 0.99,
                 gae_lambda: float = 1.) -> None:
        self.env = env
        self.policy = policy
        self.num_rollout_steps = num_rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.rollout_buffer = RolloutBuffer(
            num_rollout_steps,
            env.num_envs,
            env.observation_space.shape,
            env.action_space.shape,
            gamma=gamma,
            gae_lambda=gae_lambda,)

    def prepare(self) -> None:
        self.last_state, _ = self.env.reset()

    def perform_rollout(self) -> None:
        assert self.last_state is not None, "Call prepare() first"

        buffer = self.rollout_buffer
        buffer.reset()

        self.policy.train(False)

        for step in range(self.num_rollout_steps):
            # Gradient is not computed during rollout,
            # this would cause issues with computing the policy gradient loss.
            with torch.no_grad():
                last_state = torch.as_tensor(
                    self.last_state, dtype=torch.float32, device=self.policy.device)
                action_dist, value = self.policy(last_state)
                action_dist: Distribution
                value: Tensor

            # Sample an action vector given the policy distribution conditioned on state.
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1).cpu().numpy()
            action = action.cpu().numpy()
            value = value.cpu().numpy().flatten()

            clipped_action = action
            # Environment has boxed action space, so clip actions.
            if isinstance(self.env.action_space, spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low,
                                         self.env.action_space.high)

            next_state, reward, terminated, truncated, info = self.env.step(
                clipped_action)

            done = terminated | truncated

            for i, trunc in enumerate(truncated):
                if trunc:
                    final_state = info['final_observation'][i]
                    final_state = torch.as_tensor(
                        final_state, dtype=torch.float32, device=self.policy.device)
                    final_value = self.policy.predict_value(final_state)[0]
                    # When truncated set reward to be R += gamma * V(s_t+1)
                    # After all, the return is not zero as with terminal states.
                    reward[i] += self.gamma * final_value

            # Note that unclipped action is stored.
            buffer.add(self.last_state, action, reward, done, value, log_prob)

            # Lets store only numpy objects in rollout buffer.
            # When retrieving they are converted to tensors.

            self.last_state = next_state

        # The value of the last state is required for the TD error.
        with torch.no_grad():
            last_state = torch.as_tensor(
                self.last_state, dtype=torch.float32, device=self.policy.device)
            last_value = self.policy.predict_value(last_state)
            last_value: Tensor
            last_value = last_value.cpu().numpy().flatten()

        self.policy.train(True)

        buffer.finalize(last_value)


class RolloutBufferDataset(IterableDataset):
    """ Iterable dataset for replay buffer
        Supports random sampling of dynamic replay buffer
    """

    def __init__(self,
                 rollout_agent: RolloutAgent,
                 max_steps: int = 20) -> None:
        self.rollout_agent = rollout_agent
        self.max_steps = max_steps
        self.rollout_buffer = rollout_agent.rollout_buffer

    def __iter__(self) -> Iterator:
        for epoch_step in range(self.max_steps):
            self.rollout_agent.perform_rollout()

            # Does not return iterator,
            # but instead wraps it in generator that also performs rollout.
            for rollout_step in self.rollout_buffer:
                yield rollout_step
