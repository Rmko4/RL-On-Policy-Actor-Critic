from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal


class ActorCriticPolicy(nn.Module):
    def __init__(self, state_size: int,
                 n_actions: int,
                 hidden_size: int = 128) -> None:
        super().__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size

        # Default shared feature extractor.
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, n_actions)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

        # Initialize the log standard deviation to zeros.
        # Use log std to make sure std is always positive (such that it is differentiable)
        self.log_std = nn.Parameter(torch.zeros(
            self.n_actions), requires_grad=True)

        self.init_weights()

    def init_weights(self) -> None:
        # Initialize the weights of the network.
        # Action head weights are initialized to 0.01 to ensure the initial policy
        # will be close to the zero policy (zero actions).

        module_gains = {
            self.feature_extractor: 1.0,
            self.actor_head: 0.01,
            self.critic_head: 1.0
        }

        def _init_weights(m: nn.Module, gain):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain)
                nn.init.zeros_(m.bias)

        for module, gain in module_gains.items():
            module.apply(lambda x: _init_weights(x, gain))

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        features = self.feature_extractor(x)

        values = self.critic_head(features)
        mu = self.actor_head(features)
        std = self.log_std.exp().expand_as(mu)

        action_distribution = Normal(mu, std)
        # actions = distribution.sample()
        # log_prob = distribution.log_prob(actions)

        return action_distribution, values
