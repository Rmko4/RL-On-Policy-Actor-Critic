# %%
import torch
from torch.distributions import Normal

from torch import Tensor

# %%
mu = torch.tensor([0., 0.1, 1.0])
sigma = torch.tensor([0.1, 0.1, 0.1])


# %%
mu = torch.tensor(0.)
sigma = torch.tensor(0.1)


# %%
dist = Normal(mu, sigma)

# %%

a = dist.sample()
log_prob = dist.log_prob(mu - 0)

print(a)
print(log_prob)

# %%
