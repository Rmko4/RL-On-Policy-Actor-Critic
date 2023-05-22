from typing import Tuple

from torch import Tensor, nn

DEFAULT_STATE_SHAPE = (16,)

class DummyPolicyNetwork(nn.Module):
    def __init__(self,
                 state_shape: Tuple[int, ...] = DEFAULT_STATE_SHAPE,
                 hidden_size: int = 128,
                 ) -> None:
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(state_shape[-1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.ff(x)
        return x, x
        