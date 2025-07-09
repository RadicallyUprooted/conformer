from torch import Tensor
from torch import nn

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
    ):
        super(FeedForward, self).__init__()

        self.module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                in_features=d_model,
                out_features=d_model * 4,
            ),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=d_model * 4,
                out_features=d_model,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:

        return self.module(inputs)