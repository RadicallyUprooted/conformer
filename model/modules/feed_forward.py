from torch import Tensor
from torch import nn

class FeedForward(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        dropout_p: float = 0.1,
    ):
        super(FeedForward, self).__init__()

        self.module = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(
                in_features=encoder_dim,
                out_features=encoder_dim * 4,
            ),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(
                in_features=encoder_dim * 4,
                out_features=encoder_dim,
            ),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.module(x)