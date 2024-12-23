from torch import Tensor
from torch import nn
from einops import rearrange

class Subsampling(nn.Module):

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super(Subsampling, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, x_lengths: Tensor) -> tuple[Tensor, Tensor]:
        
        x = rearrange(x, 'b l d -> b 1 l d')

        out = self.module(x)
        out = rearrange(out, 'b c l d -> b l (c d)')

        out_lengths = x_lengths // 4 - 1

        return out, out_lengths