from torch import Tensor
from torch import nn

from utils import Rearrange

class Pointwise_Conv(nn.Module):
    # 1D convolution with kernel_size=1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(Pointwise_Conv, self).__init__()

        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
        )
    
    def forward(self, x: Tensor) -> Tensor:

        return self.pointwise_conv(x)

class Depthwise_Conv(nn.Module):
    # 1D convolution with groups=in_channels
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(Depthwise_Conv, self).__init__()

        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.depthwise_conv(x)    

class Convolution(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        kernel_size: int = 31,
        dropout_p: float = 0.1,
    ):
        super(Convolution, self).__init__()

        self.module = nn.Sequential(
            nn.LayerNorm(in_channels),
            Rearrange(pattern='b l d -> b d l'),
            Pointwise_Conv(
                in_channels=in_channels,
                out_channels=2*in_channels,
            ),
            nn.GLU(dim=1),
            Depthwise_Conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            Pointwise_Conv(
                in_channels=in_channels,
                out_channels=in_channels,
            ),
            nn.Dropout(p=dropout_p),
            Rearrange(pattern='b d l -> b l d'),
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.module(x)    