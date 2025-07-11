import torch
from torch import Tensor
from torch import nn
from einops import rearrange

class Subsampling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super(Subsampling, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model * 2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> tuple[Tensor, Tensor]:
        
        inputs = rearrange(inputs, 'b l d -> b 1 l d')

        outputs = self.module(inputs)
        outputs = rearrange(outputs, 'b c l d -> b l (c d)')

        output_lengths = torch.floor((input_lengths - self.module[0].kernel_size[0]) / self.module[0].stride[0] + 1)
        output_lengths = torch.floor((output_lengths - self.module[2].kernel_size[0]) / self.module[2].stride[0] + 1)
        output_lengths = output_lengths.to(torch.int64)

        return outputs, output_lengths
