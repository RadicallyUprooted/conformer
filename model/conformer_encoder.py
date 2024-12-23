from torch import nn, Tensor

from conformer_block import ConformerBlock
from modules.subsampling import Subsampling

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        num_layers: int,
        attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout_p: float,
    ):
        super().__init__()

        self.subsample = Subsampling(out_channels=encoder_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_features=encoder_dim * (((input_dim - 1) // 2 - 1) // 2), out_features=encoder_dim),
            nn.Dropout(p=dropout_p)
        )
        self.blocks = nn.ModuleList(
            [
            ConformerBlock(
                encoder_dim=encoder_dim,
                attention_heads=attention_heads,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:

        out, out_len = self.subsample(x, x_lengths)
        out = self.proj(out)

        for block in self.blocks:
            out = block(out)

        return out, out_len