import torch
from torch import nn, Tensor

from conformer_block import ConformerBlock
from modules.subsampling import Subsampling

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 144,
        num_layers: int = 16,
        attention_heads: int = 4,
        depthwise_conv_kernel_size: int = 31,
        input_dropout_p: float = 0.1,
        ff_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
    ):
        super().__init__()

        self.subsample = Subsampling(out_channels=encoder_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_features=encoder_dim * (((input_dim - 1) // 2 - 1) // 2), out_features=encoder_dim),
            nn.Dropout(p=input_dropout_p)
        )
        self.blocks = nn.ModuleList(
            [
            ConformerBlock(
                encoder_dim=encoder_dim,
                attention_heads=attention_heads,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                ff_dropout_p=ff_dropout_p,
                attn_dropout_p=attn_dropout_p,
                conv_dropout_p=conv_dropout_p,
            ) for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:

        out, out_len = self.subsample(x, x_lengths)
        out = self.proj(out)

        for block in self.blocks:
            out = block(out)

        return out, out_len    


if __name__ == '__main__':
    x = torch.randn(1, 512, 512)

    encoder = Encoder(input_dim=x.size(2))

    y, y_len = encoder(x)

    print(y.size())