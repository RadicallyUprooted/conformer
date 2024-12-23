import torch
from torch import nn, Tensor

from modules.feed_forward import FeedForward
from modules.attention import Attention
from modules.convolution import Convolution

class ConformerBlock(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        attention_heads: int,
        depthwise_conv_kernel_size: int,
        ff_dropout_p: float,
        attn_dropout_p: float,
        conv_dropout_p: float,
    ):
        super().__init__()

        self.ff1 = FeedForward(encoder_dim, ff_dropout_p)
        self.attn = Attention(encoder_dim, attention_heads, attn_dropout_p)
        self.conv = Convolution(encoder_dim, depthwise_conv_kernel_size, conv_dropout_p)
        self.ff2 = FeedForward(encoder_dim, ff_dropout_p)
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: Tensor) -> Tensor:

        x += 0.5 * self.ff1(x)
        x += self.attn(x)
        x += self.conv(x)
        x += 0.5 * self.ff2(x)

        out = self.layer_norm(x)

        return out
    

if __name__ == '__main__':
    x = torch.randn(1, 512, 144)

    conformer = ConformerBlock(144, 512, 4)

    y = conformer(x)

    print(y.size())
