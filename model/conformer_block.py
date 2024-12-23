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
        dropout_p: float,
    ):
        super().__init__()

        self.ff1 = FeedForward(encoder_dim, dropout_p)
        self.attn = Attention(encoder_dim, attention_heads, dropout_p)
        self.conv = Convolution(encoder_dim, depthwise_conv_kernel_size, dropout_p)
        self.ff2 = FeedForward(encoder_dim, dropout_p)
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: Tensor) -> Tensor:

        x += 0.5 * self.ff1(x)
        x += self.attn(x)
        x += self.conv(x)
        x += 0.5 * self.ff2(x)

        out = self.layer_norm(x)

        return out
