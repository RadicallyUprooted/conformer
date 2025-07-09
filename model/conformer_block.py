from torch import nn, Tensor

from .modules.feed_forward import FeedForward
from .modules.attention import Attention
from .modules.convolution import Convolution

class ConformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.ff1 = FeedForward(d_model, dropout)
        self.attn = Attention(d_model, n_heads, dropout)
        self.conv = Convolution(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForward(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:

        x = inputs + 0.5 * self.ff1(inputs)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)

        outputs = self.layer_norm(x)

        return outputs
