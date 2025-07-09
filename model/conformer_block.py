from torch import nn, Tensor

from modules.feed_forward import FeedForward
from modules.attention import Attention
from modules.convolution import Convolution

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

        inputs += 0.5 * self.ff1(inputs)
        inputs += self.attn(inputs)
        inputs += self.conv(inputs)
        inputs += 0.5 * self.ff2(inputs)

        outputs = self.layer_norm(inputs)

        return outputs
