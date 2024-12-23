from torch import nn, Tensor

from conformer_encoder import Encoder

class Conformer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        input_dim: int,
        encoder_dim: int,
        attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout_p: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim,
            encoder_dim,
            num_layers,
            attention_heads,
            depthwise_conv_kernel_size,
            dropout_p,
        )
        self.linear = nn.Linear(encoder_dim, num_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:

        enc_out, out_lengths = self.encoder(x, x_lengths)  
        
        outputs = self.linear(enc_out)
        outputs = self.softmax(outputs)

        return outputs, out_lengths


