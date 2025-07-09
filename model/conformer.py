from torch import nn, Tensor

from conformer_encoder import Encoder

class Conformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_mels: int,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            n_mels,
            d_model,
            n_layers,
            n_heads,
            conv_kernel_size,
            dropout,
        )
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)  
        
        outputs = self.linear(encoder_outputs)
        outputs = self.softmax(outputs)

        return outputs, output_lengths


