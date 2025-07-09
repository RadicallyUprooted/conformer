import torch
from torch import nn, Tensor

from conformer_block import ConformerBlock
from modules.subsampling import Subsampling

class Encoder(nn.Module):

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.subsample = Subsampling(in_channels=1, d_model=d_model)
        
        subsample_output_dim = self._get_subsample_output_dim(n_mels)

        self.proj = nn.Sequential(
            nn.Linear(in_features=subsample_output_dim, out_features=d_model),
            nn.Dropout(p=dropout)
        )
        self.blocks = nn.ModuleList(
            [
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            ) for _ in range(n_layers)
            ]
        )

    def _get_subsample_output_dim(self, n_mels):
        # Create a dummy input tensor to pass through the subsampling layer
        dummy_input = torch.randn(1, 100, n_mels)  # (B, L, D)
        dummy_lengths = torch.tensor([100])
        
        # Pass the dummy input through the subsampling layer
        with torch.no_grad():
            output, _ = self.subsample(dummy_input, dummy_lengths)
        
        return output.shape[-1]

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:

        outputs, output_lengths = self.subsample(inputs, input_lengths)
        outputs = self.proj(outputs)

        for block in self.blocks:
            outputs = block(outputs)

        return outputs, output_lengths