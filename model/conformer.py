import torch
from torch import nn, Tensor

from conformer_encoder import Encoder

class Conformer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        num_layers: int = 16,
        input_dim: int = 80,
        encoder_dim: int = 144,
        attention_heads: int = 4,
        depthwise_conv_kernel_size: int = 31,
        input_dropout_p: float = 0.1,
        ff_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim,
            encoder_dim,
            num_layers,
            attention_heads,
            depthwise_conv_kernel_size,
            input_dropout_p,
            ff_dropout_p,
            attn_dropout_p,
            conv_dropout_p,
        )
        self.linear = nn.Linear(encoder_dim, num_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:

        enc_out, out_lengths = self.encoder(x, x_lengths)  
        
        outputs = self.linear(enc_out)
        outputs = self.softmax(outputs)

        return outputs, out_lengths

if __name__ == '__main__':
    
    batch_size, sequence_length, dim = 3, 12345, 80

    cuda = torch.cuda.is_available()  
    device = torch.device('cuda' if cuda else 'cpu')

    criterion = nn.CTCLoss().to(device)

    inputs = torch.rand(batch_size, sequence_length, dim).to(device)
    input_lengths = torch.LongTensor([12345, 12300, 12000])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7])

    model = Conformer(num_classes=10, 
                    input_dim=dim, 
                    encoder_dim=32, 
                    num_layers=3).to(device)

    # Forward propagate
    outputs, output_lengths = model(inputs, input_lengths)
    # Calculate CTC Loss
    loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths) 

    #print(outputs.size())
    print(output_lengths)
    #print(targets.size())
    #print(target_lengths.size())
    #print(f"Loss: {loss}")


