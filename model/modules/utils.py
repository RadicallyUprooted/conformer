from torch import Tensor
from torch.nn import Module
from einops import rearrange

class Rearrange(Module):

    def __init__(
        self,
        pattern: str,
    ):
        super(Rearrange, self).__init__()

        self.pattern = pattern

    def forward(self, x: Tensor) -> Tensor:

        return rearrange(x, pattern=self.pattern)
    

