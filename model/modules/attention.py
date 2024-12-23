import math

import torch
from torch import Tensor
from torch import nn
from einops import rearrange, einsum

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
    ):
        super(PositionalEncoding, self).__init__()

        self.encoder_dim = encoder_dim

    def forward(self, x: Tensor) -> Tensor:
        
        x_length = x.size(1)

        encodings = torch.zeros(x_length, self.encoder_dim)    # (x_length, encoder_dim)
        positions = torch.arange(0, x_length).unsqueeze(1)      # (x_length, 1)

        div_term = torch.exp(
            torch.arange(0, self.encoder_dim, 2) * -(math.log(10000.0) / self.encoder_dim)
        )
        
        encodings[:, 0::2] = torch.sin(positions * div_term)
        encodings[:, 1::2] = torch.cos(positions * div_term)
        encodings = rearrange(encodings, 't f -> 1 t f').requires_grad_(False)
        return encodings.to(x.device)

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        attention_heads: int,
        dropout_p: float,
    ):
        super(MultiHeadSelfAttention, self).__init__()

        self.heads_dim = encoder_dim // attention_heads
        self.attention_heads = attention_heads

        self.q_proj = nn.Linear(encoder_dim, encoder_dim)
        self.k_proj = nn.Linear(encoder_dim, encoder_dim)
        self.v_proj = nn.Linear(encoder_dim, encoder_dim)
        self.pos_proj = nn.Linear(encoder_dim, encoder_dim)

        self.dropout = nn.Dropout(p=dropout_p)

        self.out_proj = nn.Linear(encoder_dim, encoder_dim)

    def forward(
        self, x: Tensor,
        pos_embeddings: Tensor
    ) -> Tensor:

        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.attention_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.attention_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.attention_heads)
        
        pos = rearrange(self.pos_proj(pos_embeddings), 'b t (h d) -> b h t d', h=self.attention_heads)

        qk_score = einsum(q, k, 'b h t1 d, b h t2 d -> b h t1 t2')
        pos_score = einsum(q, pos, 'b h t1 d, b h t2 d -> b h t1 t2')

        scores = (qk_score + pos_score) / math.sqrt(self.heads_dim)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = einsum(attn, v, 'b h t1 t2, b h t2 d -> b h t1 d')
        context = rearrange(context, 'b h t d -> b t (h d)')

        out = self.out_proj(context)

        return out

class Attention(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        attention_heads: int,
        dropout_p: float,
    ):
        super(Attention, self).__init__()

        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.pos_embeddings = PositionalEncoding(encoder_dim)
        self.attn = MultiHeadSelfAttention(encoder_dim, attention_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:

        pos_emb = self.pos_embeddings(x)
        
        x = self.layer_norm(x)
        x = self.attn(x, pos_emb)
        x = self.dropout(x)

        return x