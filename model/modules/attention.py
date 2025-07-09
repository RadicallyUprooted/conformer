import math

import torch
from torch import Tensor
from torch import nn
from einops import rearrange, einsum

class RelativePositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_relative_position: int = 30
    ):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_relative_position = max_relative_position

    def forward(self, q: Tensor) -> Tensor:
        
        q_length = q.size(2)
        
        range_vec = torch.arange(q_length)
        range_mat = rearrange(range_vec, 'j -> 1 j')

        distance_mat = range_mat - rearrange(range_vec, 'i -> i 1')
        
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        
        final_mat = distance_mat_clipped + self.max_relative_position

        encodings = torch.zeros(2 * self.max_relative_position + 1, self.d_model)       # (x_length, d_model)
        positions = torch.arange(0, (2 * self.max_relative_position + 1)).unsqueeze(1)  # (x_length, 1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        )
        
        encodings[:, 0::2] = torch.sin(positions * div_term)
        encodings[:, 1::2] = torch.cos(positions * div_term)
        encodings = encodings[final_mat]
        encodings = rearrange(encodings, 't1 t2 f -> 1 t1 t2 f').requires_grad_(False)
        return encodings.to(q.device)

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super(MultiHeadSelfAttention, self).__init__()

        self.heads_dim = d_model // n_heads
        self.attention_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.out_proj = nn.Linear(d_model, d_model)
        
        self.u_bias = nn.Parameter(torch.zeros(1, self.attention_heads, 1, self.heads_dim))
        self.v_bias = nn.Parameter(torch.zeros(1, self.attention_heads, 1, self.heads_dim))

    def forward(
        self, 
        inputs: Tensor,
        relative_positional_embeddings: Tensor
    ) -> Tensor:

        q = rearrange(self.q_proj(inputs), 'b t (h d) -> b h t d', h=self.attention_heads)
        k = rearrange(self.k_proj(inputs), 'b t (h d) -> b h t d', h=self.attention_heads)
        v = rearrange(self.v_proj(inputs), 'b t (h d) -> b h t d', h=self.attention_heads)
        
        pos = self.pos_proj(relative_positional_embeddings)
        
        q_with_bias_u = q + self.u_bias
        q_with_bias_v = q + self.v_bias

        qk_score = einsum(q_with_bias_u, k, 'b h t1 d, b h t2 d -> b h t1 t2')
        
        pos_score = einsum(q_with_bias_v, pos, 'b h t1 d, b t1 t2 d -> b h t1 t2')

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
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super(Attention, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_embeddings = RelativePositionalEncoding(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        
        inputs = self.layer_norm(inputs)
        q = rearrange(self.attn.q_proj(inputs), 'b t (h d) -> b h t d', h=self.attn.attention_heads)
        relative_positional_embedding = self.pos_embeddings(q)
        outputs = self.attn(inputs, relative_positional_embedding)
        outputs = self.dropout(outputs)

        return outputs