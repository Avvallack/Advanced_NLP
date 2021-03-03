import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, num_heads=1, dropout=0.0):
        super().__init__()
        assert model_dimension % num_heads == 0, "Input dimension isn't divisible by number of heads"
        self.num_heads = num_heads

        self.query = nn.Linear(model_dimension, model_dimension)
        self.keys = nn.Linear(model_dimension, model_dimension)
        self.values = nn.Linear(model_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)
        self.outputs = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, mask=None, dropout=None):
        dim_k = q.size()[-1]
        score = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(dim_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        if dropout is not None:
            score = dropout(score)
        score = torch.matmul(F.softmax(score, dim=-1), v)

        return score

    def _split_to_heads(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_heads(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, k, q, v, mask=None):
        q, k, v = self.query(q), self.keys(k), self.values(v)
        q = self._split_to_heads(q)
        k = self._split_to_heads(k)
        v = self._split_to_heads(v)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        y = self.attention(k, q, v, mask, self.dropout)
        y = self._reshape_from_heads(y)
        y = self.outputs(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, model_dim, linear_dim=2048, dropout=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, linear_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(linear_dim, model_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(self.dropout(x))
        x = self.linear_2(x)
        return x


class Normalization(nn.Module):
    def __init__(self, model_dim, eps=1e-8):
        super().__init__()

        self.size = model_dim
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
