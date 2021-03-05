import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, num_heads=1, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.q = nn.Linear(model_dimension, model_dimension * num_heads)
        self.k = nn.Linear(model_dimension, model_dimension * num_heads)
        self.v = nn.Linear(model_dimension, model_dimension * num_heads)
        self.outputs = nn.Linear(model_dimension * num_heads, model_dimension)

    @staticmethod
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout=None, num_heads=1):
        seq_len, bsz, embedding_dim, = q.size()
        head_dim = embedding_dim // num_heads

        q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        score = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(head_dim)

        if dropout is not None:
            score = dropout(score)
        score = torch.bmm(F.softmax(score, dim=-1), v)
        score = score.transpose(0, 1).contiguous().view(seq_len, bsz, embedding_dim)
        return score

    def forward(self, x):
        q = self.q(x).contiguous().transpose(0, 1)
        k = self.k(x).contiguous().transpose(0, 1)
        v = self.v(x).contiguous().transpose(0, 1)
        y = self.attention(q, k, v, self.dropout, self.num_heads)
        y = y.contiguous().transpose(0, 1)
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
