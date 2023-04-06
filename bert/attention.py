import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def dot_product_attention(Q, K, V) -> torch.Tensor:
        '''
        Dot-product attention 

        K: a tensor with shape (batch, (X), length, qk_channels) 
        Q: a tensor with shape (batch, (X), length, qk_channels) 
        V: a tensor with shape (batch, (X), length, output_channels)

        Returns a tensor with shape (batch, (X), length, output_channels)
        '''
        d = Q.shape[-1]

        x = torch.matmul(Q, torch.transpose(K, -2, -1)) / np.sqrt(d)
        A = F.softmax(x, -1)
                         
        out = torch.matmul(A, V)

        return out

class Attention(nn.Module):
    def __init__(self, model_dims, key_dims=None, value_dims=None, dropout_rate=0.1) -> None:
        super().__init__()

        if not key_dims:
            key_dims = model_dims
        if not value_dims:
            value_dims = model_dims

        self.Q_projection = nn.Linear(model_dims, key_dims)
        self.K_projection = nn.Linear(model_dims, key_dims)
        self.V_projection = nn.Linear(model_dims, value_dims)
        self.fc = nn.Linear(value_dims, model_dims)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(model_dims)

    def forward(self, x) -> torch.Tensor:
        '''
        Attention

        x: a tensor with shape (batch, length, model_dims) 

        Returns a tensor with shape (batch, length, model_dims)
        '''
        residual = x

        Q = self.Q_projection(x)
        K = self.K_projection(x)
        V = self.V_projection(x)

        out = dot_product_attention(Q, K, V)

        out = self.fc(out)
        out = self.dropout(out)
        
        out += residual
        out = self.norm(out)
        
        return out

# TODO: change how out_channels works
class MultiheadAttention(nn.Module):
    def __init__(self, model_dims, key_dims=None, value_dims=None, heads=1, dropout_rate=0.1) -> None:
        super().__init__()

        if not key_dims:
            key_dims = model_dims
        if not value_dims:
            value_dims = model_dims

        self.model_dims = model_dims
        self.key_dims = key_dims
        self.value_dims = value_dims
        self.heads = heads

        self.total_key_dims = heads * key_dims
        self.total_value_dims = heads * value_dims
        
        self.Q_projection = nn.Linear(model_dims, self.total_key_dims)
        self.K_projection = nn.Linear(model_dims, self.total_key_dims)
        self.V_projection = nn.Linear(model_dims, self.total_value_dims)
        self.fc = nn.Linear(self.total_value_dims, model_dims)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(model_dims)

    def forward(self, x) -> torch.Tensor:
        '''
        Multi-head attention attention 

        x: a tensor with shape (batch, length, model_dims)

        Returns a tensor with shape (batch, length, model_dims)
        '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        residual = x

        Q = self.Q_projection(x).reshape(batch_size, seq_len, self.heads, self.key_dims).permute(0, 2, 1, 3)
        K = self.K_projection(x).reshape(batch_size, seq_len, self.heads, self.key_dims).permute(0, 2, 1, 3)
        V = self.V_projection(x).reshape(batch_size, seq_len, self.heads, self.value_dims).permute(0, 2, 1, 3)

        out = dot_product_attention(Q, K, V)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.total_value_dims)

        out = self.fc(out)  # batch, len, model_dim
        out = self.dropout(out)

        out += residual
        out = self.norm(out)

        return out
