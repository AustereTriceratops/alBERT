import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def dot_product_attention(Q, K, V) -> torch.Tensor:
        '''
        Dot-product attention 

        K: a tensor with shape (batch, X, length, qk_channels) 
        Q: a tensor with shape (batch, X, length, qk_channels) 
        V: a tensor with shape (batch, X, length, output_channels)
        '''
        d = Q.shape[-1]

        x = torch.matmul(Q, torch.transpose(K, -2, -1)) / np.sqrt(d)
        A = F.softmax(x, -1)
                         
        out = torch.matmul(A, V)

        return out

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, dropout_rate=0.1) -> None:
        super().__init__()

        if not out_channels:
            out_channels = in_channels
        if not hidden_channels:
            hidden_channels = in_channels

        self.Q_projection = nn.Linear(in_channels=in_channels, out_channels=hidden_channels)
        self.K_projection = nn.Linear(in_channels=in_channels, out_channels=hidden_channels)
        self.V_projection = nn.Linear(in_channels=in_channels, out_channels=out_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x) -> torch.Tensor:
        '''
        Attention

        x: a tensor with shape (batch, length, input_channels) 

        output: a tensor with shape (batch, length, output_channels)
        '''
        Q = self.Q_projection(x)
        K = self.K_projection(x)
        V = self.V_projection(x)

        out = dot_product_attention(Q, K, V)
        out = self.dropout(out)
        
        return out


# TODO: write tests
# TODO: return values of docstrings
class MultiheadAttention(Attention):
    def __init__(self, in_channels, out_channels=None, num_heads=1, hidden_channels=None, dropout_rate=0.1) -> None:
        super().__init__()

        if not out_channels:
            out_channels = in_channels
        if not hidden_channels:
            hidden_channels = in_channels

        self.total_hidden_channels = num_heads * hidden_channels
        self.total_out_channels = num_heads * out_channels
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        
        self.Q_projection = nn.Linear(in_channels=in_channels, out_channels=self.total_hidden_channels)
        self.K_projection = nn.Linear(in_channels=in_channels, out_channels=self.total_hidden_channels)
        self.V_projection = nn.Linear(in_channels=in_channels, out_channels=self.total_out_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x) -> torch.Tensor:
        '''
        Multi-head attention attention 

        x: a tensor with shape (batch, length, channels) 
        '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        Q = self.Q_projection(x).reshape(batch_size, seq_len, self.num_heads, self.hidden_channels).permute(0, 2, 1, 3)
        K = self.K_projection(x).reshape(batch_size, seq_len, self.num_heads, self.hidden_channels).permute(0, 2, 1, 3)
        V = self.V_projection(x).reshape(batch_size, seq_len, self.num_heads, self.output_channels).permute(0, 2, 1, 3)

        out = dot_product_attention(Q, K, V)
        out = self.dropout(out)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.total_out_channels)

        return out
