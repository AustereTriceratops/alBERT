import torch
import torch.nn as nn

import numpy as np

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, dropout_rate=0.1) -> None:
        super().__init__()

        if not out_channels:
            out_channels = in_channels
        if not hidden_channels:
            hidden_channels = in_channels

        self.Q_projection = nn.Conv1d(kernel_size=1, in_channels=in_channels, out_channels=hidden_channels)
        self.K_projection = nn.Conv1d(kernel_size=1, in_channels=in_channels, out_channels=hidden_channels)
        self.V_projection = nn.Conv1d(kernel_size=1, in_channels=in_channels, out_channels=out_channels)

        # softmax will be applied over the channels
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x) -> torch.Tensor:
        '''
        Attention

        x: a tensor with shape (batch, length, input_channels) 

        output: a tensor with shape (batch, length, output_channels)
        '''
        x = x.transpose(1, 2)

        # TODO: these should just be linear layers
        Q = self.Q_projection( x ).transpose(1, 2)
        K = self.K_projection( x ).transpose(1, 2)
        V = self.V_projection( x ).transpose(1, 2)

        out = self.dot_product_attention(Q, K, V)

        return out

    def dot_product_attention(self, Q, K, V) -> torch.Tensor:
        '''
        Dot-product attention 

        K: a tensor with shape (batch, length, qk_channels) 
        Q: a tensor with shape (batch, length, qk_channels) 
        V: a tensor with shape (batch, length, output_channels)
        '''
        d = Q.shape[2]

        x = torch.matmul(Q, torch.transpose(K, 1, 2)) / np.sqrt(d)
        A = self.dropout(self.softmax(x))
                         
        out = torch.matmul(A, V)

        return out


# TODO: write tests
# TODO: return values of docstrings
# TODO: Move dot product attention out to its own class
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

        self.softmax = nn.Softmax(3)
        self.dropout = nn.Dropout(dropout_rate)

    def dot_product_attention(self, Q, K, V) -> torch.Tensor:
        '''
        Dot-product attention 

        K: a tensor with shape (batch, heads, length, hidden_channels) 
        Q: a tensor with shape (batch, heads, length, hidden_channels) 
        V: a tensor with shape (batch, heads, length, output_channels)

        returns a tensor with shape (batch, heads, length, output_channels)
        '''
        d = Q.shape[-1]

        x = torch.matmul(Q, torch.transpose(K, 2, 3)) / np.sqrt(d)
        A = self.dropout(self.softmax(x))
                         
        out = torch.matmul(A, V)

        return out

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

        out = self.dot_product_attention(Q, K, V)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.total_out_channels)

        return out
