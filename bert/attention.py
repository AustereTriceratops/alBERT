import torch
import torch.nn as nn

import numpy as np

class Attention(nn.Module):
    def __init__(self, input_channels, output_channels=None, hidden_channels=None, dropout_rate=0.1) -> None:
        super().__init__()

        if not output_channels:
            output_channels = input_channels
        if not hidden_channels:
            hidden_channels = input_channels


        self.Q_projection = nn.Conv1d(kernel_size=1, in_channels=input_channels, out_channels=hidden_channels)
        self.K_projection = nn.Conv1d(kernel_size=1, in_channels=input_channels, out_channels=hidden_channels)
        self.V_projection = nn.Conv1d(kernel_size=1, in_channels=input_channels, out_channels=output_channels)

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
        out = torch.matmul(x, V)

        return out


class MultiheadAttention(Attention):
    def __init__(self, channels_V, channels_out) -> None:
        super().__init__()

        self.linear = nn.Conv1d(kernel_size=1, in_channels=channels_V, out_channels=channels_out)

    def forward(self, Q, K, V) -> torch.Tensor:
        '''
        Multi-head attention attention 

        Q: a tensor with shape (batch, length, channels) 
        K: a tensor with shape (batch, length, channels) 
        V: a tensor with shape (batch, length, channels_) 

        TODO: implement
        TODO: make compatible with dropout
        '''
        pass
