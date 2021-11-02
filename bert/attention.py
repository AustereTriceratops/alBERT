import torch
import torch.nn as nn

import numpy as np

class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # softmax will be applied over the channels 
        self.softmax = nn.Softmax(2)


    def forward(self, Q, K, V) -> torch.Tensor:
        '''
        Dot-product attention 

        Q: a tensor with shape (batch, length, channels) 
        K: a tensor with shape (batch, length, channels) 
        V: a tensor with shape (batch, length, channels) 

        TODO: make compatible with dropout
        '''
        d = Q.shape[2]

        x = torch.matmul(Q, torch.transpose(K, 1, 2)) / np.sqrt(d)
        x = self.softmax(x)
        attn = torch.matmul(x, V)

        return attn