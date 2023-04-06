import torch
import torch.nn as nn
import torch.nn.functional as F

import attention
import utils

class EncoderLayer(nn.Module):
    def __init__(self, seq_len, model_dims, key_dims=None, value_dims=None, hidden_dims=None, heads=1, dropout_rate=0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = model_dims

        self.attention = attention.MultiHeadAttention(
            model_dims, key_dims=key_dims, value_dims=value_dims, head=heads, dropout_rate=dropout_rate
        )
        self.fc1 = nn.Linear(model_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, model_dims)

        self.position_encoding = utils.create_position_encodings(seq_len, model_dims)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(model_dims)
    
    def forward(self, x):
        batch_size = x.shape[0]
        batch_position_encoding = self.position_encoding.repeat(batch_size, 1, 1)

        out = x + batch_position_encoding
        out = self.dropout(out)
        out = self.norm(out)

        out = self.attention(out)

        residual = out

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.dropout(out)

        out += residual
        out = self.norm(out)

        return out
