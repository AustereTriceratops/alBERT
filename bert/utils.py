import torch
import numpy as np

def create_position_encodings(batch_size, seq_len, dims):
    result = torch.zeros((seq_len, dims))
    
    for i in range(seq_len):
        for k in range(0, dims, 2):
            denom = np.power(10000, k/dims)
            fac = i / denom
            
            result[i][k] = np.sin(fac)
            result[i][k + 1] = np.cos(fac)
    
    result = result.unsqueeze(0).repeat(batch_size, 1, 1)
    return result