# Author: Hongyuan He
# Time: 2023.2.25

import math
import torch
from torch import nn

# class SinusoidalPositionEmbeddings(nn.Module):
#     """
#     input: (bs, 1), i.e. the noise levels of several noisy images in a batch
#     output: (bs, dim)
#     """
#     def __init__(self, dim):
#         '''
#         dim: the dimensionality of the position embeddings
#         '''
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings