# Author: Hongyuan He
# Time: 2023.2.26

from torch import nn, einsum
from einops import rearrange
import torch

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)    #与cat原理相反， 将数组拆开成三份， dim=0按照行拆， 1按照列拆开
        # qkv = [[b, hidden_dim, h, w], [b, hidden_dim, h, w], [b, hidden_dim, h, w]]
        # qkv为一个元组，其中每一个元素的大小为torch.Size([b, hidden_dim, h, w])
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )   # [b, hidden_dim, h, w] -> [b, heads, dim_heads, h*w]
        '''
        实际应该是[b, heads, h*w, dim_heads], 但因为用的是einsum计算, 所以不用这么变换
        '''
        q = q * self.scale  # 扩大q

        sim = einsum("b h d i, b h d j -> b h i j", q, k)   # [b, heads, h*w, h*w]
        # c_ik = a_ij * b_jk -> einsum('ij, jk -> ik', a, b)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() # dim=-1, 矩阵每一行
        attn = sim.softmax(dim=-1)  # 对h*w * h*w中每一行求softmax

        out = einsum("b h i j, b h d j -> b h i d", attn, v) # [b, heads, h*w, dim_heads]
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)    # [b, hidden_dim, h, w]
        return self.to_out(out) # [b, dim, h, w]


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

