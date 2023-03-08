# Author: Hongyuan He
# Time: 2023.2.25

from inspect import isfunction
from torch import nn


def exists(x):
    # if x is None:
    #   return False
    # else:
    #   return True
    return x is not None

# 如果val非None则返回val，否则(如果d为函数则返回d(),否则返回d)
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    # 当函数中以列表或者元组的形式传参时，就要使用*args；当传入字典形式的参数时，就要使用**kwargs
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)