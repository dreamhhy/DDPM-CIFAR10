# Author: Hongyuan He
# Time: 2023.2.26


import torch
from torch import nn
from functools import partial
from DDPM_CIFAR10.utils.network_helpers import Downsample, Residual, Upsample, default, exists
from DDPM_CIFAR10.model.ResnetBlock import ConvNextBlock, ResnetBlock
from DDPM_CIFAR10.model.attention import LinearAttention, Attention
from DDPM_CIFAR10.model.preNorm import PreNorm

# from utils.network_helpers import Downsample, Residual, Upsample, default, exists
from DDPM_CIFAR10.utils.position_embeddings import SinusoidalPositionEmbeddings

class Unet(nn.Module):
    '''
    input of the network is a batch of noisy images of shape (batch_size, num_channels, height, width)
    and a batch of noise levels of shape (batch_size, 1)

    output is a tensor of shape (batch_size, num_channels, height, width)
    '''
    def __init__(self,
                 dim,   # dim=image_size=28
                 init_dim=None, # 默认为None，最终取dim // 3 * 2
                 out_dim=None, # 默认为None，最终取channels
                 dim_mults=(1,2,4,8),
                 channels=3, # 通道数默认为3
                 with_time_emb=True,    # 是否使用embeddings
                 resnet_block_groups=8, # 如果使用ResnetBlock，groups=resnet_block_groups
                 use_convnext=True, # 是True使用ConvNextBlock，是Flase使用ResnetBlock
                 convnext_mult=2, # 如果使用ConvNextBlock，mult=convnext_mult
                 ):
        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # [init_dim, dim, 2*dim, 4*dim, 8*dim]
        '''
        print(dims)
        print(dims[:-1])
        print(dims[1:])
        [18, 28, 56, 112, 224]
        [18, 28, 56, 112]
        [28, 56, 112, 224]
        '''
        in_out = list(zip(dims[:-1], dims[1:])) # [(18, 28), (28, 56), (56, 112), (112, 224)]

        if use_convnext:
            block_class = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList(
                [
                    block_class(dim_in, dim_out, time_emb_dim=time_dim), ########################?????????????
                    block_class(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ]
            ))

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList(
                [
                    block_class(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    block_class(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                ]
            ))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_class(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # Down sample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)




if __name__ == '__main__':
    batch_size = 8
    model = Unet(dim=28, channels=3, dim_mults=(1, 2, 4))
    x = torch.randn(batch_size, 3, 28, 28)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)

