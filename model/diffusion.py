# Author: Hongyuan He  
# Time: 2023.2.28

import torch
import torch.nn.functional as F
from utils.beta_schemes import linear_beta_schedule
from tqdm.auto import tqdm


timesteps = 200
betas = linear_beta_schedule(timesteps=timesteps)

alphas = 1. - betas
alpha_cumprod = torch.cumprod(alphas, axis=0)   # cumulative product
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alpha_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) # 直接取出第t位的值
    return out.reshape(batch_size, *((1, ) * (len(x_shape) -1))).to(t.device)   # [bs, 1, 1, 1], len(x_shape)=4

# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, t, t_index):
    '''
    t: i.e. [i, i, i, i, i] len=batch size
    '''
    betas_t = extract(betas, t, x.shape)    # [bs, 1], i.e. [beta_i, beta_i, beta_i, beta_i].T
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # x_{t-1}
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.rand_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise    # 看paper第四页原公式

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


