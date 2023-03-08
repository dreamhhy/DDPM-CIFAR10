# Author: Hongyuan He
# Time: 2023.2.28


import torch
import torch.nn.functional as F
from model.diffusion import q_sample


# 3种损失函数
def p_losses(denoise_model, x_start, t, noise=None, loss_type='l1'):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    
    return loss
