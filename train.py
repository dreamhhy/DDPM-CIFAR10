# Author: Hongyuan He 
# Time: 2023.2.28


import torch
from pathlib import Path

import torchvision
from torch.optim import Adam
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from DDPM_CIFAR10.model.UNet import Unet
from DDPM_CIFAR10.model.diffusion import sample
from DDPM_CIFAR10.utils.dataset import load_data
from DDPM_CIFAR10.utils.loss import p_losses

# dataset = load_dataset("fashion_mnist")
#
# image_size = 28
# channels = 1
# batch_size = 128
#
# transform = Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda t: (t * 2) - 1)
# ])
#
# def transforms(examples):
#     examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
#     del examples["image"]
#
#     return examples
#
# transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
#
# dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

image_size = 28
channels = 3
batch_size = 128

transform = Compose([
           transforms.RandomHorizontalFlip(), # 随机水平翻转图片
           transforms.ToTensor(), # 转成张量
           transforms.Lambda(lambda t: (t * 2) - 1) # 归一化到[-1,1]
])

# dataset = torchvision.datasets.FashionMNIST(
#   root="../data", train=True, transform=transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, drop_last=True)


batch = next(iter(dataloader))
print(batch[0].shape)

# setting save images

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim=image_size, 
    channels=channels, 
    dim_mults=(1, 2, 4)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 1
timesteps = 200

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # batch_size = batch["pixel_values"].shape[0]
        # batch = batch["pixel_values"].to(device)

        batch_size = batch[0].shape[0]
        batch = batch[0].to(device)

        t = torch.randint(0, timesteps, (batch_size, ), device=device).long()
        loss = p_losses(model, batch, t, loss_type='huber')

        if step % 100 == 0:
            print(f'Loss: {loss.item()}')

        loss.backward()
        optimizer.step()

        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)


# sample 64 images
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")

random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()
