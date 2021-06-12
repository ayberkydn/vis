#%%
import torch, torchvision
import matplotlib.pyplot as plt
import kornia
from kornia import tensor_to_image as t2i
import tqdm
import random

from src.utils import (
    InputImageLayer,
    get_timm_networks,
    probability_maximizer_loss,
    score_maximizer_loss,
    imagenet_class_name_of,
    show_nonzero_grads,
)

IMG_SIZE = 256
NET_INPUT_SIZE = 224

input_img_layer = InputImageLayer(
    shape=[3, IMG_SIZE, IMG_SIZE],
    param_fn=torch.nn.Sequential(
        torch.nn.Sigmoid(),
        # kornia.filters.MedianBlur([5, 5]),
        # torchvision.transforms.GaussianBlur(5, sigma=2)
    ),
).cuda()

aug_fn = torch.nn.Sequential(
    # torchvision.transforms.Pad(
    #     padding_mode="reflect",
    #     padding=[128, 128],
    # ),
    kornia.augmentation.RandomResizedCrop(
        size=(224, 224),
        scale=(224 / 256, 224 / 256),
        ratio=(1, 1),  # aspect ratio
        same_on_batch=False,
        resample="bicubic",
    ),
    kornia.augmentation.RandomRotation(
        degrees=0,
        same_on_batch=False,
    ),
)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


network_names = [
    "resnet50",
    "inception_v4",
    "densenet121",
]

networks = get_timm_networks(network_names)

optimizer = torch.optim.Adam(input_img_layer.parameters(), lr=0.05)
#%% train
TARGET_CLASS = 309
ITERATIONS = 5000
BATCH_SIZE = 8

for n in tqdm.tqdm(range(ITERATIONS)):
    for net in networks:
        aug_net = add_noise()
        input_imgs = input_img_layer(BATCH_SIZE)
        aug_imgs = aug_fn(input_imgs)
        out = net(normalize(aug_imgs))
        prob_loss = probability_maximizer_loss(out, TARGET_CLASS) / len(networks)
        loss = prob_loss
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
#%%


show_nonzero_grads(input_img_layer, aug_fn)

#%%


def validate(networks, img_layer, aug_fn, normalize):
    with torch.no_grad():
        min_prob = 1
        for n in range(32):
            imgs = img_layer
            input_imgs = input_img_layer(32)
            aug_imgs = aug_fn(input_imgs)
            out = net(normalize(aug_imgs))
            probs = torch.softmax(out, dim=-1)[:, TARGET_CLASS]
            min_prob = probs.min() if probs.min() < min_prob else min_prob
        print(min_prob)


# %%
final_img = input_img_layer(1)
plt.figure(figsize=(10, 10))
plt.imshow(t2i(final_img))
plt.show()
plt.savefig("bee.png")
plt.close()
