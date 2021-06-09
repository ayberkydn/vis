import torch, torchvision
import matplotlib.pyplot as plt
import kornia
from kornia import tensor_to_image as t2i
from kornia import image_to_tensor as i2t
import cv2
import tqdm
import numpy as np
import einops
import timm
import numpy as np
from PIL import Image
import random
import os
from utils import InputImageLayer
from utils import get_timm_networks, probability_maximizer_loss


aug_fn = torch.nn.Sequential(
    kornia.augmentation.RandomRotation(
        degrees=30,
        same_on_batch=False,
    ),
    kornia.augmentation.RandomResizedCrop(
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(0.5, 2.0),  # aspect ratio
        same_on_batch=False,
    ),
)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

input_img_layer = InputImageLayer(
    shape=[3, 512, 512],
    param_fn=torch.nn.Sequential(
        torch.nn.Sigmoid(),
    ),
).cuda()

network_names = [
    "vit_base_patch16_224",
    # "resnet50",
    # "tf_efficientnet_b0_ap",
    # "adv_inception_v3",
    # "inception_v4",
    # "dla102",
    # "densenet121",
    # "mobilenetv3_large_100",
    # "resnext101_32x8d",
    # "seresnet152d",
    # "ig_resnext101_32x16d",
    # "nf_resnet50",
]

networks = get_timm_networks(network_names)

optimizer = torch.optim.Adam(input_img_layer.parameters(), lr=0.01)

TARGET_CLASS = 35
ITERATIONS = 100
BATCH_SIZE = 1
TV_LOSS_COEFF = 0

#%%
for n in tqdm.tqdm(range(ITERATIONS)):

    input_imgs = input_img_layer(BATCH_SIZE)

    net = random.choice(networks)

    aug_imgs = aug_fn(input_imgs)

    out = net(normalize(aug_imgs))

    loss = probability_maximizer_loss(out, TARGET_CLASS)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

# %%
final_img = input_img_layer(1)
plt.figure(figsize=(10, 10))
plt.imshow(t2i(final_img))
plt.show()

# %%
