#%%
from numpy.lib.function_base import gradient
import torch, torchvision
import matplotlib.pyplot as plt
import kornia
from kornia import tensor_to_image as t2i
import tqdm
import random

import logging

logging.basicConfig(level=logging.WARN)

from src.utils import (
    InputImageLayer,
    get_timm_networks,
    probability_maximizer_loss,
    score_maximizer_loss,
    imagenet_class_name_of,
    add_noise,
    RandomCircularShift,
)

from src.debug import (
    nonzero_grads,
    pixel_sample_ratio_map,
)

IMG_SIZE = 512
NET_INPUT_SIZE = 224

input_img_layer = InputImageLayer(
    shape=[3, IMG_SIZE, IMG_SIZE],
    param_fn=torch.nn.Sequential(
        torch.nn.Sigmoid(),
    ),
).cuda()

aug_fn = torch.nn.Sequential(
    RandomCircularShift(),
    # kornia.augmentation.RandomRotation(
    #     degrees=90,
    #     same_on_batch=False,
    #     p=1,
    # ),
    kornia.augmentation.RandomResizedCrop(
        size=(NET_INPUT_SIZE, NET_INPUT_SIZE),
        scale=(0.1, 1),
        ratio=(0.8, 1.2),  # aspect ratio
        same_on_batch=False,
    ),
    # kornia.augmentation.RandomPerspective(
    #     distortion_scale=0.5,
    #     p=1,
    #     same_on_batch=False,
    # ),
    # kornia.augmentation.RandomHorizontalFlip(),
    # kornia.augmentation.RandomVerticalFlip(),
)


network_names = [
    # "densenet121",  # good
    # "resnet50",  # good
    "resnet18",  # good
    # "efficientnet_b4",  # bad
    # "inception_v4",  # good
    # "vit_base_patch16_224",  # meh
]

networks = get_timm_networks(network_names)
optimizer = torch.optim.Adam(input_img_layer.parameters(), lr=0.025)


#%% train
ITERATIONS = 100
BATCH_SIZE = 16
for TARGET_CLASS in [309]:
    for n in tqdm.tqdm(range(ITERATIONS)):
        net = random.choice(networks)
        input_imgs = input_img_layer(BATCH_SIZE)
        aug_imgs = aug_fn(input_imgs)
        out = net(aug_imgs)

        prob_loss = probability_maximizer_loss(out, TARGET_CLASS)
        score_loss = score_maximizer_loss(out, TARGET_CLASS)
        loss = score_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

# %%

plt.figure(figsize=[10, 10])

plt.imshow(t2i(input_img_layer(1)))
plt.show()
# %%
with torch.no_grad():
    input_imgs = input_img_layer(32)
    aug_imgs = aug_fn(input_imgs)
    out = net(aug_imgs)
    probs = torch.softmax(out, dim=-1)
