#%%
import torch, torchvision
import matplotlib.pyplot as plt
import kornia
from kornia import tensor_to_image as t2i
import tqdm
import random

import logging

logging.basicConfig(level=logging.INFO)

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

IMG_SIZE = 256
NET_INPUT_SIZE = 224

input_img_layer = InputImageLayer(
    shape=[3, IMG_SIZE, IMG_SIZE],
    param_fn=torch.nn.Sequential(
        torch.nn.Sigmoid(),
    ),
).cuda()

aug_fn = torch.nn.Sequential(
    RandomCircularShift(),
    kornia.augmentation.RandomRotation(
        degrees=90,
        same_on_batch=False,
        p=1,
    ),
    kornia.augmentation.RandomResizedCrop(
        size=(224, 224),
        scale=(0.1, 1),
        ratio=(0.8, 1.2),  # aspect ratio
        same_on_batch=False,
        # resample="bicubic",
    ),
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomVerticalFlip(),
)


network_names = [
    # "densenet121",  # good
    # "resnet50",  # good
    "resnet18",  # good
    # "efficientnet_b4",  # bad
    # "efficientnet_b1",  # bad
    # "efficientnet_b1_pruned",  # bad
    # "inception_v4",  # good
    # "adv_inception_v3",  # good
    # "vit_base_patch16_224",  # meh
]

networks = get_timm_networks(network_names)
optimizer = torch.optim.Adam(input_img_layer.parameters(), lr=0.01)

#%%
plt.imshow(t2i(nonzero_grads(input_img_layer, aug_fn)))
plt.show()


#%%
sample_ratio_map = pixel_sample_ratio_map(
    input_img_layer, aug_fn, times=10, sample_size=64
)
print(f"Max %: {sample_ratio_map.max().item()}")
print(f"Min %: {sample_ratio_map.min().item()}")
print(f"Mean %: {sample_ratio_map.mean().item()}")
print(f"Std %: {sample_ratio_map.std().item()}")
#%%

#%% train
ITERATIONS = 10000
BATCH_SIZE = 8
for TARGET_CLASS in [309]:
    for n in tqdm.tqdm(range(ITERATIONS)):
        net = random.choice(networks)
        input_imgs = input_img_layer(BATCH_SIZE)
        aug_imgs = aug_fn(input_imgs)
        out = net(aug_imgs)

        prob_loss = probability_maximizer_loss(out, TARGET_CLASS)
        score_loss = score_maximizer_loss(out, TARGET_CLASS)
        loss = (prob_loss + score_loss) / 2
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

# %%

plt.figure(figsize=[10, 10])
plt.imshow(t2i(input_img_layer(1)))
plt.show()
# %%
