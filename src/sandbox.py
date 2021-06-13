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
    add_noise,
)

IMG_SIZE = 512
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
    torchvision.transforms.Pad(
        padding_mode="reflect",
        padding=[64, 64],
    ),
    kornia.augmentation.RandomRotation(
        degrees=45,
        same_on_batch=False,
    ),
    kornia.augmentation.RandomResizedCrop(
        size=(224, 224),
        scale=(0.25, 1),
        ratio=(0.5, 2),  # aspect ratio
        same_on_batch=False,
    ),
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomVerticalFlip(),
)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


network_names = [
    "resnet18",
    "resnet34",
    "resnet50",
]

networks = get_timm_networks(network_names)

optimizer = torch.optim.Adam(input_img_layer.parameters(), lr=0.01)
#%% train
TARGET_CLASS = 1000
ITERATIONS = 10000
BATCH_SIZE = 8

for n in tqdm.tqdm(range(ITERATIONS)):
    net = random.choice(networks)
    input_imgs = input_img_layer(BATCH_SIZE)
    aug_imgs = aug_fn(input_imgs)
    out = net(normalize(aug_imgs))
    prob_loss = probability_maximizer_loss(out, TARGET_CLASS)
    score_loss = score_maximizer_loss(out, TARGET_CLASS)
    loss = (prob_loss + score_loss) / 2
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
# %%
final_img = input_img_layer(1)
plt.figure(figsize=(10, 10))
plt.imshow(t2i(final_img))
plt.show()
plt.close()

# %%
