#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops, argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os, sys

from utils import (
    InputImageLayer,
    get_timm_networks,
    mixup_criterion,
    probability_maximizer_loss,
    score_maximizer_loss,
    imagenet_class_name_of,
    add_noise,
    RandomCircularShift,
)

from debug import (
    nonzero_grads,
    pixel_sample_ratio_map,
)


parser = argparse.ArgumentParser()

parser.add_argument("--IMG_SIZE", type=int, default=512)
parser.add_argument("--NET_INPUT_SIZE", type=int, default=224)
parser.add_argument("--LEARNING_RATE", type=float, default=0.01)
parser.add_argument("--ITERATIONS", type=int, default=1000000)
parser.add_argument("--BATCH_SIZE", type=int, default=32)
parser.add_argument("--CLASS", type=int, default=309)
parser.add_argument("--LOG_FREQUENCY", type=int, default=500)
parser.add_argument("--PARAM_FN", type=str, default="sigmoid")

parser.add_argument(
    "--NETWORKS",
    type=list,
    default=[
        "inception_v3",
    ],
)
cfg = parser.parse_args()
#%%

aug_fn = torch.nn.Sequential(
    RandomCircularShift(),
    kornia.augmentation.RandomRotation(
        degrees=180,
        same_on_batch=False,
        p=1,
    ),
    kornia.augmentation.RandomResizedCrop(
        size=(cfg.NET_INPUT_SIZE, cfg.NET_INPUT_SIZE),
        scale=(
            cfg.NET_INPUT_SIZE / cfg.IMG_SIZE,
            cfg.IMG_SIZE,
        ),
        ratio=(1, 1),  # aspect ratio
        same_on_batch=False,
    ),
    kornia.augmentation.RandomPerspective(
        distortion_scale=0.5,
        p=1,
        same_on_batch=False,
    ),
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.augmentation.RandomVerticalFlip(),
)

networks = get_timm_networks(cfg.NETWORKS)
input_img_layer = InputImageLayer(
    img_class=cfg.CLASS,
    img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
    param_fn=cfg.PARAM_FN,
    aug_fn=aug_fn,
).cuda()

optimizer = torch.optim.Adam(
    input_img_layer.parameters(),
    lr=cfg.LEARNING_RATE,
)


wandb.login()
with wandb.init(project="vis", config=cfg):
    wandb.watch(input_img_layer, log="all", log_freq=cfg.LOG_FREQUENCY)

    for n in tqdm.tqdm(range(cfg.ITERATIONS)):
        optimizer.zero_grad()
        net = random.choice(networks)
        imgs = input_img_layer(cfg.BATCH_SIZE)
        logits = net(imgs)
        loss = score_maximizer_loss(logits, cfg.CLASS)
        loss.backward()
        optimizer.step()

        if n % cfg.LOG_FREQUENCY == 0:
            tensor = input_img_layer.input_tensor
            img = input_img_layer.get_image()
            wandb.log(
                {
                    "loss": loss,
                    "tensor_max_value": tensor.max(),
                    "tensor_min_value": tensor.min(),
                    "image_max_value": img.max(),
                    "image_min_value": img.min(),
                    "tensor_grad_max_abs_value": tensor.grad.abs().max(),
                },
                step=n,
            )
        if n % cfg.LOG_FREQUENCY == 0:

            wandb.log(
                {
                    "image": wandb.Image(input_img_layer.get_image()),
                },
                step=n,
            )
