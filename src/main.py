#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops
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

wandb.login()


@dataclass
class TrainConfig:
    GPU: int
    IMG_SIZE: int
    NET_INPUT_SIZE: int
    NETWORKS: str
    LEARNING_RATE: float
    ITERATIONS: int
    BATCH_SIZE: int
    CLASS: int
    LOG_FREQUENCY: int
    PARAM_FN: str


cfg = TrainConfig(
    GPU=2,
    IMG_SIZE=512,
    NET_INPUT_SIZE=224,
    NETWORKS=[
        # "densenet121",
        "inception_v3",
        # "resnet50",
    ],
    LEARNING_RATE=0.01,
    ITERATIONS=100000,
    BATCH_SIZE=8,
    CLASS=309,
    LOG_FREQUENCY=100,
    PARAM_FN="sin",
)
################################################################################

torch.cuda.device(cfg.GPU)

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
