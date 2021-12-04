#%%
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--IMG_SIZE", type=int, default=512)
parser.add_argument("--NET_INPUT_SIZE", type=int, default=224)
parser.add_argument("--LEARNING_RATE", type=float, default=0.1)
parser.add_argument("--ITERATIONS", type=int, default=5000)
parser.add_argument("--BATCH_SIZE", type=int, default=64)
parser.add_argument("--CLASS", type=int, default=309)
parser.add_argument("--LOG_FREQUENCY", type=int, default=100)
parser.add_argument("--PARAM_FN", type=str, default="sigmoid")
parser.add_argument("--SCORE_LOSS_COEFF", type=float, default=1)
parser.add_argument("--PROB_LOSS_COEFF", type=float, default=0)
parser.add_argument("--BN_LOSS_COEFF", type=float, default=0.1)
parser.add_argument("--TV_LOSS_COEFF", type=float, default=0.1)

parser.add_argument("--NETWORK", type=str, default="vgg11_bn")
#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops
import matplotlib.pyplot as plt
import os, sys

from utils import (
    InputImageLayer,
    bn_stats_loss,
    get_timm_network,
    mixup_criterion,
    probability_maximizer_loss,
    score_maximizer_loss,
    imagenet_class_name_of,
    add_noise,
    RandomCircularShift,
    # RollWrapper,
)

from debug import (
    nonzero_grads,
    pixel_sample_ratio_map,
)

#%%

torch.backends.cudnn.benchmark = True
args = parser.parse_args(args=[])
# with wandb.init(project="vis", config=args, mode="disabled") as run:
with wandb.init(project="vis", config=args) as run:
    cfg = wandb.config
    aug_fn = torch.nn.Sequential(
        RandomCircularShift(),
        kornia.augmentation.RandomResizedCrop(
            size=(cfg.NET_INPUT_SIZE, cfg.NET_INPUT_SIZE),
            scale=(
                cfg.NET_INPUT_SIZE / cfg.IMG_SIZE,
                1,
            ),
            ratio=(1, 1),  # aspect ratio
            same_on_batch=False,
        ),
        kornia.augmentation.RandomRotation(
            degrees=180,
            same_on_batch=False,
            p=1,
        ),
        kornia.augmentation.RandomPerspective(
            distortion_scale=0.5,
            p=1,
            same_on_batch=False,
        ),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomVerticalFlip(),
    )

    net = get_timm_network(cfg.NETWORK)
    input_img_layer = InputImageLayer(
        img_class=cfg.CLASS,
        img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
        param_fn=cfg.PARAM_FN,
        aug_fn=aug_fn,
    ).cuda()

    optimizer = torch.optim.RAdam(
        input_img_layer.parameters(),
        lr=cfg.LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.99,
        patience=100,
        threshold=1e-4,
    )

    wandb.watch(input_img_layer, log="all", log_freq=cfg.LOG_FREQUENCY)

    for n in tqdm.tqdm(range(cfg.ITERATIONS + 1)):
        optimizer.zero_grad(set_to_none=True)
        imgs = input_img_layer(cfg.BATCH_SIZE)
        logits, activations = net(imgs)

        prob_loss = probability_maximizer_loss(logits, cfg.CLASS)
        score_loss = score_maximizer_loss(logits, cfg.CLASS)
        bn_loss = bn_stats_loss(activations)
        tv_loss = kornia.losses.total_variation(imgs).mean() / (
            imgs.shape[-1] * imgs.shape[-2]
        )

        loss = (
            bn_loss * cfg.BN_LOSS_COEFF
            + score_loss * cfg.SCORE_LOSS_COEFF
            + prob_loss * cfg.PROB_LOSS_COEFF
            + tv_loss * cfg.TV_LOSS_COEFF
        )
        loss.backward()
        scheduler.step(loss)
        optimizer.step()

        log_dict = {
            "losses/prob_loss": prob_loss,
            "losses/score_loss": score_loss,
            "losses/bn_loss": bn_loss,
            "losses/tv": tv_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if n % cfg.LOG_FREQUENCY == 0:
            tensor = input_img_layer.input_tensor
            img = input_img_layer.get_image()
            log_dict.update(
                {
                    "tensor_stats/max_value": tensor.max(),
                    "tensor_stats/grad_max_abs": tensor.grad.abs().max(),
                    "tensor_stats/grad_mean_abs": tensor.grad.abs().mean(),
                    "tensor_stats/grad_std": tensor.grad.std(),
                    "tensor_stats/min_value": tensor.min(),
                    "image_max_value": img.max(),
                    "image_min_value": img.min(),
                    "image": wandb.Image(input_img_layer.get_image()),
                },
            )

        wandb.log(
            log_dict,
            step=n,
            commit=True,
        )
