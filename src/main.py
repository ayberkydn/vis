#%%
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--GPU", type=int, default=0)
parser.add_argument("--IMG_SIZE", type=int, default=512)
parser.add_argument("--NET_INPUT_SIZE", type=int, default=224)
parser.add_argument("--LEARNING_RATE", type=float, default=0.01)
parser.add_argument("--ITERATIONS", type=int, default=10000)
parser.add_argument("--BATCH_SIZE", type=int, default=8)
parser.add_argument("--LOG_FREQUENCY", type=int, default=100)
parser.add_argument("--PARAM_FN", type=str, default="sigmoid")

parser.add_argument("--LOSS_SCORE_COEFF", type=float, default=1)
parser.add_argument("--LOSS_PROB_COEFF", type=float, default=0)
parser.add_argument("--LOSS_BN_COEFF", type=float, default=0)
parser.add_argument("--LOSS_TV_COEFF", type=float, default=50)

parser.add_argument("--AUG_H_FLIP", type=bool, default=False)
parser.add_argument("--AUG_V_FLIP", type=bool, default=False)
parser.add_argument("--AUG_ROTATE_DEGREES", type=int, default=30)

parser.add_argument("--CLASSES", type=int, default=[309, 340, 851])
# parser.add_argument("--CLASSES", type=int, default=list(range(10)))
parser.add_argument("--NETWORK", type=str, default="swin_base_patch4_window7_224")

args = parser.parse_args()


#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from input_img_layer import InputImageLayer
from losses import (
    bn_stats_loss,
    probability_maximizer_loss,
    score_maximizer_loss,
)
from utils import (
    BNStatsModelWrapper,
    get_timm_network,
    imagenet_class_name_of,
    RandomCircularShift,
)

# with wandb.init(project="vis", config=args, mode="disabled") as run:
with wandb.init(project="vis", config=args) as run:
    cfg = wandb.config

    device = f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    aug_fn = torch.nn.Sequential(
        RandomCircularShift(),
        kornia.augmentation.RandomResizedCrop(
            size=(cfg.NET_INPUT_SIZE, cfg.NET_INPUT_SIZE),
            scale=(
                0.5 * cfg.NET_INPUT_SIZE / cfg.IMG_SIZE,
                1,
            ),
            ratio=(1, 1),  # aspect ratio
            same_on_batch=False,
        ),
        kornia.augmentation.RandomRotation(
            degrees=cfg.AUG_ROTATE_DEGREES,
            same_on_batch=False,
            p=1,
        ),
        kornia.augmentation.RandomPerspective(
            distortion_scale=0.5,
            p=1,
            same_on_batch=False,
        ),
        kornia.augmentation.RandomHorizontalFlip(p=0.5 * cfg.AUG_H_FLIP),
        kornia.augmentation.RandomVerticalFlip(p=0.5 * cfg.AUG_V_FLIP),
    )

    net = get_timm_network(cfg.NETWORK, device=device)
    net = BNStatsModelWrapper(net)

    input_img_layer = InputImageLayer(
        img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
        classes=cfg.CLASSES,
        param_fn=cfg.PARAM_FN,
        aug_fn=aug_fn,
    ).to(device)

    optimizer = torch.optim.RAdam(
        input_img_layer.parameters(),
        lr=cfg.LEARNING_RATE,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     factor=0.99,
    #     patience=100,
    #     threshold=1e-4,
    # )

    for n in tqdm.tqdm(range(cfg.ITERATIONS + 1)):
        optimizer.zero_grad(set_to_none=True)
        bn_losses = []
        score_losses = []
        prob_losses = []
        tv_losses = []
        losses = []
        for idx in range(input_img_layer.num_classes):
            imgs, classes = input_img_layer(cfg.BATCH_SIZE)
            logits, activations = net(imgs)

            prob_loss = probability_maximizer_loss(logits, classes)
            score_loss = score_maximizer_loss(logits, classes)
            if len(activations) > 0:
                bn_loss = bn_stats_loss(activations)
            else:
                bn_loss = torch.tensor(0)
            tv_loss = kornia.losses.total_variation(imgs).mean() / (
                imgs.shape[-1] * imgs.shape[-2]
            )

            loss = (
                bn_loss * cfg.LOSS_BN_COEFF
                + score_loss * cfg.LOSS_SCORE_COEFF
                + prob_loss * cfg.LOSS_PROB_COEFF
                + tv_loss * cfg.LOSS_TV_COEFF
            )
            prob_losses.append(prob_loss.item())
            score_losses.append(score_loss.item())
            bn_losses.append(bn_loss.item())
            tv_losses.append(tv_loss.item())
            losses.append(loss.item())
            loss.backward()

        loss = np.mean(losses)
        # scheduler.step(loss)
        optimizer.step()

        log_dict = {
            "losses/prob_loss": np.mean(prob_losses),
            "losses/score_loss": np.mean(score_losses),
            "losses/bn_loss": np.mean(bn_losses),
            "losses/tv": np.mean(tv_losses),
            "lr": optimizer.param_groups[0]["lr"],
        }

        if n % cfg.LOG_FREQUENCY == 0:
            tensor = input_img_layer.input_tensor
            log_imgs = input_img_layer.get_images()
            wandb_imgs = [
                wandb.Image(img, caption=imagenet_class_name_of(cfg.CLASSES[n]))
                for n, img in enumerate(log_imgs)
            ]

            log_dict.update(
                {
                    "tensor_stats/max_value": tensor.max(),
                    "tensor_stats/grad_max_abs": tensor.grad.abs().max(),
                    "tensor_stats/grad_mean_abs": tensor.grad.abs().mean(),
                    "tensor_stats/grad_std": tensor.grad.std(),
                    "tensor_stats/min_value": tensor.min(),
                    # "image_max_value": imgs.max(),
                    # "image_min_value": imgs.min(),
                    "images": wandb_imgs,
                },
            )

        wandb.log(
            log_dict,
            step=n,
            commit=True,
        )
