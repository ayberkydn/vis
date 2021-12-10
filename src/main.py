#%%
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--IMG_SIZE", type=int, default=512)
parser.add_argument("--LEARNING_RATE", type=float, default=0.025)
parser.add_argument("--ITERATIONS", type=int, default=3000)
parser.add_argument("--BATCH_SIZE", type=int, default=8)
parser.add_argument("--PARAM_FN", type=str, default="sigmoid")

parser.add_argument("--LOSS_SCORE_COEFF", type=float, default=1)
parser.add_argument("--LOSS_PROB_COEFF", type=float, default=0)
parser.add_argument("--LOSS_BN_COEFF", type=float, default=100)
parser.add_argument("--LOSS_DIV_COEFF", type=float, default=100)
parser.add_argument("--LOSS_TV_COEFF", type=float, default=0)

parser.add_argument("--AUG_FLIP", type=bool, default=False)
parser.add_argument("--AUG_ROTATE_DEGREES", type=int, default=15)

parser.add_argument("--CLASSES", default=[76, 254, 309, 340, 445, 851, 949, 988])
parser.add_argument("--NETWORK", type=str, default="resnet18")

args = parser.parse_args()


#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops, datetime
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from input_img_layer import InputImageLayer
from losses import (
    tv_loss_fn,
    softmax_loss_fn,
    score_loss_fn,
)

from hook_wrappers import (
    BNStatsLossWrapper,
    ConvSimilarityLossWrapper,
)

from augmentations import RandomCircularShift
from utils import imagenet_class_name_of, get_timm_network

# with wandb.init(project="vis", config=args, mode="disabled") as run:
with wandb.init(project="vis-denemeler", config=args) as run:
    cfg = wandb.config

    # device = f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu"
    device = "cuda:0"
    torch.backends.cudnn.benchmark = True

    net, mean, std, input_size = get_timm_network(cfg.NETWORK, device=device)
    aug_fn = torch.nn.Sequential(
        RandomCircularShift(),
        kornia.augmentation.RandomResizedCrop(
            size=input_size,
            scale=(
                0.9 * input_size[0] / cfg.IMG_SIZE,
                1.1 * input_size[0] / cfg.IMG_SIZE,
            ),
            ratio=(1, 1),  # aspect ratio
            same_on_batch=False,
        ),
        kornia.augmentation.RandomRotation(
            degrees=cfg.AUG_ROTATE_DEGREES,
            same_on_batch=False,
            p=1,
        ),
        kornia.augmentation.RandomHorizontalFlip(p=0.5 * cfg.AUG_FLIP),
        kornia.augmentation.RandomVerticalFlip(p=0.5 * cfg.AUG_FLIP),
    )

    net_convsim = ConvSimilarityLossWrapper(net)
    net_bnstats = BNStatsLossWrapper(net)

    in_layer = InputImageLayer(
        img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
        classes=cfg.CLASSES,
        param_fn=cfg.PARAM_FN,
        aug_fn=aug_fn,
    ).to(device)

    optimizer = torch.optim.RAdam(
        in_layer.parameters(),
        lr=cfg.LEARNING_RATE,
    )

    with tqdm.tqdm(total=cfg.ITERATIONS + 1, disable=cfg.DISABLE_TQDM) as pbar:
        # for n in tqdm.tqdm(range(cfg.ITERATIONS + 1), disable=cfg.DISABLE_TQDM):
        for n in range(cfg.ITERATIONS + 1):
            pbar.update()
            optimizer.zero_grad(set_to_none=True)
            bn_losses = []
            score_losses = []
            prob_losses = []
            tv_losses = []
            similarity_losses = []
            losses = []
            for idx in range(in_layer.n_classes):
                random_idx = random.choices(range(in_layer.n_classes), k=cfg.BATCH_SIZE)
                same_idx = [idx] * cfg.BATCH_SIZE

                rand_imgs, rand_classes = in_layer(random_idx)
                rand_logits, bnstats_loss = net_bnstats(rand_imgs)

                same_imgs, same_classes = in_layer(same_idx)
                same_logits, similarity_loss = net_convsim(same_imgs)

                imgs = torch.cat([same_imgs, rand_imgs], dim=0)
                logits = torch.cat([same_logits, rand_logits], dim=0)
                classes = torch.cat([same_classes, rand_classes], dim=0)

                prob_loss = softmax_loss_fn(logits, classes)
                score_loss = score_loss_fn(logits, classes)
                tv_loss = tv_loss_fn(imgs)

                score_losses.append(score_loss.item())
                prob_losses.append(prob_loss.item())
                tv_losses.append(tv_loss.item())
                similarity_losses.append(similarity_loss.item())
                bn_losses.append(bnstats_loss.item())

                loss = (
                    score_loss * cfg.LOSS_SCORE_COEFF
                    + similarity_loss * cfg.LOSS_DIV_COEFF
                    + prob_loss * cfg.LOSS_PROB_COEFF
                    + tv_loss * cfg.LOSS_TV_COEFF
                    + bnstats_loss * cfg.LOSS_BN_COEFF
                )
                loss.backward()

            optimizer.step()

            log_dict = {
                "losses/score_loss": np.mean(score_losses),
                "losses/prob_loss": np.mean(prob_losses),
                "losses/tv_loss": np.mean(tv_losses),
                "losses/div_loss": np.mean(similarity_losses),
                "losses/bn_loss": np.mean(bn_losses),
            }

            total = pbar.format_dict["total"]
            rate = pbar.format_dict["rate"]
            elapsed = pbar.format_dict["elapsed"]
            if rate != None:
                remaining_secs = total / rate - elapsed

                log_dict.update(
                    {
                        "timer/remaining_mins": remaining_secs / 60,
                        "timer/iter_per_seconds": rate,
                        "timer/elapsed_mins": elapsed / 60,
                    }
                )

            if n % 100 == 0:
                tensor = in_layer.input_tensor
                log_imgs = in_layer.get_images()
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
                        "images": wandb_imgs,
                    },
                )

            wandb.log(
                log_dict,
                step=n,
                commit=True,
            )
