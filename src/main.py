#%%
import argparse


cfg_parser = argparse.ArgumentParser()

cfg_parser.add_argument("--IMG_SIZE", type=int, default=1024)
cfg_parser.add_argument("--LEARNING_RATE", type=float, default=0.1)
cfg_parser.add_argument("--MAX_ITERATIONS", type=int, default=10000)
cfg_parser.add_argument("--BATCH_SIZE", type=int, default=64)
cfg_parser.add_argument("--PARAM_FN", type=str, default="clip")
cfg_parser.add_argument("--USE_AMP", type=bool, default=True)

cfg_parser.add_argument("--LOSS_SCORE_COEFF", type=float, default=0)
cfg_parser.add_argument("--LOSS_TV_COEFF", type=float, default=0)
cfg_parser.add_argument("--LOSS_BN_COEFF", type=float, default=1)
cfg_parser.add_argument("--LOSS_BN_LAYERS_MODE", type=str, default="all")
cfg_parser.add_argument("--LOSS_BN_LAYERS_N", type=int, default=10)


cfg_parser.add_argument("--AUG_FLIP", type=bool, default=False)
cfg_parser.add_argument("--AUG_ROTATE_DEGREES", type=int, default=30)

cfg_parser.add_argument("--CLASSES", default=[999])
cfg_parser.add_argument(
    "--NETWORKS",
    type=str,
    default=[
        # "tf_efficientnet_b0_ap",
        "resnet18",
        # "resnetblur18",
        "resnet26",
        "resnet34",
        "resnet50",
        # "resnetblur50",
        # "densenet121",
        # "densenet161",
        # "densenet169",
        # "densenet201",
        # "xception",
    ],
)

cfg_args = cfg_parser.parse_args()

#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops, time
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
    AuxLossWrapper,
)

from augmentations import RandomCircularShift, RandomDownResolution
from utils import imagenet_class_name_of, get_timm_network

with wandb.init(project="vis-denemeler", config=cfg_args) as run:
    cfg = wandb.config

    device = "cuda:0"
    torch.backends.cudnn.benchmark = True

    networks = [
        get_timm_network(network_name, device=device)[0]
        for network_name in cfg.NETWORKS
    ]
    networks = [
        AuxLossWrapper(
            net,
            mode=cfg.LOSS_BN_LAYERS_MODE,
            n=cfg.LOSS_BN_LAYERS_N,
        )
        for net in networks
    ]

    aug_fn = torch.nn.Sequential(
        RandomCircularShift(),
        kornia.augmentation.RandomResizedCrop(
            size=[224, 224],
            # scale=(0.8 * input_size[0] / cfg.IMG_SIZE, 1),
            # ratio=(0.8, 1.2),  # aspect ratio
            scale=(0.9 * 224 / cfg.IMG_SIZE, 1.1 * 224 / cfg.IMG_SIZE),
            ratio=(0.9, 1.1),
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.9,
        patience=100,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)
    start_time = time.time()
    for n in tqdm.tqdm(range(cfg.MAX_ITERATIONS + 1)):
        with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
            optimizer.zero_grad(set_to_none=True)
            aux_losses = []
            score_losses = []
            prob_losses = []
            tv_losses = []
            losses = []
            for idx in range(in_layer.n_classes):
                net = random.choice(networks)
                for net in networks:
                    random_idx = random.choices(
                        range(in_layer.n_classes), k=cfg.BATCH_SIZE
                    )
                    imgs, classes = in_layer(random_idx)
                    logits, aux_loss = net(imgs)

                    score_loss = softmax_loss_fn(logits, classes) * cfg.LOSS_SCORE_COEFF
                    # score_loss = score_loss_fn(logits, classes) * cfg.LOSS_SCORE_COEFF
                    tv_loss = tv_loss_fn(imgs) * cfg.LOSS_TV_COEFF
                    aux_loss = aux_loss * cfg.LOSS_BN_COEFF

                    loss = (score_loss + tv_loss + aux_loss) / len(networks)
                    loss.backward()

                    score_losses.append(score_loss.item())
                    tv_losses.append(tv_loss.item())
                    aux_losses.append(aux_loss.item())
                    losses.append(loss.item())

            optimizer.step()
            scheduler.step(np.mean(losses))

        log_dict = {
            "losses/score_loss": np.mean(score_losses),
            "losses/tv_loss": np.mean(tv_losses),
            "losses/aux_loss": np.mean(aux_losses),
            "losses/total_loss": np.mean(losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        if optimizer.param_groups[0]["lr"] < 1e-4:
            break

        # Log heavy things
        if n % 100 == 0 and n >= 100:
            with torch.no_grad():
                # random_idx = random.choices(range(in_layer.n_classes), k=1024)
                # imgs, classes = in_layer(random_idx)
                # logits, aux_loss = net(imgs)
                # probs = torch.nn.functional.softmax(logits)

                # Log timer things
                now = time.time()
                elapsed = now - start_time
                rate = (n + 1) / elapsed
                remaining = cfg.MAX_ITERATIONS / rate - elapsed

                log_dict.update(
                    {
                        "timer/remaining_mins": remaining / 60,
                        "timer/rate": rate,
                        "timer/elapsed_mins": elapsed / 60,
                    }
                )

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

    # Save model
    # os.makedirs("saved_models", exist_ok=True)
    # model_save_path = os.path.join("saved_models", "in_layer.pt")
    # torch.save(in_layer, model_save_path)

    # inlayer_artifact = wandb.Artifact(
    #     name="my_inlayer_artifact",
    #     type="input_layer_type",
    #     description="SOME optional description",
    #     metadata=dict(cfg),
    # )
    # inlayer_artifact.add_file(model_save_path)

    # wandb.log_artifact(inlayer_artifact)
