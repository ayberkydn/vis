#%%
import argparse

cfg_parser = argparse.ArgumentParser()

cfg_parser.add_argument("--IMG_SIZE", type=int, default=512)
cfg_parser.add_argument("--BATCH_SIZE", type=int, default=8)
cfg_parser.add_argument("--LEARNING_RATE", type=float, default=0.01)
cfg_parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
cfg_parser.add_argument("--MAX_ITERATIONS", type=int, default=10000)
cfg_parser.add_argument("--PARAM_FN", type=str, default="clip")
cfg_parser.add_argument("--USE_AMP", type=bool, default=False)
cfg_parser.add_argument("--LOG_ONLY_LAST_IMG", type=bool, default=False)

cfg_parser.add_argument("--LOSS_SCORE_COEFF", type=float, default=1)
cfg_parser.add_argument("--LOSS_TV_COEFF", type=float, default=300.0)
cfg_parser.add_argument("--LOSS_BN_COEFF", type=float, default=0)
cfg_parser.add_argument("--LOSS_BN_LAYERS_MODE", type=str, default="all")
cfg_parser.add_argument("--LOSS_BN_LAYERS_N", type=int, default=10)

cfg_parser.add_argument("--AUG_HFLIP", type=bool, default=True)
cfg_parser.add_argument("--AUG_VFLIP", type=bool, default=False)
cfg_parser.add_argument("--AUG_ROTATE_DEGREES", type=int, default=15)
cfg_parser.add_argument("--AUG_CROP_RATIO", type=float, default=2)
cfg_parser.add_argument("--AUG_RESAMPLE", type=str, default="bicubic")

cfg_parser.add_argument("--CLASSES", default=[107])
# cfg_parser.add_argument("--CLASSES", default=[107, 409, 530, 531, 826, 892])
cfg_parser.add_argument(
    "--NETWORKS",
    type=str,
    default=[
        # "tf_efficientnet_b0_ap",
        # "ens_adv_inception_resnet_v2",
        # "swin_tiny_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "adv_inception_v3",
        # "vgg11",
        # "vgg11_bn",
        # "resnet18",
        # "resnet26",
        # "resnet34",
        # "resnet50",
        # "resnetblur50",
        # "resnext50_32x4d",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        # "xception",
        # "visformer_small",
        # "convit_tiny",
        # "convit_base",
        # "mobilenetv3_large_100",
        # "cifar10_resnet20"
    ],
)

cfg_args = cfg_parser.parse_args()

#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops, time, piqa, piq
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from input_img_layer import InputImageLayer
from losses import (
    inception_loss,
    tv_loss_fn,
    softmax_loss_fn,
    score_loss_fn,
)
from hook_wrappers import (
    AuxLossWrapper,
)

from augmentations import (
    RandomCircularShift,
    RandomDownResolution,
    RandomWeightedResizedCrop,
)
from utils import (
    get_images_log_dict,
    imagenet_class_name_of,
    cifar_class_name_of,
    get_network,
)

with wandb.init(project="vis-denemeler", config=cfg_args) as run:
    cfg = wandb.config

    device = "cuda"
    # device = "cpu"
    torch.backends.cudnn.benchmark = True

    networks = [
        get_network(network_name, device=device)[0] for network_name in cfg.NETWORKS
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
        RandomWeightedResizedCrop(
            size=[224, 224],
            scale=(1 / cfg.AUG_CROP_RATIO, 1),
            ratio=(0.9, 1.1),
            same_on_batch=False,
            resample=cfg.AUG_RESAMPLE,
        ),
        kornia.augmentation.RandomRotation(
            degrees=cfg.AUG_ROTATE_DEGREES,
            same_on_batch=False,
            p=1,
        ),
        kornia.augmentation.RandomHorizontalFlip(p=0.5 * cfg.AUG_HFLIP),
        kornia.augmentation.RandomVerticalFlip(p=0.5 * cfg.AUG_VFLIP),
        # kornia.augmentation.RandomGaussianBlur(
        #     kernel_size=[5, 5],
        #     sigma=[1, 1],
        #     same_on_batch=False,
        #     p=0.5,
        # ),
    )

    in_layer = InputImageLayer(
        img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
        classes=cfg.CLASSES,
        param_fn=cfg.PARAM_FN,
        aug_fn=aug_fn,
    ).to(device)

    optimizer = torch.optim.Adam(
        in_layer.parameters(),
        lr=cfg.LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.9,
        patience=100,
    )

    start_time = time.time()
    loss_history = []
    for n in tqdm.tqdm(range(cfg.MAX_ITERATIONS + 1)):
        if optimizer.param_groups[0]["lr"] < 1e-4:
            break
        optimizer.zero_grad(set_to_none=True)
        aux_losses = []
        score_losses = []
        prob_losses = []
        tv_losses = []
        losses = []
        for idx in range(in_layer.n_classes):
            for net in networks:
                batch_size = cfg.BATCH_SIZE or (cfg.AUG_CROP_RATIO ** 2) * 2
                random_idx = random.choices(range(in_layer.n_classes), k=batch_size)
                imgs, classes = in_layer(random_idx)

                logits, aux_loss = net(imgs)

                score_loss = score_loss_fn(logits, classes) * cfg.LOSS_SCORE_COEFF

                classes2 = torch.ones_like(classes) * 409
                score_loss = (
                    score_loss + score_loss_fn(logits, classes2) * cfg.LOSS_SCORE_COEFF
                )

                tv_loss = tv_loss_fn(imgs)
                aux_loss = aux_loss * cfg.LOSS_BN_COEFF

                loss = (score_loss + tv_loss + aux_loss) / len(networks)
                loss.backward()

                score_losses.append(score_loss.item())
                tv_losses.append(tv_loss.item())
                aux_losses.append(aux_loss.item())
                losses.append(loss.item())

        optimizer.step()

        loss_history.append(np.mean(losses))
        if len(loss_history) > 100:
            loss_history = loss_history[-100:]

        scheduler.step(np.mean(loss_history))

        log_dict = {
            "losses/score_loss": np.mean(score_losses),
            "losses/tv_loss": np.mean(tv_losses),
            "losses/aux_loss": np.mean(aux_losses),
            "losses/total_loss": np.mean(losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        # Log heavy things
        with torch.no_grad():
            if n % 100 == 0 and n >= 100:
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

                log_dict.update(
                    {
                        "tensor_stats/max_value": tensor.max(),
                        "tensor_stats/grad_max_abs": tensor.grad.abs().max(),
                        "tensor_stats/grad_mean_abs": tensor.grad.abs().mean(),
                        "tensor_stats/grad_std": tensor.grad.std(),
                        "tensor_stats/min_value": tensor.min(),
                    },
                )

            if n in [10, 100, 250, 500] or (n % 1000 == 0):
                log_dict.update(get_images_log_dict(in_layer))

        wandb.log(
            log_dict,
            step=n,
            commit=True,
        )

    wandb.log(
        get_images_log_dict(in_layer),
        step=n + 1,
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
