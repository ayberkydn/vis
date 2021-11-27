#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops
import matplotlib.pyplot as plt
from kornia import tensor_to_image as t2i

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

wandb.login()
config = {
    "IMG_SIZE": 512,
    "NET_INPUT_SIZE": 224,
    "NETWORKS": [
        # "resnet18",
        "mixnet_s",
    ],
    "LEARNING_RATE": 0.025,
    "ITERATIONS": 20000,
    "BATCH_SIZE": 16,
    "CLASSES": [
        318,
        512,
        4,
    ],
}

with wandb.init(project="vis", config=config):
    cfg = wandb.config

    input_img_layer = InputImageLayer(
        img_classes=cfg.CLASSES,
        img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
        param_fn=torch.nn.Sequential(
            torch.nn.Sigmoid(),
        ),
    ).cuda()

    aug_fn = torch.nn.Sequential(
        RandomCircularShift(),
        kornia.augmentation.RandomRotation(
            degrees=30,
            same_on_batch=False,
            p=1,
        ),
        kornia.augmentation.RandomResizedCrop(
            size=(cfg.NET_INPUT_SIZE, cfg.NET_INPUT_SIZE),
            scale=(0.1, 1),
            ratio=(0.5, 2),  # aspect ratio
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
    mixup = kornia.augmentation.RandomMixUp()

    networks = get_timm_networks(cfg.NETWORKS)
    optimizer = torch.optim.Adam(input_img_layer.parameters(), lr=cfg.LEARNING_RATE)

    #%% train
    wandb.watch(input_img_layer, log="all", log_freq=10)
    for n in tqdm.tqdm(range(cfg.ITERATIONS)):
        net = random.choice(networks)
        input_imgs, input_labels = input_img_layer(cfg.BATCH_SIZE)
        # input_imgs = einops.repeat(input_img, "c h w -> b c h w", b=cfg.BATCH_SIZE)
        aug_imgs = aug_fn(input_imgs)
        out = net(aug_imgs)

        # loss=5

        loss.backward()
        if n % 50 == 0:
            wandb.log(
                {
                    "loss": loss,
                    "tensor_max_value": input_img_layer.input_tensor.max(),
                    "tensor_min_value": input_img_layer.input_tensor.min(),
                    "image_max_value": input_img_layer().max(),
                    "image_min_value": input_img_layer().min(),
                },
                step=n,
            )
        if n % 500 == 0:
            wandb.log(
                {
                    "image": wandb.Image(input_img_layer()),
                },
                step=n,
            )

        optimizer.step()
        optimizer.zero_grad()
