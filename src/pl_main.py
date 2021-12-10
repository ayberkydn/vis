#%%
import torch, torchvision, kornia, tqdm, random, wandb, einops
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from kornia import tensor_to_image as t2i
import os, sys
from dataclasses import dataclass

from utils import (
    DummyDataset,
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


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        aug_fn = torch.nn.Sequential(
            RandomCircularShift(),
            # kornia.augmentation.RandomRotation(
            #     degrees=180,
            #     same_on_batch=False,
            #     p=1,
            # ),
            kornia.augmentation.RandomResizedCrop(
                size=(cfg.NET_INPUT_SIZE, cfg.NET_INPUT_SIZE),
                scale=(
                    cfg.NET_INPUT_SIZE / cfg.IMG_SIZE,
                    cfg.IMG_SIZE / cfg.NET_INPUT_SIZE,
                ),
                ratio=(1, 1),  # aspect ratio
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

        self.networks = get_timm_networks(cfg.NETWORKS)
        self.input_imgs_layer = InputImageLayer(
            img_class=[cfg.CLASS],
            img_shape=[3, cfg.IMG_SIZE, cfg.IMG_SIZE],
            param_fn=cfg.PARAM_FN,
            aug_fn=aug_fn,
        )

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        net = random.choice(self.networks)
        imgs, labels = self.input_imgs_layer(cfg.BATCH_SIZE)
        logits = net(imgs)
        loss = score_maximizer_loss(logits, cfg.CLASS)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.input_imgs_layer.parameters(),
            lr=cfg.LEARNING_RATE,
        )
        return optimizer


mymodule = LitAutoEncoder(cfg)
import pl_bolts

# dataset = DummyDataset()
# loader = DataLoader(dataset)

loader = DataLoader(pl_bolts.datasets.RandomDataset(size=5))
trainer = pl.Trainer(devices=2, accelerator="gpu", strategy="ddp")
trainer.fit(mymodule, loader)


# with wandb.init(project="vis", config=config):
#     cfg = wandb.config

#     wandb.config.update({"aug_fn": aug_fn})

#     #%% train
#     wandb.watch(input_imgs_layer, log="all", log_freq=cfg.LOG_FREQ)
#     for n in tqdm.tqdm(range(cfg.ITERATIONS)):
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if n % cfg.LOG_FREQ == 0:
#             wandb.log(
#                 {
#                     "loss": loss,
#                     "tensor_max_value": input_imgs_layer.input_tensor.max(),
#                     "tensor_min_value": input_imgs_layer.input_tensor.min(),
#                     "image_max_value": input_imgs_layer()[0].max(),
#                     "image_min_value": input_imgs_layer()[0].min(),
#                     "tensor_grad_max_abs_value": input_imgs_layer.input_tensor.grad.abs().max(),
#                 },
#                 step=n,
#             )
#         if n % cfg.LOG_FREQ == 0:
#             wandb.log(
#                 {
#                     "image": wandb.Image(input_imgs_layer()[0]),
#                 },
#                 step=n,
#             )
