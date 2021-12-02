import copy
from math import inf

import einops
import kornia
import matplotlib.pyplot as plt
import timm
from timm.data import resolve_data_config
from timm.models.layers import activations
import torch
import torchvision
import logging
import random
from torch.utils.data import Dataset, DataLoader


def imagenet_class_name_of(n: int) -> str:
    with open("imagenet1000_clsidx_to_labels.txt", "r") as labels_file:
        labels = labels_file.read().splitlines()
    return labels[n]


def get_timm_network(name):
    """
    - Creates networks, moves to cuda, disables parameter grads and set eval mode.
    - Merges the necessary normalization preprocessing with the networks
    - Returns the list of networks
    """
    network = timm.create_model(name, pretrained=True).eval().cuda()
    config = resolve_data_config({}, model=network)

    for param in network.parameters():
        param.requires_grad = False

    input_size = config["input_size"][-2:]
    mean = config["mean"]
    std = config["std"]

    logging.info(f"Network name: {name}")
    logging.info(f"Mean: {mean}")
    logging.info(f"Std: {std}")
    logging.info(f"Input size: {input_size}")

    preprocess = torch.nn.Sequential(
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.Normalize(mean, std),
    )

    pre_and_net = torch.nn.Sequential(
        preprocess,
        network,
    )

    return VerboseModelWrapper(pre_and_net)


def score_maximizer_loss(x, target_class):
    return -x[:, target_class].mean()


def probability_maximizer_loss(x, target_class):
    b = x.shape[0]
    dev = x.device

    target = torch.ones(size=[b], dtype=torch.long, device=x.device) * target_class
    return torch.nn.CrossEntropyLoss()(x, target)


def add_noise(network, factor):
    new_network = copy.deepcopy(network)
    with torch.no_grad():
        for param in new_network.parameters():
            param += torch.randn_like(param) * param.std() * factor
    return new_network


class RandomCircularShift(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        H = x.shape[-2]
        W = x.shape[-1]

        img_width_expand = torch.cat([x] * 2, dim=-1)
        img_torus = torch.cat([img_width_expand] * 2, dim=-2)

        return kornia.augmentation.RandomCrop(size=[H, W])(img_torus)


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return int(1000)

    def __getitem__(self, index):
        return "THIS IS WHAT YOU GET EVERY TIME"


class InputImageLayer(torch.nn.Module):
    def __init__(self, img_shape, img_class, param_fn, aug_fn=None):

        super().__init__()
        self.aug_fn = aug_fn

        if param_fn == "sigmoid":
            self.input_tensor = torch.nn.Parameter(torch.randn(*img_shape) * 0.0)
            self.param_fn = torch.nn.Sigmoid()

        elif param_fn == "clip":
            self.input_tensor = torch.nn.Parameter(torch.randn(*img_shape) * 0.0 + 0.5)
            self.param_fn = lambda x: torch.clip(x, 0, 1)

        elif param_fn == "scale":
            self.input_tensor = torch.nn.Parameter(torch.rand(*img_shape))

            def scale(x):
                x = x - x.min()
                x = x / x.max()
                return x

            self.param_fn = scale

        elif param_fn == "sin":
            self.input_tensor = torch.nn.Parameter(torch.randn(*img_shape) * 0.0)
            self.param_fn = lambda x: torch.sin(x) / 2 + 0.5

        else:
            raise Exception("Invalid param_fn")

    def forward(self, batch_size, augment=True):
        img = self.param_fn(self.input_tensor)
        imgs = einops.repeat(img, "c h w -> b c h w", b=batch_size)
        if self.aug_fn:
            imgs = self.aug_fn(imgs)

        return imgs

    def get_image(self, uint=False):
        with torch.no_grad():
            img_np = kornia.tensor_to_image(self.param_fn(self.input_tensor))
            if uint == True:
                scaled_img_np = img_np * 255
                return scaled_img_np.astype(int)
            else:
                return img_np


class VerboseModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = []
        self.submodule_names = [name for name, layer in self.model.named_modules()]

        def create_hook(name):
            def hook(module, inputs, output):
                self.activations.append(
                    {
                        "name": name,
                        "module": module,
                        "output": output.mean(dim=0),
                    }
                )

            return hook

        for name, layer in self.model.named_modules():
            # if isinstance(layer, torch.nn.Conv2d):
            layer.register_forward_hook(create_hook(name))

    def forward(self, x):
        self.activations = []
        outputs, activations = self.model(x), self.activations
        return outputs, activations

    def get_submodule_names(self):
        return self.submodule_names


net = timm.create_model("vgg11_bn")
net = VerboseModelWrapper(net)
input = torch.rand(1, 3, 224, 224)
out, activations = net(input)

bn_activations = [
    act for act in activations if isinstance(act["module"], torch.nn.BatchNorm2d)
]
