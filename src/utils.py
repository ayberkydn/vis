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
from torchvision.transforms.functional import center_crop


def imagenet_class_name_of(n: int) -> str:
    with open("imagenet1000_clsidx_to_labels.txt", "r") as labels_file:
        labels = labels_file.read().splitlines()
    return labels[n]


def get_timm_network(name, device):
    """
    - Creates networks, moves to cuda, disables parameter grads and set eval mode.
    - Merges the necessary normalization preprocessing with the networks
    - Returns the list of networks
    """
    network = timm.create_model(name, pretrained=True).eval().to(device)
    config = resolve_data_config({}, model=network)

    for param in network.parameters():
        param.requires_grad = False

    input_size = config["input_size"][-2:]
    mean = config["mean"]
    std = config["std"]

    preprocess = torch.nn.Sequential(
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.Normalize(mean, std),
    )
    pre_and_net = torch.nn.Sequential(
        preprocess,
        network,
    )
    return pre_and_net


class RandomCircularShift(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        H = x.shape[-2]
        W = x.shape[-1]

        shift_H = random.randint(0, H - 1)
        shift_W = random.randint(0, W - 1)
        shifted_img = torch.roll(x, shifts=(shift_H, shift_W), dims=(-2, -1))
        return shifted_img


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return int(1000)

    def __getitem__(self, index):
        return "THIS IS WHAT YOU GET EVERY TIME"


class BNStatsModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bn_stats = []
        self.submodule_names = [name for name, layer in self.model.named_modules()]

        def create_hook(name):
            def hook(module, inputs, outputs):
                assert len(inputs) == 1
                self.bn_stats.append(
                    {
                        "name": name,
                        "module": module,
                        "inputs_mean": inputs[0].mean(dim=[0, -1, -2]),
                        "inputs_var": inputs[0].var(dim=[0, -1, -2]),
                    }
                )

            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.register_forward_hook(create_hook(name))

    def forward(self, x):
        self.bn_stats = []
        outputs, activations = self.model(x), self.bn_stats
        return outputs, activations


if __name__ == "__main__":
    net = timm.create_model("vgg11_bn")
    net = BNStatsModelWrapper(net)
    input = torch.rand(1, 3, 224, 224)
    out, activations = net(input)

    bn_activations = [
        act for act in activations if isinstance(act["module"], torch.nn.BatchNorm2d)
    ]
