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




class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return int(1000)

    def __getitem__(self, index):
        return "THIS IS WHAT YOU GET EVERY TIME"




if __name__ == "__main__":
    pass