import timm
import torch
import einops
import matplotlib.pyplot as plt
import copy
import kornia


def imagenet_class_name_of(n: int) -> str:
    with open("imagenet1000_clsidx_to_labels.txt", "r") as labels_file:
        labels = labels_file.read().splitlines()
    return labels[n]


def get_timm_networks(network_name_list):
    networks = []
    for name in network_name_list:
        network = timm.create_model(name, pretrained=True).eval().cuda()
        for param in network.parameters():
            param.requires_grad = False

        networks.append(network)

    return networks


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


def validate(networks, img_layer, aug_fn, normalize):
    with torch.no_grad():
        min_prob = 1
        for n in range(32):
            imgs = img_layer
            input_imgs = input_img_layer(32)
            aug_imgs = aug_fn(input_imgs)
            out = net(normalize(aug_imgs))
            probs = torch.softmax(out, dim=-1)[:, TARGET_CLASS]
            min_prob = probs.min() if probs.min() < min_prob else min_prob
        print(min_prob)


class RandomCircularShift(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        H = x.shape[-2]
        W = x.shape[-1]

        img_width_expand = torch.cat([x] * 2, dim=-1)
        img_torus = torch.cat([img_width_expand] * 2, dim=-2)

        return kornia.augmentation.RandomCrop(size=[H, W])(img_torus)


class InputImageLayer(torch.nn.Module):
    def __init__(self, shape, param_fn=None):

        super().__init__()

        if param_fn == None:
            self.param_fn = torch.Identity()

        else:
            self.param_fn = param_fn

        self.input_tensor = torch.nn.Parameter(
            torch.zeros(shape),
            requires_grad=True,
        )
        self.param_fn = param_fn

    def forward(self, batch_size):
        batch = einops.repeat(
            self.input_tensor,
            "c h w -> b c h w",
            b=batch_size,
        )

        return self.param_fn(batch)
