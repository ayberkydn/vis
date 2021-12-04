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

    return BNStatsModelWrapper(pre_and_net)


def score_maximizer_loss(logits, target_classes):
    return -logits[:, target_classes].mean()


def probability_maximizer_loss(logits, target_classes):
    b = logits.shape[0]
    dev = logits.device

    target = (
        torch.ones(size=[b], dtype=torch.long, device=logits.device) * target_classes
    )
    return torch.nn.CrossEntropyLoss()(logits, target)


def bn_stats_loss(activations):
    losses = []
    for act_n in range(len(activations)):
        act_mean = activations[act_n]["inputs_mean"]
        act_var = activations[act_n]["inputs_var"] + 1e-8
        running_mean = activations[act_n]["module"].running_mean
        running_var = activations[act_n]["module"].running_var

        loss_n = torch.log(torch.sqrt(act_var) / torch.sqrt(running_var)) - 0.5 * (
            1 - ((running_var + torch.square(act_mean - running_mean)) / act_var)
        )
        losses.append(loss_n.mean())

    loss = torch.mean(torch.stack(losses))
    return loss


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

        shift_H = random.randint(0, H - 1)
        shift_W = random.randint(0, W - 1)
        shifted_img = torch.roll(x, shifts=(shift_H, shift_W), dims=(-2, -1))
        return shifted_img


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
