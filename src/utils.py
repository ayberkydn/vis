import timm
import torch
import einops


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
    pass


def probability_maximizer_loss(x, target_class):
    b = x.shape[0]
    dev = x.device

    target = torch.ones(size=[b], dtype=torch.long, device=x.device)
    return torch.nn.CrossEntropyLoss()(x, target)


class InputImageLayer(torch.nn.Module):
    def __init__(self, shape, param_fn=None):

        super().__init__()

        if param_fn == None:
            self.param_fn = torch.Identity()

        else:
            self.param_fn = param_fn

        self.input_tensor = torch.nn.Parameter(
            torch.randn(shape),
            requires_grad=True,
        )
        self.param_fn = param_fn

    def forward(self, batch_size):
        batch = einops.repeat(
            self.input_tensor,
            "c h w -> b c h w",
            b=batch_size,
        )

        return self.param_fn(self.input_tensor)
