import torch, random


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
