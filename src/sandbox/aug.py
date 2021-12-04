from kornia.augmentation.augmentation import RandomResizedCrop
import torch
import kornia
from kornia import tensor_to_image as t2i
import PIL
import torchvision
import matplotlib.pyplot as plt


class RollWrapper(torch.nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.aug = aug

    def forward(self, x):

        H = x.shape[-2]
        W = x.shape[-1]

        repeated_img = x.repeat(1, 1, 3, 3)
        aug_img = self.aug(repeated_img)
        return kornia.augmentation.CenterCrop(size=(H, W))(aug_img)


aug = RollWrapper(kornia.augmentation.RandomRotation(degrees=180, p=1))
to_tensor = torchvision.transforms.ToTensor()

img = to_tensor(PIL.Image.open("../../sample.jpeg")).unsqueeze(0)
print(img.shape)

aug_img = aug(img)
plt.imshow(t2i(aug_img))
plt.show()

# to_tensor = torchvision.transforms.ToTensor()

# img = to_tensor(PIL.Image.open("sample.jpeg")).unsqueeze(0)
# print(img.shape)

# aug_img = aug(img)
# plt.imshow(t2i(aug_img))
# plt.show()
