from kornia.augmentation.augmentation import RandomResizedCrop
import torch
import kornia
from kornia import tensor_to_image as t2i
import PIL
import torchvision
import matplotlib.pyplot as plt


aug = kornia.augmentation.RandomPerspective(distortion_scale=1, p=1)
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
