import torch
import kornia
import PIL
import torchvision


aug = kornia.augmentation.RandomPerspective(distortion_scale=0.5, p=1)

to_tensor = torchvision.transforms.ToTensor()

img = to_tensor(PIL.Image.open("img.png")).squeeze(0)
print(img.shape)
