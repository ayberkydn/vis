from kornia.augmentation.augmentation import RandomResizedCrop
from numpy import log2
import torch
import kornia
import random
from kornia import tensor_to_image as t2i
import PIL
import torchvision
import matplotlib.pyplot as plt


dres = RandomDownResolution()


to_tensor = torchvision.transforms.ToTensor()
img = to_tensor(PIL.Image.open("sample.jpeg")).unsqueeze(0)

pyramid = dres(img)

plt.imshow(t2i(img))
plt.show()
