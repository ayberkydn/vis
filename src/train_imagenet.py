import torch, torchvision
import matplotlib.pyplot as plt
import kornia
from kornia import tensor_to_image as t2i
import tqdm
import random

net = torchvision.models.resnet50(pretrained=False)
dataset = torchvision.datasets.ImageFolder()
