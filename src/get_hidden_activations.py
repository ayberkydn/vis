import torch


import os, sys
import utils

print(os.getcwd())
print(sys.path)

model = utils.get_timm_network("vgg11_bn")
input = torch.rand(1, 3, 224, 224).cuda()

logits, activations = model(input)

print([activation["name"] for activation in activations if "bn" in activation["name"]])
