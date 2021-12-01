import torch
import timm

model = timm.create_model("resnet50", pretrained=True)
