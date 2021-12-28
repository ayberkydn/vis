import timm
import torch
from src.input_img_layer import InputImageLayer

# in_layer = InputImageLayer([3, 50, 50], [2], param_fn="clip")
# imgs, classes = in_layer([0])

timm.list_models("*resnex*", pretrained=True)
