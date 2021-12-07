import timm
import torch

print(timm.list_models("*swin*", pretrained=True))

model = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
