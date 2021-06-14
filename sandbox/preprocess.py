import timm

timm.list_models("*mix*")


import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model_names = timm.list_models(pretrained=True)

network_names = [
    "resnet18",
    "resnet50",  # good
    "densenet121",  # good
    "efficientnet_b4",  # bad
    "efficientnet_b1",  # bad
    "efficientnet_b1_pruned",  # bad
    "inception_v4",  # good
    "adv_inception_v3",  # good
    "vit_base_patch16_224",  # meh
]

for name in model_names:
    model = timm.create_model(name)
    # print(f'Name: {name}')
    config = resolve_data_config({}, model=model)
    print(f"Config: {config}")
    transform = create_transform(**config)
    # print(f'Transform: {transform}')
