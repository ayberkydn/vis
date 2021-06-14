import timm

timm.list_models("*mix*")


import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# model_names = timm.list_models(pretrained=True)

network_names = [
    "cspdarknet53",
]

for name in network_names:
    model = timm.create_model(name)
    # print(f'Name: {name}')
    config = resolve_data_config({}, model=model)
    print(f"Config: {config}")
    transform = create_transform(**config)
    # print(f'Transform: {transform}')
