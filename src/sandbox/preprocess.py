import timm

print(timm.list_models("resnet*", pretrained=True))
