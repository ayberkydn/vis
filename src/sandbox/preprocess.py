import timm

print(timm.list_models("*dense*", pretrained=True))
