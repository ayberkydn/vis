#%%
import torch, torchvision

img = torch.arange(9).reshape(1, 1, 3, 3)
print(img)
#%%


class Torus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        img_width_expand = torch.cat([x, x], dim=-1)
        img_torus = torch.cat([img_width_expand, img_width_expand], dim=-2)
        return img_torus


# %%
