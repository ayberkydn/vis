import torch, einops, kornia


class InputImageLayer(torch.nn.Module):
    def __init__(self, img_shape, classes, param_fn, aug_fn=None, init_img=None):

        super().__init__()
        if aug_fn:
            self.aug_fn = aug_fn
        else:
            self.aug_fn = torch.nn.Identity()

        self.n_classes = len(classes)
        self.classes = torch.nn.Parameter(torch.tensor(classes), requires_grad=False)

        self.input_tensor = torch.nn.Parameter(
            torch.randn(self.n_classes, *img_shape) * 0.001
        )
        self.param_fn = lambda x: torch.clip(x, -0.5, 0.5) + 0.5
        self.inv_param_fn = lambda x: x - 0.5

    def forward(self, indices, augment=True):
        device = self.input_tensor.device
        indices = torch.tensor(indices, dtype=torch.long, device=device)

        tensors = self.input_tensor[indices]
        classes = self.classes[indices]
        imgs = self.param_fn(tensors)

        if augment == True:
            imgs = self.aug_fn(imgs)

        return imgs, classes

    def get_images(self, uint=False):
        images = []
        with torch.no_grad():
            for n in range(self.n_classes):
                img_np = kornia.tensor_to_image(self.param_fn(self.input_tensor[n]))
                if uint == True:
                    scaled_img_np = img_np * 255
                    img_ = scaled_img_np.astype(int)
                else:
                    img_ = img_np
                images.append(img_)
            return images
