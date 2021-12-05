import torch, einops, kornia


class InputImageLayer(torch.nn.Module):
    def __init__(self, img_shape, classes, param_fn, aug_fn=None):

        super().__init__()
        self.aug_fn = aug_fn
        self.num_classes = len(classes)
        self.classes = torch.nn.Parameter(torch.tensor(classes), requires_grad=False)

        if param_fn == "sigmoid":
            self.input_tensor = torch.nn.Parameter(
                torch.randn(self.num_classes, *img_shape) * 0.0
            )
            self.param_fn = torch.nn.Sigmoid()

        elif param_fn == "clip":
            self.input_tensor = torch.nn.Parameter(
                torch.randn(self.num_classes, *img_shape) * 0.0 + 0.5
            )
            self.param_fn = lambda x: torch.clip(x, 0, 1)

        # elif param_fn == "scale":
        #     self.input_tensor = torch.nn.Parameter(torch.rand(self.num_classes, *img_shape))

        #     def scale(x):
        #         x = x - x.min()
        #         x = x / x.max()
        #         return x

        #     self.param_fn = scale

        elif param_fn == "sin":
            self.input_tensor = torch.nn.Parameter(
                torch.randn(self.num_classes, *img_shape) * 0.0
            )
            self.param_fn = lambda x: torch.sin(x) / 2 + 0.5

        else:
            raise Exception("Invalid param_fn")

    def forward(self, batch_size, index=None, augment=True):
        if index == None:
            indices = torch.randint(
                0, self.num_classes, [batch_size], device=self.input_tensor.device
            )
        else:
            assert isinstance(index, int)
            indices = (
                torch.ones(
                    batch_size, dtype=torch.long, device=self.input_tensor.device
                )
                * index
            )

        tensors = self.input_tensor[indices]
        classes = self.classes[indices]
        imgs = self.param_fn(tensors)

        # imgs = einops.repeat(img, "c h w -> b c h w", b=batch_size)
        if self.aug_fn:
            imgs = self.aug_fn(imgs)

        return imgs, classes

    def get_images(self, uint=False):
        images = []
        with torch.no_grad():
            for n in range(self.num_classes):
                img_np = kornia.tensor_to_image(self.param_fn(self.input_tensor[n]))
                if uint == True:
                    scaled_img_np = img_np * 255
                    img_ = scaled_img_np.astype(int)
                else:
                    img_ = img_np
                images.append(img_)
            return images
