import PIL
from kornia.geometry.transform.pyramid import pyrup
from kornia.utils.image import image_list_to_tensor
import torch, random, kornia, numpy
import torchvision
import matplotlib.pyplot as plt


class RandomCircularShift(torch.nn.Module):
    def __init__(self, ratio=1.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        H = x.shape[-2]
        W = x.shape[-1]

        h_max = int(H * self.ratio)
        w_max = int(W * self.ratio)

        shift_H = random.randint(-h_max, h_max - 1)
        shift_W = random.randint(-w_max, w_max - 1)
        shifted_img = torch.roll(x, shifts=(shift_H, shift_W), dims=(-2, -1))
        return shifted_img


class RandomDownResolution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        max_level = int(numpy.log2(min(H, W))) - 1
        random_level = random.choice(range(max_level))
        # print(random_level)

        pyramid = kornia.geometry.pyramid.build_pyramid(
            x,
            max_level=max_level,
        )
        random_downscaled = pyramid[random_level]
        # random_downscaled = kornia.filters.blur_pool2d(
        #     input=x,
        #     kernel_size=2 ** random_level,
        #     stride=2 ** random_level,
        # )
        upscaled_back = torch.nn.functional.interpolate(
            random_downscaled,
            scale_factor=2 ** random_level,
            mode="nearest",
        )
        return upscaled_back


if __name__ == "__main__":
    aug = RandomCircularShift(ratio=0.25)
    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(PIL.Image.open("sample.jpeg").resize([256, 256])).unsqueeze(0)
    dres_img = aug(img)
    print(dres_img.shape)

    plt.imshow(kornia.utils.tensor_to_image(aug(img)))
    plt.show()
