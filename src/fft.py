import torch
import PIL
import torchvision.transforms.functional as TF


img_pil = PIL.Image.open("img.png")

img = TF.to_tensor(img_pil)

img = torch.ones_like(img)


norm = "forward"
img_freq = torch.fft.rfft2(img, norm=norm)

print(f"({img_freq.real.min()}----{img_freq.real.max()})")
print(f"({img_freq.imag.min()}----{img_freq.imag.max()})")

img_reconst = torch.fft.irfft2(img_freq, norm=norm)

img_re_pil = TF.to_pil_image(img_reconst)

img_re_pil
