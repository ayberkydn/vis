import torch
import tqdm


def nonzero_grads(input_img_layer, aug_fn):
    input_imgs = input_img_layer(1)
    input_imgs.retain_grad()
    aug_imgs = aug_fn(input_imgs)
    loss = aug_imgs.mean()
    loss.backward()
    nonzero_grads = input_imgs.grad != 0
    nonzero_grads = nonzero_grads.double()
    return nonzero_grads


def pixel_sample_ratio_map(input_img_layer, aug_fn, sample_size=100, times=10):
    with torch.no_grad():
        sample_ratio = torch.zeros_like(input_img_layer(1))
    for n in tqdm.tqdm(range(times)):
        input_imgs = input_img_layer(sample_size)
        input_imgs.retain_grad()
        aug_imgs = aug_fn(input_imgs)
        loss = aug_imgs.mean()
        loss.backward()
        nonzero_grads = input_imgs.grad != 0
        nonzero_grads = nonzero_grads.double()
        sample_ratio = sample_ratio + nonzero_grads / (sample_size * times)

    return torch.sum(sample_ratio, dim=0)
