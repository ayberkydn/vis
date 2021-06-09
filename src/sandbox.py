import torch, torchvision
import matplotlib.pyplot as plt
import kornia
import cv2
import tqdm
import numpy as np
import einops
import timm
from PIL import Image
import random
import os
from input_layer import InputImageLayer

input_image_layer = InputImageLayer(
    shape=[512, 512],
    param_fn=torch.nn.Sigmoid(),
)

augmentation = torch.nn.Sequential(
    kornia.augmentation.RandomRotation(
        degrees=30,
        same_on_batch=False,
    ),
    kornia.augmentation.RandomResizedCrop(
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(0.5, 2.0),  # aspect ratio
        same_on_batch=False,
    ),
)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


models = [
    "vit_base_patch16_224",
    # "resnet50",
    # "tf_efficientnet_b0_ap",
    # "adv_inception_v3",
    # "inception_v4",
    # "dla102",
    # "densenet121",
    # "mobilenetv3_large_100",
    # "resnext101_32x8d",
    # "seresnet152d",
    # "ig_resnext101_32x16d",
    # "nf_resnet50",
]


TARGET_CLASS = 35
ITERATIONS = 3000
BATCH_SIZE = 1
TV_LOSS_COEFF = 0

optimizer = torch.optim.Adam(input_image_layer.parameters(), lr=0.01)


for n in tqdm.tqdm(range(ITERATIONS)):

    input_img = input_image_layer()

    net_dict = random.choice(network_dicts)

    net = net_dict["net"]
    input_img_batch = einops.repeat(
        input_img, "b c h w -> (b repeat) c h w", repeat=BATCH_SIZE
    )
    transformed_input_img_batch = batch_transforms(input_img_batch)

    out = net(transformed_input_img_batch)
    target = out[:, TARGET_CLASS].mean()
    activations.append(target.item())

    tv_loss = kornia.losses.total_variation(input_img[0]) / torch.numel(input_img[0])

    loss = tv_loss * TV_LOSS_COEFF - target

    loss.backward()

    if n % 10 == 0:
        images.append(np.uint8(kornia.tensor_to_image(input_img[0]) * 255))

    optimizer.step()
    optimizer.zero_grad()

plt.figure()
plt.plot(max_list)
plt.plot(grads_max_list)
plt.savefig(os.path.join(output_path, "max_list.png"))
plt.close()

plt.figure()
plt.scatter(np.arange(len(activations)), activations, s=0.01)
plt.savefig(os.path.join(output_path, "activations.png"))
plt.close()

plt.figure(figsize=[10, 10])
plt.imshow(kornia.tensor_to_image(input_img[0]))

title_string = ""
with torch.no_grad():
    for n, net_dict in tqdm.tqdm(enumerate(network_dicts)):
        net = net_dict["net"]
        input_img_batch = einops.repeat(
            input_img, "b c h w -> (b repeat) c h w", repeat=32
        )
        transformed_input_img_batch = batch_transforms(input_img_batch)

        out = net(transformed_input_img_batch).mean(dim=0)

        out_argmax = torch.argmax(out, -1).item()
        out_probs = torch.softmax(out, -1)
        out_label = labels[out_argmax]
        out_prob = out_probs[out_argmax].item()
        title_string += f"{net_dict['name']} says it's a {out_label} with probability {out_prob:0.3f}. \n"

plt.title(title_string)
plt.savefig(os.path.join(output_path, "final_image_with_probs.png"))

SHAPE = input_tensor.shape[-2:]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_images = cv2.VideoWriter(
    os.path.join(output_path, "animation.avi"), fourcc, 30.0, SHAPE
)

for frame in images:
    frame = cv2.resize(frame, SHAPE)
    if frame.max() > 1:
        frame = frame * 255.0

    frame_int = np.uint8(frame)
    out_images.write(frame_int)

out_images.release()

cv2.imwrite(
    os.path.join(output_path, "final_img.png"),
    cv2.cvtColor(kornia.tensor_to_image(input_img[0]) * 255, cv2.COLOR_BGR2RGB),
)
