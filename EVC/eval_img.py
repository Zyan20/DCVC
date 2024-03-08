import torch
import os, math
from PIL import Image
import torch.nn.functional as F

from torchvision import transforms

from compressai.zoo import cheng2020_anchor

import numpy as np

inter_model = cheng2020_anchor(quality = 6, pretrained = True)
inter_model.cuda()

dataset = "D:/kodak"

images = os.listdir(dataset)

transform = transforms.Compose([
    transforms.ToTensor()
])


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def _compute_intra_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in out_net['likelihoods'].values()).item()

def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


psnrs = []
bpps = []
for x in images:
    path = os.path.join(dataset, x)

    img_pil = Image.open(path)
    image = transform(img_pil)
    image = image.cuda().unsqueeze(0)

    pic_height = image.shape[2]
    pic_width = image.shape[3]


    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
    x_padded = F.pad(
        image,
        (padding_l, padding_r, padding_t, padding_b),
        mode="constant",
        value=0,
    )

    out = inter_model(x_padded)

    bpp = _compute_intra_bpp(out)

    recon_frame = out["x_hat"]

    x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
    psnr = PSNR(x_hat, image)

    bpps.append(bpp)
    psnrs.append(psnr)

    print(x, bpp, psnr)


print(np.mean(bpps))
print(np.mean(psnrs))
