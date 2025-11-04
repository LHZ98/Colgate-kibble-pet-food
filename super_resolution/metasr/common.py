import random

import numpy as np
import skimage.color as sc

import torch
from torchvision import transforms
## 02Feb: changed get_patch line73 and argument line114. get_patch is backed up as get_patch_ori
## set_channel is modified for OCT images, where n_channels and c = 1
## np2Tensor line 102 is modified
def get_patch_ori(*args, patch_size=96, scale=1, multi_scale=False):
    ih, iw = args[0].shape[:2]

    multi_scale = True
    if multi_scale:
        tp = int(scale* patch_size)
        ip = patch_size
    else:
        tp = int(scale* patch_size)
        ip = patch_size

    #ix = random.randrange(0, iw - ip + 1)
    #iy = random.randrange(0, ih - ip + 1)
    #ix = random.randrange(0, (iw-ip)//(scale*10))*scale*10
    #iy  = random.randrange(0, (ih-ip)//(scale*10))*scale*10
    if scale==int(scale):
        step = 1
    elif (scale*2)== int(scale*2):
        step = 2
    elif (scale*5) == int(scale*5):
        step = 5
    else:
        step = 10

    ix = random.randrange(0, (iw-ip)//step)*step
    iy = random.randrange(0, (ih-ip)//step) *step

    tx, ty = int(scale * ix), int(scale * iy)
    print(args[0].shape)
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

def get_patch(*args, patch_size=96, scale=1, multi_scale=False):
    ih, iw = args[0].shape[:2]

    multi_scale = True
    if multi_scale:
        tp = int(scale* patch_size)
        ip = patch_size
    else:
        tp = int(scale* patch_size)
        ip = patch_size

    if scale==int(scale):
        step = 1
    elif (scale*2)== int(scale*2):
        step = 2
    elif (scale*5) == int(scale*5):
        step = 5
    else:
        step = 10

    # print(args[0].shape)
    # print(iw-ip,ih-ip)
    if ih-ip<=0:
        print(ih,ip)
    ix = random.randrange(0, (iw-ip)//step)*step
    iy = random.randrange(0, (ih-ip)//step) *step

    tx, ty = int(scale * ix), int(scale * iy)

    ret = [
        args[0][iy:iy + ip, ix:ix + ip],
        *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        # print('1:',img.shape)
        # print('2:',n_channels)
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        elif n_channels == 1 and c == 1:
            img = img;
        elif n_channels == 3 and c == 3:
            img = img
        else:
            print('Something is wrong with the input images!')
        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=False):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0)

        return img

    return [_augment(a) for a in args]

