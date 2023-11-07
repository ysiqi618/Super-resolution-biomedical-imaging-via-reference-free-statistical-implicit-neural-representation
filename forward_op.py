import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

def img_degrad_2d(img_hr, shift_vec, shift_dim, ker_type, ker_size, downscale):
    # shift_vec: how much a tensor is shifted. e.g., (1, -1): (down 1, up 1)
    # shift_dim: shift along which dimension. e.g., (0,1)
    # img_hr: tensor
    
    img_shift = torch.roll(img_hr.squeeze(), shifts = shift_vec, dims = shift_dim)

    if ker_type == 'Avg': # average 
        pad = int(np.ceil(ker_size[0]/2 - 1))
       # print('pad is ...',pad)
        blur_filter = nn.AvgPool2d(ker_size, padding=pad, stride=(downscale, downscale))

        img_degrad = blur_filter(img_shift[None,None,...])
        # print('img_degrad shape:', img_degrad.shape)
    elif ker_type == 'Down':
        img_degrad = img_shift[None, None, ::2, ::2]
    return img_degrad


