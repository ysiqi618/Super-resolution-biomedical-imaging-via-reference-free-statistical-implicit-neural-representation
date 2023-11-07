import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image




def display_arr_stats(arr):
    shape, vmin, vmax, vmean, vstd = arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def create_grid_LR(offset_x, offset_y, h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(offset_x, 1, steps=h),
                                     torch.linspace(offset_y, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


class Img_2D_HR(Dataset):

    def __init__(self,isTIFF,img_path, HR_dim, offset_x, offset_y):
        '''
        img_dim: new image size [h, w]
        '''
        print('isTIFF=', isTIFF)
        if isTIFF == 1:
            im = Image.open(img_path)
            image = np.array(im)
            image = image.astype(np.float64)

        else:
            image = np.load(img_path)['data']  # [100, 320, 260] (0, 1)
        imsize = image.shape

        # complete as a squared image
        if not(imsize[0] == imsize[1]):
            zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

        # Scaling normalization
        self.max_val = np.max(image)
        image = image / self.max_val  # [0, 1]
        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
        display_tensor_stats(self.img)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.img_dim = image.shape
        print('img_dim:', image.shape)        
    def __getitem__(self, idx):

        # grid = create_grid(*self.img_dim[::-1])
        grid = create_grid_LR(self.offset_x, self.offset_y, *self.img_dim[::-1])
        print('*img_dim:', *self.img_dim[::-1])
        return grid, self.img# , self.max_val

    def __len__(self):
        return 1





