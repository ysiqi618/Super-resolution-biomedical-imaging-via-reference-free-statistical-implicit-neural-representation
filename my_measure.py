#### Compute RMSE, PSNR, and SSIM of images #####
#### Siqi Ye, Dept. Rad. Onc., Stanford Univ. 
#### 2022-04-20

## inputs should be tensors

import torch
import torch.nn as nn
from skimage.metrics import structural_similarity 



def my_rmse(img, img_ref): 
    loss_fn = torch.nn.MSELoss()
    rmse = torch.sqrt(loss_fn(img, img_ref))
    #print('rmse:', rmse, 'rmse_item:', rmse.item())
    return rmse.item()

def my_psnr(img, img_ref):
    loss_fn = torch.nn.MSELoss()
    rmse = torch.sqrt(loss_fn(img, img_ref))
    maxf = torch.max(img_ref)
   # print('maxf:', maxf)
    psnr = 20*torch.log10(maxf) - 20 * torch.log10(rmse)
    return psnr.item()

def my_ssim(img, img_ref):
    ssim = structural_similarity(img.squeeze().numpy(), img_ref.squeeze().numpy())  # greyscale
 #   print('img shape:', img.shape)
   # print(ssim)
    return ssim
