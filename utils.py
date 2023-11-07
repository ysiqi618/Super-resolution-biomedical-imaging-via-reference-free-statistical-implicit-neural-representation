import os
import yaml
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


from data_micro import Img_2D_HR

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory



################################## HR loader, 2022-07-20 ########################
def get_data_loader_HR(data, isTIFF,img_path, HR_dim, offset_x, offset_y, train, batch_size, num_workers = 4, return_data_idx = False):
    dataset = Img_2D_HR(isTIFF,img_path, HR_dim, offset_x, offset_y)
    loader = DataLoader(dataset = dataset,
                        batch_size = batch_size,
                        shuffle = train,
                        num_workers = num_workers)
    return loader



