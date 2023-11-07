## HR image restoration
## loss = 1/N sum_i \|G_i M_\theta (coord) - X_LR_i \|^2
## G_i: forward model for the i-th sample
## For SR problem, G_i is a composed operation of downsampling, blurring and motion
## G_i can be generalized to other applications. e.g., forward model in CT/MRI/...
########### Siqi Ye, 2022-07-19, Rad Onc., Stanford ###################


import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tensorboardX
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader_HR
from my_measure import my_rmse, my_psnr, my_ssim
from forward_op import img_degrad_2d

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--pretrain', action='store_true', help="load pretrained model weights")

parser.add_argument('--num_LR', type=int, default=-1, help="num_LR")
parser.add_argument('--downscale', type=int, default=-1, help="downscale")
parser.add_argument('--lr', type=float, default=-1, help="learn rate")


# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

if opts.num_LR > 0:
    config['num_of_LRs'] = int(opts.num_LR)
    print('opts.num_LR', config['num_of_LRs'])
if opts.downscale > 0:
    config['downscale'] = int(opts.downscale)
    print('opts.downscale', config['downscale'])

if opts.lr > 0:
    config['lr'] = float(opts.lr)
    print('opts.lr', config['lr'])


cudnn.benchmark = True

slice_idx = list(np.linspace(0,0, num=1, dtype = int))

scheduler_step = config['step_size']
scheduler_gamma = config['gamma']

############################ start main training ######################3
# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
if opts.pretrain: 
    output_subfolder = config['data'] + '_pretrain'
else:
    output_subfolder = config['data']
if config['ker_type'] == 'Avg':
    model_name = os.path.join(output_subfolder, output_folder + '/SRscale{}_numLRs{}_blur{}_kerSz{}_{}_{}_{}_{}_{}_{}_lr{:.2g}_StepLR{}_{}_wd{}_encoder_{}_embedSz{}_scale{}' \
    .format(config['downscale'], config['num_of_LRs'], config['ker_type'], config['ker_size'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['optimizer'], config['lr'], scheduler_step, scheduler_gamma, config['weight_decay'], config['encoder']['embedding'], config['encoder']['embedding_size'], config['encoder']['scale']))
elif config['ker_type'] == 'Down':
    model_name = os.path.join(output_subfolder, output_folder + '/SRscale{}_numLRs{}_{}_{}_{}_{}_{}_{}_{}_lr{:.2g}_StepLR{}_{}_wd{}_encoder_{}_embedSz{}_scale{}' \
    .format(config['downscale'], config['num_of_LRs'], config['ker_type'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['optimizer'], config['lr'], scheduler_step, scheduler_gamma, config['weight_decay'], config['encoder']['embedding'], config['encoder']['embedding_size'], config['encoder']['scale']))

print(model_name)

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
log_path = os.path.join(output_directory, 'logs')
if not os.path.exists(log_path):
    print('create log path: {}'.format(log_path))
    os.makedirs(log_path)


# Setup input encoder:
encoder = Positional_Encoder(config['encoder'])

# Setup model
if config['model'] == 'SIREN':
    model = SIREN(config['net'])
elif config['model'] == 'FFN':
    model = FFN(config['net'])
else:
    raise NotImplementedError
model.cuda()
model.train()

# Load pretrain model
if opts.pretrain:
    model_path = config['pretrain_model_path'].format(config['downscale'], \
                    config['ker_type'], config['ker_size'], config['model'], config['net']['network_input_size'], config['net']['network_width'], \
                    config['net']['network_depth'], config['encoder']['embedding_size'], config['encoder']['scale'])
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['net'])
    encoder.B = state_dict['enc']
    print('Load pretrain model: {}'.format(model_path))

# Setup optimizer
if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
elif config['optimizer'] == 'SGD':
    optim = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])  # momentum=config['momentum'], dampening=config['dampening'], nesterov=True)
else:
    NotImplementedError


scheduler = torch.optim.lr_scheduler.StepLR(optim, scheduler_step, scheduler_gamma)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=scheduler_gamma,patience=10,verbose=True,threshold=0.2, threshold_mode='abs',min_lr=5e-7, eps=1e-7)
 
# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError


# setup LR shifts
shift_positions = ['00', '11', '01', '10']

shifts = np.array([[0,0],
                   [-1,-1],
                   [0,-1],
                   [-1,0]])

shift_dim =tuple([0,1])

# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader_HR(config['data'],config['isTIFF'], config['img_path'], config['HR_dim'],0, 0, train=True, batch_size=config['batch_size'])

ker_type = config['ker_type']
ker_size = config['ker_size']
downscale = config['downscale']
print('ker_size:', ker_size)
        
train_loss_history = []
test_rmse_history = []
test_psnr_history = []
test_ssim_history = []
val_iter_history = []

for it, (grid, img_HR) in enumerate(data_loader):
    # Input coordinates (x,y) grid and target image
    # img_HR: only used for generating LR images and validation. Not used in network training
    grid = grid.cuda()  # [bs, h, w, 2], [0, 1]
    torchvision.utils.save_image(img_HR.cpu().permute(0, 3, 1, 2).data, os.path.join(image_directory, "HR_org.png"))
     
    print('dataloader HR shape:', img_HR.shape)
    print('grid shape:', grid.shape)
    ### generate LR obervations #########
    for count in range(config['num_of_LRs']):
        shift_vec = tuple(shifts[count,:])
      #  print('shift vec:', shift_vec)
        LR_i = img_degrad_2d(img_HR.permute(0,3,1,2), shift_vec, shift_dim, ker_type, ker_size, downscale)
        torchvision.utils.save_image(LR_i.cpu().data, os.path.join(image_directory, "genLR_"+shift_positions[count]+'.png'))
        if count == 0:
            imgs_LR = LR_i
        else:
            imgs_LR = torch.cat([imgs_LR, LR_i], dim=0)
       #     print('cat LR dims:', imgs_LR.shape)
    ### Train model
    for iterations in range(max_iter):
        model.train()
        optim.zero_grad()
        train_loss_total = 0
        train_embedding = encoder.embedding(grid)  # [B, H, W, embedding*2]
        train_output = model(train_embedding)  # [B, H, W, 3]
        if iterations == 0:
            test_loss = my_rmse(train_output, img_HR.cuda())
            test_psnr = my_psnr(train_output, img_HR.cuda())
            test_ssim = my_ssim(train_output.detach().cpu(), img_HR.cpu())
            print("Initial model: Test rmse {:.4g} | Test psnr: {:.4g} | Test ssim: {:.3g}".format(test_loss, test_psnr, test_ssim))

        for count in range(imgs_LR.shape[0]):
            shift_vec = tuple(shifts[count,:])
            train_projs = img_degrad_2d(train_output.permute(0,3,1,2), shift_vec, shift_dim, ker_type, ker_size, downscale)
            LR_i = imgs_LR[count, ...]
            LR_i = LR_i[None,...]
          #  print('LR_i shape:', LR_i.shape)
            train_loss = 0.5 * loss_fn(train_projs, LR_i.cuda())
            train_loss_total += train_loss
        train_loss_total = train_loss_total/imgs_LR.shape[0]
        train_loss_history.append(train_loss_total.item())     
        train_loss_total.backward()
        optim.step()
        scheduler.step() 
        # Compute training psnr
        if (iterations + 1) % config['log_iter'] == 0:
            
            fig1 = plt.figure()
            plt.semilogy(np.array(train_loss_history))
            plt.title('Train loss (data-fit only)')
            plt.xlabel('iterations')
            plt.ylabel('loss (log)')
            plt.grid(True)
            fig1.savefig(os.path.join(log_path + "/train_loss_history.png"),dpi=fig1.dpi)
            print("[Iteration: {}/{}] Train loss: {:.4g}".format(iterations + 1, max_iter, train_loss))
            torch.save(train_loss_history, os.path.join(log_path + "/train_loss_history.pt"))

        # Compute testing psnr
        if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(grid)
                test_output = model(test_embedding)

                test_loss = my_rmse(test_output, img_HR.cuda())
                test_psnr = my_psnr(test_output, img_HR.cuda())
                test_ssim = my_ssim(test_output.cpu(), img_HR.cpu())
            test_rmse_history.append(test_loss)
            test_psnr_history.append(test_psnr)
            test_ssim_history.append(test_ssim)

            val_iter_history.append(iterations)
            fig2 = plt.figure()
            plt.plot(np.array(val_iter_history), np.array(test_rmse_history))
            plt.title('Test RMSE')
            plt.xlabel('iterations')
            plt.ylabel('RMSE')
            plt.grid(True)
            fig2.savefig(os.path.join(log_path + "/validate_RMSE.png"),dpi=fig2.dpi)
            torch.save(test_rmse_history, os.path.join(log_path + "/test_rmse_history.pt"))
            
            fig3 = plt.figure()
            plt.plot(np.array(val_iter_history), np.array(test_psnr_history))
            plt.title('Test PSNR')
            plt.xlabel('iterations')
            plt.ylabel('PSNR')
            plt.grid(True)
            fig3.savefig(os.path.join(log_path + "/validate_PSNR.png"),dpi=fig3.dpi)
            torch.save(test_psnr_history, os.path.join(log_path + "/test_psnr_history.pt"))
            
            fig4 = plt.figure()
            plt.plot(np.array(val_iter_history), np.array(test_ssim_history))
            plt.title('Test SSIM')
            plt.xlabel('iterations')
            plt.ylabel('SSIM')
            plt.grid(True)
            fig4.savefig(os.path.join(log_path + "/validate_SSIM.png"),dpi=fig4.dpi)
            torch.save(test_ssim_history, os.path.join(log_path + "/test_ssim_history.pt"))

           # Must transfer to .cpu() tensor firstly for saving images
            torchvision.utils.save_image(test_output.permute(0,3,1,2).cpu().data, os.path.join(image_directory, "recon_{}_{:.4g}dB_{:.3g}.png".format(iterations + 1, test_psnr, test_ssim)))
            print("[Validation Iteration: {}/{}] Test rmse {:.4g} | Test psnr: {:.4g} | Test ssim: {:.3g}".format(iterations + 1, max_iter, test_loss, test_psnr, test_ssim))
            
        

 ########################## record degraded images from \hat{HR} ###########################
            for count in range(imgs_LR.shape[0]):
                shift_vec = tuple(shifts[count,:])
                test_projs = img_degrad_2d(test_output.permute(0,3,1,2), shift_vec, shift_dim, ker_type, ker_size, downscale)
                torchvision.utils.save_image(test_projs.cpu().data, os.path.join(image_directory, "recon_genLR_"+shift_positions[count]+'.png'))
 
 
             ###############################################################################################
 
             
       #    Save final model
            model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
            torch.save({'net': model.state_dict(), \
                        'enc': encoder.B, \
                        'opt': optim.state_dict(), \
                        }, model_name)
 
 

        
