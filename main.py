import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil
import argparse

import sys
import os.path as osp
import time
import numpy as np
import pickle
import pdb

from data_utils import load_data, load_data_img
from torchvision.models import resnet50#, ResNet50_Weights
# This code part from https://github.com/sunlin-ai/diffusion_tutorial


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


class DimConverter(nn.Module):
    def __init__(self, input_dim=64, out_dim=64):
        super(DimConverter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            # # nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # # nn.ReLU()
        )
        
    def forward(self, x): 
        x = self.fc(x)
        return x

def main(config, logger, exp_dir, args):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    unet = Guide_UNet(config).cuda()
    # print(unet)
    
    # traj = np.load('./xxxxxx',
    #                allow_pickle=True)
    # traj = traj[:, :, :2]
    # head = np.load('./xxxxxx',
    #                allow_pickle=True)
    # traj = np.swapaxes(traj, 1, 2)
    # traj = torch.from_numpy(traj).float()
    # head = torch.from_numpy(head).float()
    # ###########################################################
    # # The input shape of traj and head list as follows:
    # # traj: [batch_size, 2, traj_length]   2: latitude and longitude
    # # head: [batch_size, 8]   8: departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed, start_id, end_id
    # ###########################################################
    # dataset = TensorDataset(traj, head)
    # dataloader = DataLoader(dataset,
    #                         batch_size=config.training.batch_size,
    #                         shuffle=True,
    #                         num_workers=8)
    _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori, train_loader_source_mix = load_data_img(config)
    dataloader = train_loader_source_ori

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    # lr = 2e-4  # Explore this - might want it lower when training on the full dataset
    lr = config.training.lr

    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    model_save = exp_dir + '/models/' + (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        losses = []  # Store losses for later plotting
        for _, batch_data in enumerate(dataloader):
            x0 = batch_data[0][:,:,:2].cuda() 
            label = batch_data[1].unsqueeze(1)
            
            # pad_mask = batch_data[0][:,:,2]!=0
            # x0 = x0[pad_mask].unsqueeze(0)
            
            max_feat = torch.max(batch_data[0][:,:,4:8], dim=1)[0] # v, a, j, br
            avg_feat = torch.sum(batch_data[0][:,:,3:8], dim=1) / (torch.sum(batch_data[0][:,:,2]!=0, dim=1)+1e-6).unsqueeze(1)
            # pdb.set_trace()
            total_dist = torch.sum(batch_data[0][:,:,3], dim=1).unsqueeze(1)
            total_time = torch.sum(batch_data[0][:,:,2], dim=1).unsqueeze(1)
            avg_dist = avg_feat[:,0].unsqueeze(1)
            avg_speed = avg_feat[:,1].unsqueeze(1)
            
            # head = torch.cat([avg_feat,max_feat,label],dim=1)
            head = torch.cat([total_dist, total_time, avg_dist, avg_speed, label],dim=1)
            head = head.cuda()
            
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, )).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            x0 = x0.permute(0,2,1)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head)
            # Compare the predictions with the targets
            if config.training.loss=='mse':
                loss = F.mse_loss(noise.float(), pred_noise)
            elif config.training.loss=='rmse':
                loss = torch.sqrt(F.mse_loss(noise.float(), pred_noise))
            else:
                raise NotImplemented
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        logger.info("<----Epoch-{}----> loss: {:.4f}".format(epoch,np.array(losses).mean()))

        if (epoch) % 100 == 0:
            m_path = model_save + f"/unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir + '/results/' + f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


def main_img(config, logger, exp_dir, args):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    unet = Guide_UNet(config).cuda()
    img_encoder = resnet50(True).cuda()
    dim_converter = DimConverter(input_dim=1000, out_dim=config.model.ch).cuda()
    for name,param in img_encoder.named_parameters():
        param.requires_grad = False 
    # print(unet)
    
    # traj = np.load('./xxxxxx',
    #                allow_pickle=True)
    # traj = traj[:, :, :2]
    # head = np.load('./xxxxxx',
    #                allow_pickle=True)
    # traj = np.swapaxes(traj, 1, 2)
    # traj = torch.from_numpy(traj).float()
    # head = torch.from_numpy(head).float()
    # ###########################################################
    # # The input shape of traj and head list as follows:
    # # traj: [batch_size, 2, traj_length]   2: latitude and longitude
    # # head: [batch_size, 8]   8: departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed, start_id, end_id
    # ###########################################################
    # dataset = TensorDataset(traj, head)
    # dataloader = DataLoader(dataset,
    #                         batch_size=config.training.batch_size,
    #                         shuffle=True,
    #                         num_workers=8)
    _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori, train_loader_source_mix = load_data_img(config)
    dataloader = train_loader_source_mix

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    # lr = 2e-4  # Explore this - might want it lower when training on the full dataset
    lr = config.training.lr

    # optimizer
    optim = torch.optim.AdamW([{"params": unet.parameters()},{"params": dim_converter.parameters()}], lr=lr)  # Optimizer
    # optimizer_g = Adam([
    #     {"params": G.parameters()},
    #     {"params": dim_converter.parameters()},
    # ], args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    model_save = exp_dir + '/models/' + (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        losses = []  # Store losses for later plotting
        for _, batch_data in enumerate(dataloader):
            print(1)
            x0 = batch_data[0][:,:,:2].cuda() 
            img = batch_data[1].cuda() 
            label = batch_data[2].unsqueeze(1)
            
            img_feat = img_encoder(img.float())
            img_feat = dim_converter(img_feat)
            
            max_feat = torch.max(batch_data[0][:,:,4:8], dim=1)[0] # v, a, j, br
            avg_feat = torch.sum(batch_data[0][:,:,3:8], dim=1) / (torch.sum(batch_data[0][:,:,2]!=0, dim=1)+1e-6).unsqueeze(1)
            # pdb.set_trace()
            total_dist = torch.sum(batch_data[0][:,:,3], dim=1).unsqueeze(1)
            total_time = torch.sum(batch_data[0][:,:,2], dim=1).unsqueeze(1)
            avg_dist = avg_feat[:,0].unsqueeze(1)
            avg_speed = avg_feat[:,1].unsqueeze(1)
            
            # head = torch.cat([avg_feat,max_feat,label],dim=1)
            head = torch.cat([total_dist, total_time, avg_dist, avg_speed, label],dim=1)
            head = head.float().cuda()
            
            t = torch.randint(low=0, high=n_steps, size=(len(x0) // 2 + 1, )).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            x0 = x0.permute(0,2,1)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head, img_feat)
            # Compare the predictions with the targets
            if config.training.loss=='mse':
                loss = F.mse_loss(noise.float(), pred_noise)
            elif config.training.loss=='rmse':
                loss = torch.sqrt(F.mse_loss(noise.float(), pred_noise))
            else:
                raise NotImplemented
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        logger.info("<----Epoch-{}----> loss: {:.4f}".format(epoch,np.array(losses).mean()))

        if (epoch) % 100 == 0:
            m_path = model_save + f"/unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir + '/results/' + f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
    parser.add_argument('--job_name', type=str, default='test', help='activation')
    parser.add_argument('--epoch', type=int, default=200, help='activation')
    parser.add_argument('--n_step', type=int, default=500, help='activation')
    parser.add_argument('--batch_size', type=int, default=1024, help='activation')
    parser.add_argument('--lr', type=float, default=2e-4, help='activation')
    parser.add_argument('--mode', type=str, default='label_only', help='activation')
    parser.add_argument('--filter_nopad', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--unnormalize', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--interpolated', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--guidance_scale', type=float, default=3., help='activation')
    parser.add_argument('--loss', type=str, default='mse', help='activation')

    tmp_args = parser.parse_args()
    
    print(args)
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    config.training.job_name = tmp_args.job_name
    config.model.mode = tmp_args.mode
    config.training.n_epochs = tmp_args.epoch
    config.training.batch_size = tmp_args.batch_size
    config.training.lr = tmp_args.lr
    config.training.loss = tmp_args.loss
    config.data.filter_nopad = tmp_args.filter_nopad
    config.data.unnormalize = tmp_args.unnormalize
    config.data.interpolated = tmp_args.interpolated
    config.diffusion.num_diffusion_timesteps = tmp_args.n_step
    config.model.guidance_scale = tmp_args.guidance_scale

    # root_dir = Path(__name__).resolve().parents[0]
    root_dir = "/home/yichen/DiffTraj/results"
    # result_name = '{}_steps={}_len={}_{}_bs={}'.format(
    #     config.data.dataset, config.diffusion.num_diffusion_timesteps,
    #     config.data.traj_length, config.diffusion.beta_end,
    #     config.training.batch_size)
    result_name = config.training.job_name
    exp_dir = root_dir + "/DiffTraj/" + result_name
    for d in ["/results", "/models", "/logs","/Files"]:
        os.makedirs(exp_dir + d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    # files_save = exp_dir / 'Files' / (timestamp + '/')
    files_save = exp_dir + '/Files/' + (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy('./utils/config.py', files_save)
    shutil.copy('./utils/Traj_UNet.py', files_save)

    logger = Logger(
        __name__,
        log_path=exp_dir + "/logs/" + (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)
    main_img(config, logger, exp_dir, args)
