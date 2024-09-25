
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# from tqdm.notebook import tqdm
from tqdm import tqdm
from types import SimpleNamespace
from utils.config import args
from utils.utils import *
from torch.utils.data import DataLoader
from utils.Traj_UNet import *
import argparse
from main import load_data

use_gpu=False

parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
parser.add_argument('--job_name', type=str, default='test', help='activation')
parser.add_argument('--mode', type=str, default='label_only', help='activation')
parser.add_argument('--resume', type=str, default='label_only', help='activation')
parser.add_argument('--interpolated', action='store_true', help='whether to output attention in encoder')

tmp_args = parser.parse_args()

temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)
config = SimpleNamespace(**temp)

config.training.job_name = tmp_args.job_name
config.model.mode = tmp_args.mode
config.data.filter_nopad = True
config.data.unnormalize = True
config.data.interpolated = tmp_args.interpolated
 
    
unet = Guide_UNet(config)#.cuda()

config.training.batch_size = batchsize = 5

Gen_traj = []
Gen_head = []
_,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(config)
batch_data = next(iter(train_loader_target_ori))
x0 = batch_data[0][:,:,:2] #.cuda() 
Gen_traj.append(x0[0])
# Gen_traj.append(x0[0])
x0 = x0.permute(0,2,1)


# model_dir_list=[
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_nstep10/models/09-16-23-53-52/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_nstep50/models/09-16-17-35-29/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_nstep100/models/09-16-17-24-26/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512/models/09-16-23-56-03/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_nstep1000/models/09-16-17-24-11/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_nstep2000/models/09-16-23-54-44/unet_500.pt'
# ]
# model_dir_list=[
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_scale01/models/09-16-17-19-39/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_scale1/models/09-16-17-20-36/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512/models/09-16-23-56-03/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_scale10/models/09-16-23-53-31/unet_500.pt',
# ]
# model_dir_list=[
#     '/home/yichen/DiffTraj/results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512/models/09-16-23-56-03/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_lr1e4/models/09-17-08-56-19/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_lr5e5/models/09-17-08-56-06/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_lr1e5/models/09-17-08-56-41/unet_500.pt',
# ]
# model_dir_list=[
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_scale0_rmse/models/09-17-16-22-59/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_scale0_rmse/models/09-17-16-22-59/unet_100.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_scale0_rmse/models/09-17-16-22-59/unet_300.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_nopad_unnorm_bs512_scale0_rmse/models/09-17-16-22-59/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch2000_nopad_unnorm_bs512_rmse/models/09-17-16-03-22/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch2000_nopad_unnorm_bs512_rmse/models/09-17-16-03-22/unet_1000.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch2000_nopad_unnorm_bs512_rmse/models/09-17-16-03-22/unet_1500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch2000_nopad_unnorm_bs512_rmse/models/09-17-16-03-22/unet_2000.pt',
# ]
# model_dir_list=[
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_unnorm_bs512_interpolated_mse/models/09-18-13-05-40/unet_100.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_unnorm_bs512_interpolated_mse/models/09-18-13-05-40/unet_300.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_unnorm_bs512_interpolated_mse/models/09-18-13-05-40/unet_500.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_unnorm_bs512_interpolated_rmse/models/09-18-13-04-35/unet_300.pt',
#     '/home/yichen/DiffTraj/results/DiffTraj/0917_label_max_vajb_epoch500_unnorm_bs512_interpolated_rmse/models/09-18-13-04-35/unet_500.pt',
# ]
model_dir_list=[
    "/home/yichen/DiffTraj/results/DiffTraj/0924_label_oridiff_epoch500_nopad_bs512/models/09-24-22-05-47/unet_500.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/0925_label_oridiff_epoch500_nopad_bs512_shuffle/models/09-25-08-05-15/unet_500.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/0925_label_oridiff_epoch500_nopad_bs512_shuffle_unnorm/models/09-25-09-12-37/unet_500.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/0925_label_oridiff_epoch500_nopad_bs512_shuffle_lr1e4/models/09-25-10-08-19/unet_500.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/0925_label_oridiff_epoch500_nopad_bs512_shuffle_lr1e5/models/09-25-10-05-46/unet_500.pt"
]
cnt=0
for model_dir in model_dir_list:
    # cnt+=1
    # if cnt==5:
    #     Gen_traj.append(x0[0])

    # unet.load_state_dict(torch.load())
    # ckpt_dir = '/home/yichen/DiffTraj/results/DiffTraj/0903_attr_label_embed/models/09-03-18-30-16/unet_200.pt'
    # ckpt_dir = '/home/yichen/DiffTraj/results/DiffTraj/0904_attr_label_embed_only/models/09-04-16-13-31/unet_200.pt'
    # ckpt_dir = '/home/yichen/DiffTraj/results/DiffTraj/0904_attr_label_embed_only/models/09-04-16-13-31/unet_200.pt'
    # ckpt_dir = '/home/yichen/DiffTraj/results/DiffTraj/0911_label_avgmax_vajb/models/09-11-10-54-11/unet_200.pt'
    # ckpt_dir = tmp_args.resume
    ckpt_dir = model_dir
    unet.load_state_dict(torch.load(ckpt_dir), strict=False)



    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps)#.cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    if use_gpu:
        beta = beta.cuda()
        unet.cuda()

    eta=0.0
    timesteps=100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)


    # minmax_list=[
    #     (45.230416, 45.9997262293), (-74.31479102, -72.81248199999999),  \
    #     (0.9999933186918497, 1198.999998648651), # time
    #     (0.0, 50118.17550774085), # dist
    #     (0.0, 49.95356703911097), # speed
    #     (-9.99348698095659, 9.958323482935628), #acc
    #     (-39.64566646191948, 1433.3438889109589), #jerk
    #     (0.0, 359.95536847383516) #bearing
    # ]

    # for i in range(6):
    #     if i==2:
    #         continue
    #     traj[:,i] = traj[:,i] * (minmax_list[i][1]-minmax_list[i][0]) + minmax_list[i][0]
        
    head = np.array([1,1,1,1,1,1,1])
    head = np.array([7e-2, 5.05e-1, 2.6863e-2, 23.61, 7e-2, 5.05e-1, 2.6863e-2, 23.61, 0])   
    head = np.array([1.4630e+00, 6.2562e-04, 5.6896e-03, 2.3672e+01, 4.5764e+00, 1.2705e+00,
            1.3502e+00, 2.7067e+02, 3])   
    head = np.array([0.2, 0.06, 0.0055, 23, 0.1, 0.1, 0.1, 0, 0])   
    head = np.array([3.6685e-02, 1.2650e+03, 1.4792e-04, 2.9288e-02, 0.0000e+00])   
    # [3.9043e-02, 1.2200e+03, 1.5743e-04, 3.1675e-02, 0.0000e+00]
    # [6.8350e-02, 4.9600e+02, 2.7560e-04, 1.3826e-01, 3.0000e+00]
    head = np.tile(head,[5,1])
    head = torch.from_numpy(head).float()

    # # load head information for guide trajectory generation
    # batchsize = 500
    # head = np.load('heads.npy', allow_pickle=True)
    # head = torch.from_numpy(head).float()
    dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)


    # _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(config)


    # # the mean and std of head information, using for rescaling
    # # departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
    # hmean=[0, 10283.41600429,   961.66920921,   292.30299616,    36.02766493, 10.98568072]
    # hstd=[1, 8782.599246414231, 379.41939897358264, 107.24874828393955, 28.749924691281066, 8.774629812281198]
    # mean = np.array([104.07596303,   30.68085491])
    # std = np.array([2.15106194e-02, 1.89193207e-02])
    # # the original mean and std of trajectory length, using for rescaling the trajectory length
    # len_mean = 292.30299616  # Chengdu
    # len_std = 107.2487482839  # Chengdu

    # n_steps = config.diffusion.num_diffusion_timesteps
    # beta = torch.linspace(config.diffusion.beta_start,
    #                         config.diffusion.beta_end, n_steps).cuda()
    # alpha = 1. - beta
    # alpha_bar = torch.cumprod(alpha, dim=0)
    # print(alp)


    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise


    lengths=248

    for i in tqdm(range(1)):
        head = next(iter(dataloader))
        # lengths = head[:, 3]
        # lengths = lengths * len_std + len_mean
        # lengths = lengths.int()
        # tes = head[:,:6].numpy()
        # Gen_head.extend((tes*hstd+hmean))
        
        # batch_data = next(iter(train_loader_target_ori))
        # x0 = batch_data[0][:,:,:2]#.cuda() 
        # x0 = x0.permute(0,2,1)
        # # # Start with random noise
        # # x = torch.randn(batchsize, 2, config.data.traj_length)
        
        if use_gpu:
            head = head.cuda()
            x = x.cuda()
        ims = []
        # n = x.size(0)
        n = batchsize
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).long()
            next_t = (torch.ones(n) * j).long()
            if use_gpu:
                t = t.to(x.device)
                next_t = next_t.to(x.device)
            with torch.no_grad():
                x, noise = q_xt_x0(x0, t)
                pred_noise = unet(x, t, head)
                x = p_xt(x, pred_noise, t, next_t, beta, eta)
                if i % 10 == 0:
                    ims.append(x.cpu().squeeze(0))
        trajs = ims[-1].cpu().numpy()
        trajs = trajs[:,:2,:]
        
        # resample the trajectory length
        # for j in range(batchsize):
        for j in range(1):
            new_traj = resample_trajectory(trajs[j].T, lengths)
            # new_traj = resample_trajectory(trajs[j].T, lengths[j])
            # new_traj = new_traj * std + mean
            
            # lat_min,lat_max = (45.230416, 45.9997262293)
            # lon_min,lon_max = (-74.31479102, -72.81248199999999)
            lat_min,lat_max = (18.249901, 55.975593)
            lon_min,lon_max = (-122.3315333, 126.998528)
            
            # print(new_traj)
            
            new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
            new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
            Gen_traj.append(new_traj)
        break

    # plt.figure(figsize=(8,8))
    # traj=x0.permute(0,2,1)[0]
    # pdb.set_trace()
    # plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
    # plt.tight_layout()
    # plt.savefig('geolife_traj.png')
    # exit()

fig = plt.figure(figsize=(12,8))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    ax1 = fig.add_subplot(231+i)  
    ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
    # # plt.plot(traj[:,0],traj[:,1],'o',color='black',alpha=0.1)
    # plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
plt.tight_layout()
plt.savefig('mtl_traj_inter.png')
# print(new_traj)
plt.show()

