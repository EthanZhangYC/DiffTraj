
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


use_gpu=False


temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)
config = SimpleNamespace(**temp)
unet = Guide_UNet(config)#.cuda()
# unet.load_state_dict(torch.load('/home/yichen/DiffTraj/results/DiffTraj/0903_attr_label_embed/models/09-03-18-30-16/unet_200.pt'))
unet.load_state_dict(torch.load('/home/yichen/DiffTraj/results/DiffTraj/0904_attr_label_embed_only/models/09-04-16-13-31/unet_200.pt'))



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


minmax_list=[
    (45.230416, 45.9997262293), (-74.31479102, -72.81248199999999),  \
    (0.9999933186918497, 1198.999998648651), # time
    (0.0, 50118.17550774085), # dist
    (0.0, 49.95356703911097), # speed
    (-9.99348698095659, 9.958323482935628), #acc
    (-39.64566646191948, 1433.3438889109589), #jerk
    (0.0, 359.95536847383516) #bearing
]

# for i in range(6):
#     if i==2:
#         continue
#     traj[:,i] = traj[:,i] * (minmax_list[i][1]-minmax_list[i][0]) + minmax_list[i][0]
    
batchsize = 5
head = np.array([1,1,1,1,1,1,1])
head = np.tile(head,[5,1])
head = torch.from_numpy(head).float()


# # load head information for guide trajectory generation
# batchsize = 500
# head = np.load('heads.npy', allow_pickle=True)
# head = torch.from_numpy(head).float()
dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)




# # the mean and std of head information, using for rescaling
# # departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
# hmean=[0, 10283.41600429,   961.66920921,   292.30299616,    36.02766493, 10.98568072]
# hstd=[1, 8782.599246414231, 379.41939897358264, 107.24874828393955, 28.749924691281066, 8.774629812281198]
# mean = np.array([104.07596303,   30.68085491])
# std = np.array([2.15106194e-02, 1.89193207e-02])
# # the original mean and std of trajectory length, using for rescaling the trajectory length
# len_mean = 292.30299616  # Chengdu
# len_std = 107.2487482839  # Chengdu



lengths=248
Gen_traj = []
Gen_head = []
for i in tqdm(range(1)):
    print(i)
    head = next(iter(dataloader))
    # lengths = head[:, 3]
    # lengths = lengths * len_std + len_mean
    # lengths = lengths.int()
    # tes = head[:,:6].numpy()
    # Gen_head.extend((tes*hstd+hmean))
    if use_gpu:
        head = head.cuda()
    # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length).cuda()
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        print(i,j)
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        if use_gpu:
            t = t.to(x.device)
            next_t = next_t.to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    for j in range(batchsize):
        print(j)
        new_traj = resample_trajectory(trajs[j].T, lengths)
        # new_traj = resample_trajectory(trajs[j].T, lengths[j])
        # new_traj = new_traj * std + mean
        Gen_traj.append(new_traj)
    break




plt.figure(figsize=(8,8))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
plt.tight_layout()
plt.savefig('geolife_traj.png')
plt.show()