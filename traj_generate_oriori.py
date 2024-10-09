import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# from tqdm.notebook import tqdm
from types import SimpleNamespace
from utils.Traj_UNet_ori import *
from utils.config_ori import args
from utils.utils import *
from torch.utils.data import DataLoader


temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)

# config.training.batch_size = batchsize = 2
unet = Guide_UNet(config).cuda()
# load the model
unet.load_state_dict(torch.load('./model.pt'))


n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta=0.0
timesteps=100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# load head information for guide trajectory generation
batchsize = 2
head = np.load('heads.npy',
                   allow_pickle=True)
head = np.load('heads.npy', allow_pickle=True)
head = head[:2]
head = torch.from_numpy(head).float()
# dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)


# the mean and std of head information, using for rescaling
# departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
hmean=[0, 10283.41600429,   961.66920921,   292.30299616,    36.02766493, 10.98568072]
hstd=[1, 8782.599246414231, 379.41939897358264, 107.24874828393955, 28.749924691281066, 8.774629812281198]
mean = np.array([104.07596303,   30.68085491])
std = np.array([2.15106194e-02, 1.89193207e-02])
# the original mean and std of trajectory length, using for rescaling the trajectory length
len_mean = 292.30299616  # Chengdu
len_std = 107.2487482839  # Chengdu

Gen_traj = []
Gen_head = []
# for i in tqdm(range(1)):
# head = next(iter(dataloader))
lengths = head[:, 3]
lengths = lengths * len_std + len_mean
lengths = lengths.int()
# tes = head[:,:6].numpy()
# Gen_head.extend((tes*hstd+hmean))
head = head.cuda()
# Start with random noise
x0 = torch.randn(batchsize, 2, config.data.traj_length).cuda()
ims = []
n = x0.size(0)
seq_next = [-1] + list(seq[:-1])
x = x0
for i, j in zip(reversed(seq), reversed(seq_next)):
    t = (torch.ones(n) * i).to(x.device)
    next_t = (torch.ones(n) * j).to(x.device)
    with torch.no_grad():
        pred_noise = unet(x, t, head)
        # print(pred_noise.shape)
        x = p_xt(x, pred_noise, t, next_t, beta, eta)
        if i % 10 == 0:
            ims.append(x.cpu().squeeze(0))
trajs = ims[-1].cpu().numpy()
trajs = trajs[:,:2,:]
# resample the trajectory length
# for j in range(batchsize):
for j in range(1):
    new_traj = resample_trajectory(trajs[j].T, lengths[j])
    new_traj = new_traj * std + mean
    Gen_traj.append(new_traj)
# break



# fig = plt.figure(figsize=(12,12))
# for i in range(len(Gen_traj)):
#     traj=Gen_traj[i]
#     ax1 = fig.add_subplot(441+i)  
#     ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# plt.tight_layout()
# plt.savefig(filename)
# plt.show()

plt.figure(figsize=(8,8))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
plt.tight_layout()
plt.savefig('Chengdu_traj.png')
plt.show()