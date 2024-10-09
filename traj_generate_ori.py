import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from types import SimpleNamespace
from utils.Traj_UNet_ori import *
from utils.config_ori import args
from utils.utils import *
from torch.utils.data import DataLoader


temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)

config.training.batch_size = batchsize = 2
unet = Guide_UNet(config).cuda()
# # load the model
# unet.load_state_dict(torch.load('./model.pt'))


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

# # load head information for guide trajectory generation
# batchsize = 500
# head = np.load('heads.npy',
#                    allow_pickle=True)
# head = torch.from_numpy(head).float()
# dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)


# # # the mean and std of head information, using for rescaling
# # # departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
# # hmean=[0, 10283.41600429,   961.66920921,   292.30299616,    36.02766493, 10.98568072]
# # hstd=[1, 8782.599246414231, 379.41939897358264, 107.24874828393955, 28.749924691281066, 8.774629812281198]
mean = np.array([104.07596303,   30.68085491])
std = np.array([2.15106194e-02, 1.89193207e-02])
# # # the original mean and std of trajectory length, using for rescaling the trajectory length
len_mean = 292.30299616  # Chengdu
len_std = 107.2487482839  # Chengdu

model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen200/models/10-04-08-01-11/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen200/models/10-04-08-01-11/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen200/models/10-04-08-01-11/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen200/models/10-04-08-01-11/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen300/models/10-04-08-03-54/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen300/models/10-04-08-03-54/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen300/models/10-04-08-03-54/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_interlen300/models/10-04-08-03-54/unet_200.pt",
] 
model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight05/models/10-04-07-59-31/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight05/models/10-04-07-59-31/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight05/models/10-04-07-59-31/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight05/models/10-04-07-59-31/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight10/models/10-04-07-59-39/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight10/models/10-04-07-59-39/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight10/models/10-04-07-59-39/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1004_ori_filterarea_weight10/models/10-04-07-59-39/unet_200.pt",
] 
filename='1004mtl_traj_ori_len.png'
head = np.array([[2.0000e+00, 6.2662e-02, 1.4033e-01, 1.0000e+00, 3.1331e-04, 1.5628e-01, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])

model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam/models/10-07-16-07-08/unet_10.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam/models/10-07-16-07-08/unet_30.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam/models/10-07-16-07-08/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam/models/10-07-16-07-08/unet_100.pt",
    # "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam/models/10-07-16-07-08/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam/models/10-07-16-07-08/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam1e4/models/10-07-16-07-45/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam1e4/models/10-07-16-07-45/unet_100.pt",
    # "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam1e4/models/10-07-16-07-45/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_adam1e4/models/10-07-16-07-45/unet_200.pt",
] 
filename='1008mtl_traj_adam.png'

model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart001/models/10-07-23-50-18/unet_10.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart001/models/10-07-23-50-18/unet_30.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart001/models/10-07-23-50-18/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart001/models/10-07-23-50-18/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart001/models/10-07-23-50-18/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart0001/models/10-07-23-50-03/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart0001/models/10-07-23-50-03/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_betastart0001/models/10-07-23-50-03/unet_200.pt",
] 
filename='1008mtl_traj_beta.png'

model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_ema099/models/10-07-23-49-01/unet_10.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_ema099/models/10-07-23-49-01/unet_30.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_ema099/models/10-07-23-49-01/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_ema099/models/10-07-23-49-01/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_ema099/models/10-07-23-49-01/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_noema/models/10-07-22-17-01/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_noema/models/10-07-22-17-01/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_noema/models/10-07-22-17-01/unet_200.pt",
] 
filename='1008mtl_traj_ema.png'

model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_300.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_400.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_500.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_750.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_epoch1000/models/10-07-23-48-18/unet_1000.pt",
] 
filename='1008mtl_traj_epoch.png'

model_dir_list=[
    # './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_nresblk1/models/10-07-23-51-44/unet_10.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_nresblk1/models/10-07-23-51-44/unet_30.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_nresblk1/models/10-07-23-51-44/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_nresblk1/models/10-07-23-51-44/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_nresblk1/models/10-07-23-51-44/unet_150.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_nresblk1/models/10-07-23-51-44/unet_200.pt",
] 
filename='1008mtl_traj_resblk.png'

model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_step100/models/10-07-16-09-50/unet_10.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_step100/models/10-07-16-09-50/unet_30.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_step100/models/10-07-16-09-50/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_step100/models/10-07-16-09-50/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1007_ori_filterarea_step100/models/10-07-16-09-50/unet_200.pt",
] 
filename='1008mtl_traj_step.png'


model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad/models/10-08-15-11-34/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad/models/10-08-15-11-34/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad/models/10-08-15-11-34/unet_300.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad/models/10-08-15-11-34/unet_500.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nointer/models/10-08-15-14-45/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nointer/models/10-08-15-14-45/unet_300.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nointer/models/10-08-15-14-45/unet_500.pt",
] 
filename='1009mtl_traj_nopad.png'



model_dir_list=[
    './model.pt',
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nocond/models/10-08-15-10-32/unet_50.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nocond/models/10-08-15-10-32/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nocond/models/10-08-15-10-32/unet_200.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nocond/models/10-08-15-10-32/unet_400.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nocond/models/10-08-15-10-32/unet_800.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_filterpad_nocond/models/10-08-15-10-32/unet_1600.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_nocond/models/10-08-14-56-44/unet_100.pt",
    "/home/yichen/DiffTraj/results/DiffTraj/1008_ori_filterarea_nocond/models/10-08-14-56-44/unet_500.pt",
] 
filename='1009mtl_traj_nocond.png'




# lengths=200
# head = np.load('heads.npy', allow_pickle=True)
# head = head[:2]


print(filename)
x0 = torch.randn(batchsize, 2, config.data.traj_length).cuda()
head = torch.from_numpy(head).float().cuda()



Gen_traj = []
Gen_head = []
# for i in tqdm(range(1)):
#     # head = next(iter(dataloader))
#     # lengths = head[:, 3]
#     # lengths = lengths * len_std + len_mean
#     # lengths = lengths.int()
#     # tes = head[:,:6].numpy()
#     # Gen_head.extend((tes*hstd+hmean))
#     # head = head.cuda()
#     # Start with random noise
#     x = torch.randn(batchsize, 2, config.data.traj_length).cuda()

for model_dir in model_dir_list:
    ckpt_dir = model_dir
    print(ckpt_dir)
    unet.load_state_dict(torch.load(ckpt_dir))#, strict=False)
    
    lengths = head[:, 3].cpu()
    lengths = lengths * 200
    lengths = lengths.int()
    # lengths = head[:, 3].cpu()
    # lengths = lengths * len_std + len_mean
    # lengths = lengths.int()

    ims = []
    n = x0.size(0)
    x = x0
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    # for j in range(batchsize):
    j=0
    new_traj = resample_trajectory(trajs[j].T, lengths[j])
    # new_traj = new_traj * std + mean
    # print(new_traj)
    lat_min,lat_max = (18.249901, 55.975593)
    lon_min,lon_max = (-122.3315333, 126.998528)
    new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
    new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
    # print(new_traj)
    Gen_traj.append(new_traj)


# try:
fig = plt.figure(figsize=(12,12))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    ax1 = fig.add_subplot(331+i)  
    ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# except:
#     pdb.set_trace()
plt.tight_layout()
plt.savefig(filename)
plt.show()

# plt.figure(figsize=(8,8))
# for i in range(len(Gen_traj)):
#     traj=Gen_traj[i]
#     plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# plt.tight_layout()
# plt.savefig('Chengdu_traj.png')
# plt.show()