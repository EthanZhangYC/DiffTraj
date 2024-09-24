
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
# from utils.Traj_UNet import *
import argparse
from data_utils import load_data


import math
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F
import pdb

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # apply softmax along the attributes dimension
        weights = F.softmax(weights, dim=1)
        return weights

    
class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, mode=''):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        self.sid_embedding = nn.Embedding(257, hidden_dim)
        self.eid_embedding = nn.Embedding(257, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        # Continuous attributes
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(
        ), attr[:, 6].long(), attr[:, 7].long()

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        categorical_embed = torch.cat(
            (depature_embed, sid_embed, eid_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed
    
    
    
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(
            config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.traj_length
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None):
        assert x.shape[2] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)]
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(i_level, i_block, h.shape)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # print(hs[-1].shape)
        # print(len(hs))
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(h.shape)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                # print(i_level, i_block, h.shape)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Guide_UNet(nn.Module):
    def __init__(self, config):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale
        self.unet = Model(config)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = WideAndDeep(self.ch, mode=config.model.mode)
        self.place_emb = WideAndDeep(self.ch, mode=config.model.mode)

    def forward(self, x, t, attr):
        guide_emb = self.guide_emb(attr)
        place_vector = torch.zeros(attr.shape, device=attr.device)
        place_emb = self.place_emb(place_vector)
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise





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



cnt=0

ckpt_dir = "/home/yichen/DiffTraj/model.pt"
unet.load_state_dict(torch.load(ckpt_dir))#, strict=False)



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
    
# departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
head = np.array([1,1,1,1,1,1,1])
head = np.array([7e-2, 5.05e-1, 2.6863e-2, 23.61, 7e-2, 5.05e-1, 2.6863e-2, 23.61, 0])   
head = np.array([1.4630e+00, 6.2562e-04, 5.6896e-03, 2.3672e+01, 4.5764e+00, 1.2705e+00,
        1.3502e+00, 2.7067e+02, 3])   
head = np.array([0.2, 0.06, 0.0055, 23, 0.1, 0.1, 0.1, 0, 0])   
head = np.array([ 1.5500e+02, -5.0322e-01,  4.3233e+00, -6.8094e-02, -5.3749e-01, -9.9515e-01,  21,  21])
head = np.array([ 155, -5.0322e-01,  4.3233e+00, -6.8094e-02, -5.3749e-01, -9.9515e-01,  0,  21])
head = np.tile(head,[5,1])
head = torch.from_numpy(head).float()

# # # load head information for guide trajectory generation
# batchsize = 500
# head = np.load('heads.npy', allow_pickle=True)
# head = torch.from_numpy(head).float()
# pdb.set_trace()
# head = [ 1.5500e+02, -5.0322e-01,  4.3233e+00, -6.8094e-02, -5.3749e-01,
#         -9.9515e-01,  2.1000e+01,  2.1000e+01]
dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)


# _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(config)


# the mean and std of head information, using for rescaling
# departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
hmean=[0, 10283.41600429,   961.66920921,   292.30299616,    36.02766493, 10.98568072]
hstd=[1, 8782.599246414231, 379.41939897358264, 107.24874828393955, 28.749924691281066, 8.774629812281198]
mean = np.array([104.07596303,   30.68085491])
std = np.array([2.15106194e-02, 1.89193207e-02])
# the original mean and std of trajectory length, using for rescaling the trajectory length
len_mean = 292.30299616  # Chengdu
len_std = 107.2487482839  # Chengdu

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
    # # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length)
    
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
            # x, noise = q_xt_x0(x0, t)
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
        new_traj = new_traj * std + mean
        
        # # lat_min,lat_max = (45.230416, 45.9997262293)
        # # lon_min,lon_max = (-74.31479102, -72.81248199999999)
        # lat_min,lat_max = (18.249901, 55.975593)
        # lon_min,lon_max = (-122.3315333, 126.998528)
        
        print(new_traj)
        
        # new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        # new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        Gen_traj.append(new_traj)
    break

# plt.figure(figsize=(8,8))
# traj=x0.permute(0,2,1)[0]
# pdb.set_trace()
# plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# plt.tight_layout()
# plt.savefig('geolife_traj.png')
# exit()




# Gen_traj = []
# Gen_head = []
# for i in tqdm(range(1)):
#     head = next(iter(dataloader))
#     lengths = head[:, 3]
#     lengths = lengths * len_std + len_mean
#     lengths = lengths.int()
#     tes = head[:,:6].numpy()
#     Gen_head.extend((tes*hstd+hmean))
#     head = head.cuda()
#     # Start with random noise
#     x = torch.randn(batchsize, 2, config.data.traj_length).cuda()
#     ims = []
#     n = x.size(0)
#     seq_next = [-1] + list(seq[:-1])
#     for i, j in zip(reversed(seq), reversed(seq_next)):
#         t = (torch.ones(n) * i).to(x.device)
#         next_t = (torch.ones(n) * j).to(x.device)
#         with torch.no_grad():
#             pred_noise = unet(x, t, head)
#             # print(pred_noise.shape)
#             x = p_xt(x, pred_noise, t, next_t, beta, eta)
#             if i % 10 == 0:
#                 ims.append(x.cpu().squeeze(0))
#     trajs = ims[-1].cpu().numpy()
#     trajs = trajs[:,:2,:]
#     # resample the trajectory length
#     for j in range(batchsize):
#         new_traj = resample_trajectory(trajs[j].T, lengths[j])
#         new_traj = new_traj * std + mean
#         Gen_traj.append(new_traj)
#     break

plt.figure(figsize=(8,8))
# for i in range(len(Gen_traj)):
traj=Gen_traj[-1]
plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
plt.tight_layout()
plt.savefig('mtl_traj_ori.png')
plt.show()

# fig = plt.figure(figsize=(12,8))
# for i in range(len(Gen_traj)):
#     traj=Gen_traj[i]
#     ax1 = fig.add_subplot(231+i)  
#     ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
#     # # plt.plot(traj[:,0],traj[:,1],'o',color='black',alpha=0.1)
#     # plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# plt.tight_layout()
# plt.savefig('mtl_traj_ori.png')
# # print(new_traj)
# plt.show()