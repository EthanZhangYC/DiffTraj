
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

import pdb

import sys
import os.path as osp
import time
import numpy as np
import pickle
import pdb

import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



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
 


# config.training.batch_size = batchsize = 1

# Gen_traj = []
# Gen_head = []
# _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(config)
# # batch_data = next(iter(train_loader_target_ori))
# # x0 = batch_data[0][:,:,:2] #.cuda() 
# # Gen_traj.append(x0[0])
# # # Gen_traj.append(x0[0])
# # x0 = x0.permute(0,2,1)

filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_0817_sharedminmax_balanced.pickle'
print(filename_mtl)
with open(filename_mtl, 'rb') as f:
    kfold_dataset, X_unlabeled_mtl = pickle.load(f)
dataset_mtl = kfold_dataset

# train_x_mtl = dataset_mtl[1].squeeze(1)
# test_x = dataset_mtl[4].squeeze(1)
# train_y_mtl = dataset_mtl[2]
# test_y = dataset_mtl[5]


# if config.data.interpolated:
#     train_x_mtl_ori = dataset_mtl[1].squeeze(1)[:,:,2:]
# else:
train_x_mtl_ori = dataset_mtl[0].squeeze(1)[:,:,2:]
pad_mask_target_train_ori = train_x_mtl_ori[:,:,2]==0
train_x_mtl_ori[pad_mask_target_train_ori] = 0.
train_y_mtl_ori = dataset_mtl[2]

minmax_list=[
    (45.230416, 45.9997262293), (-74.31479102, -72.81248199999999),  \
    (0.9999933186918497, 1198.999998648651), # time
    (0.0, 50118.17550774085), # dist
    (0.0, 49.95356703911097), # speed
    (-9.99348698095659, 9.958323482935628), #acc
    (-39.64566646191948, 1433.3438889109589), #jerk
    (0.0, 359.95536847383516) #bearing
] 
for i in range(7):
    if i==2:
        continue
    train_x_mtl_ori[:,:,i] = train_x_mtl_ori[:,:,i] * (minmax_list[i][1]-minmax_list[i][0]) + minmax_list[i][0]

# train_x_mtl = train_x_mtl[:,:,4:]
# test_x = test_x[:,:,4:]

# pad_mask_target_train = train_x_mtl[:,:,0]==0
# pad_mask_target_test = test_x[:,:,0]==0
# train_x_mtl[pad_mask_target_train] = 0.
# test_x[pad_mask_target_test] = 0.

result_list=[]
for idx,traj in enumerate(train_x_mtl_ori):
    # if idx>10:
    #     break
    pad_mask = traj[:,2]!=0
    traj = traj[pad_mask]
    lat_lon = traj[:,:2]
    max_lat,max_lon = np.max(lat_lon,axis=0)
    min_lat,min_lon = np.min(lat_lon,axis=0)
    msg = "%d, %.10f, %.10f, %.10f, %.10f\n"%(idx,max_lat,min_lat,max_lon,min_lon)
    print(msg)
    result_list.append(msg)
    
with open('mtl_minmax_latlon.txt', "w") as f:
    
    for result in result_list:
        f.writelines(result)



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
    
# _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(config)


    
# # resample the trajectory length
# # for j in range(batchsize):
# for j in range(1):
#     new_traj = resample_trajectory(trajs[j].T, lengths)
#     # new_traj = resample_trajectory(trajs[j].T, lengths[j])
#     new_traj = new_traj * std + mean
    
#     # # lat_min,lat_max = (45.230416, 45.9997262293)
#     # # lon_min,lon_max = (-74.31479102, -72.81248199999999)
#     # lat_min,lat_max = (18.249901, 55.975593)
#     # lon_min,lon_max = (-122.3315333, 126.998528)
    
#     print(new_traj)
    
#     # new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
#     # new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
#     Gen_traj.append(new_traj)
# break

