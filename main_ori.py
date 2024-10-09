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
from utils.config_ori import args
from utils.EMA import EMAHelper
from utils.Traj_UNet_ori import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil


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

import glob
from torchvision import transforms
from PIL import Image
import argparse
# This code part from https://github.com/sunlin-ai/diffusion_tutorial


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_label(single_dataset,idx,label_dict):
    label = single_dataset[idx][1].item()
    return label

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        
        label_dict={'0':0,'1':0,'2':1,'3':1,'4':1}
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, label_dict)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        
        weights = [1.0 / label_to_count[self._get_label(dataset, idx, label_dict)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx, label_dict):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx, label_dict)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    # sample class balance training batch 
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def filter_area(trajs, labels, pad_masks):
    new_list=[]
    new_list_y=[]
    lat_min,lat_max = (18.249901, 55.975593)
    lon_min,lon_max = (-122.3315333, 126.998528)
    len_traj = trajs.shape[0]
    # avg_lat_list=[]
    # avg_lon_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        label = labels[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        # avg_lat_list.append(avg_lat)
        # avg_lon_list.append(avg_lon)
        if avg_lat<41 and avg_lat>39:
            if avg_lon>115 and avg_lon<117:
                new_list.append(traj)
                new_list_y.append(label)
                
    return np.array(new_list), np.array(new_list_y)

def generate_posid(trajs, pad_masks, min_max=[(18.249901, 55.975593),(-122.3315333, 126.998528)]):
    lat_min,lat_max = min_max[0]
    lon_min,lon_max = min_max[1]
    
    new_list=[]
    new_list_y=[]
    len_traj = trajs.shape[0]
    
    max_list_lat=[]
    max_list_lon=[]
    min_list_lat=[]
    min_list_lon=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        tmp_max_lat,tmp_max_lon = np.max(new_traj, axis=0)
        tmp_min_lat,tmp_min_lon = np.min(new_traj, axis=0)
        max_list_lat.append(tmp_max_lat)
        max_list_lon.append(tmp_max_lon)
        min_list_lat.append(tmp_min_lat)
        min_list_lon.append(tmp_min_lon)
    
    tmp_max_lat = np.max(np.array(max_list_lat))+1e-6
    tmp_max_lon = np.max(np.array(max_list_lon))+1e-6
    tmp_min_lat = np.min(np.array(min_list_lat))-1e-6
    tmp_min_lon = np.min(np.array(min_list_lon))-1e-6
    print(tmp_max_lat,tmp_max_lon,tmp_min_lat,tmp_min_lon)
        
    # tmp_max_lat,tmp_max_lon,tmp_min_lat,tmp_min_lon = 40.8855, 117.2707, 38.44578, 114.93135
    patchlen_lat = (tmp_max_lat-tmp_min_lat) / 16
    patchlen_lon = (tmp_max_lon-tmp_min_lon) / 16
    sid_list=[]
    eid_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        # label = labels[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        # avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        sid = (new_traj[0,0]-tmp_min_lat)//patchlen_lat*16+(new_traj[0,1]-tmp_min_lon)//patchlen_lon
        eid = (new_traj[-1,0]-tmp_min_lat)//patchlen_lat*16+(new_traj[-1,1]-tmp_min_lon)//patchlen_lon
        sid_list.append(sid)
        eid_list.append(eid)
        # if sid>=256 or eid>=256:
        #     pdb.set_trace()

    return np.array(sid_list), np.array(eid_list)

def load_data(config):
    batch_sizes = config.training.batch_size
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'
    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    # test_x_geolife = dataset[5].squeeze(1)
    # test_y_geolife = dataset[7]
    # test_x_geolife = test_x_geolife[:,:,4:]   
    # pad_mask_source_test = test_x_geolife[:,:,0]==0
    # test_x_geolife[pad_mask_source_test] = 0.
    
    train_x = dataset[1].squeeze(1)
    train_y = dataset[3]
    train_x = train_x[:,:,4:]   
    pad_mask_source = train_x[:,:,0]==0
    train_x[pad_mask_source] = 0.
    
    # if config.data.interpolated:
    train_x_ori = dataset[1].squeeze(1)[:,:,2:]
    # else:
    # train_x_ori = dataset[0].squeeze(1)[:,:,2:]
    train_y_ori = dataset[3]
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    train_x_ori[pad_mask_source_train_ori] = 0.
    
    
    # if config.data.filter_area:
    print('filtering area')
    train_x_ori,train_y_ori = filter_area(train_x_ori, train_y_ori, pad_mask_source_train_ori)
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0

    if config.data.traj_length < train_x_ori.shape[1]:
        train_x_ori = train_x_ori[:,:config.data.traj_length,:]
        pad_mask_source_train_ori = pad_mask_source_train_ori[:,:config.data.traj_length]

    # if "seid" in config.model.mode:
    sid,eid = generate_posid(train_x_ori, pad_mask_source_train_ori)
    se_id = np.stack([sid, eid]).T
    
    print('filtering nopadding segments')
    pad_mask_source_incomplete = np.sum(pad_mask_source_train_ori,axis=1) == 0
    train_x_ori = train_x_ori[pad_mask_source_incomplete]
    train_y_ori = train_y_ori[pad_mask_source_incomplete]
    se_id = se_id[pad_mask_source_incomplete]
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))



    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_0817_sharedminmax_balanced.pickle'
    print(filename_mtl)
    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    train_x_mtl = dataset_mtl[1].squeeze(1)
    test_x = dataset_mtl[4].squeeze(1)
    train_y_mtl = dataset_mtl[2]
    test_y = dataset_mtl[5]


    # if config.data.interpolated:
    train_x_mtl_ori = dataset_mtl[1].squeeze(1)[:,:,2:]
    # else:
    # train_x_mtl_ori = dataset_mtl[0].squeeze(1)[:,:,2:]
    pad_mask_target_train_ori = train_x_mtl_ori[:,:,2]==0
    train_x_mtl_ori[pad_mask_target_train_ori] = 0.
    train_y_mtl_ori = dataset_mtl[2]
    
    # if "seid" in config.model.mode:
    #     min_max = []
    #     sid,eid = generate_posid(train_x_mtl_ori, pad_mask_target_train_ori, min_max=min_max)
    #     se_id_mtl = np.stack([sid, eid]).T

    train_x_mtl = train_x_mtl[:,:,4:]
    test_x = test_x[:,:,4:]
    
    pad_mask_target_train = train_x_mtl[:,:,0]==0
    pad_mask_target_test = test_x[:,:,0]==0
    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    # logger.info('Total shape: '+str(train_data.shape))
    print('GeoLife shape: '+str(train_x_ori.shape))
    print('MTL shape: '+str(train_x_mtl_ori.shape))

    
    n_geolife = train_x.shape[0]
    n_mtl = train_x_mtl.shape[0]
    train_dataset_geolife = TensorDataset(
        torch.from_numpy(train_x).to(torch.float),
        torch.from_numpy(train_y),
        torch.from_numpy(np.array([0]*n_geolife)).float()
    )
    train_dataset_mtl = TensorDataset(
        torch.from_numpy(train_x_mtl).to(torch.float),
        torch.from_numpy(train_y_mtl), # add label for debug
        torch.from_numpy(np.array([1]*n_mtl)).float(),
        torch.from_numpy(np.arange(n_mtl))
    )


    sampler = ImbalancedDatasetSampler(train_dataset_geolife, callback_get_label=get_label, num_samples=len(train_dataset_mtl))
    train_loader_source = DataLoader(train_dataset_geolife, batch_size=min(batch_sizes, len(train_dataset_geolife)), sampler=sampler, num_workers=8, shuffle=False, drop_last=True)
    train_loader_target = DataLoader(train_dataset_mtl, batch_size=min(batch_sizes, len(train_dataset_mtl)), num_workers=8, shuffle=True, drop_last=False)
    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    train_loader = (train_source_iter, train_tgt_iter)
    
        
    # if config.data.traj_length<train_x_ori.shape[1]:
    #     train_x_ori = train_x_ori[:,:config.data.traj_length,:]
    #     train_x_mtl_ori = train_x_mtl_ori[:,:config.data.traj_length,:]
        
    train_dataset_ori = TensorDataset(
        torch.from_numpy(train_x_ori).to(torch.float),
        torch.from_numpy(se_id).to(torch.float),
        torch.from_numpy(train_y_ori)
    )
    train_dataset_mtl_ori = TensorDataset(
        torch.from_numpy(train_x_mtl_ori).to(torch.float),
        # torch.from_numpy(se_id_mtl).to(torch.float),
        torch.from_numpy(train_y_mtl_ori)
    )
    train_loader_source_ori = DataLoader(train_dataset_ori, batch_size=min(batch_sizes, len(train_dataset_geolife)), num_workers=0, shuffle=True, drop_last=False)
    train_loader_target_ori = DataLoader(train_dataset_mtl_ori, batch_size=min(batch_sizes, len(train_dataset_mtl)), num_workers=0, shuffle=False, drop_last=False)
    # train_loader_target_ori=train_loader_source_ori=None
    
    test_dataset = TensorDataset(
        torch.from_numpy(test_x).to(torch.float),
        torch.from_numpy(test_y),
    )
    test_loader = DataLoader(test_dataset, batch_size=min(batch_sizes, len(test_dataset)))

    train_source_iter=train_tgt_iter=test_loader=train_loader_target=None
    return train_source_iter, train_tgt_iter, test_loader, train_loader_target, train_loader_target_ori, train_loader_source_ori


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def main(config, logger, exp_dir):

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
    _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(config)
    dataloader = train_loader_source_ori

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    # model_save = exp_dir / 'models' / (timestamp + '/')
    model_save = exp_dir + '/models/' + (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        # for _, (trainx, head) in enumerate(dataloader):
        #     x0 = trainx.cuda()
        #     head = head.cuda()
        for _, batch_data in enumerate(dataloader):
            x0 = batch_data[0][:,:,:2].cuda() 
            x0 = x0.permute(0,2,1)
            
            
            # pdb.set_trace()
            # Gen_traj=[]
            # all_pad_mask = batch_data[0][:,:,2]!=0
            # for i in range(9):
            #     pad_mask = all_pad_mask[i]
            #     new_traj = batch_data[0][i,:,:2][pad_mask]
            #     lat_min,lat_max = (18.249901, 55.975593)
            #     lon_min,lon_max = (-122.3315333, 126.998528)
            #     new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
            #     new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
            #     Gen_traj.append(new_traj)
            #     print(i)
            #     print(new_traj)
            #     print('---------')
            # fig = plt.figure(figsize=(12,12))
            # filename = '1008_test.png'
            # for i in range(len(Gen_traj)):
            #     traj=Gen_traj[i]
            #     ax1 = fig.add_subplot(331+i)  
            #     ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
            # plt.tight_layout()
            # plt.savefig(filename)
            # plt.show()
            # exit()
            
    
            label = batch_data[-1].unsqueeze(1)
            
            sid = batch_data[1][:,0].unsqueeze(1)
            eid = batch_data[1][:,1].unsqueeze(1)
            
            trip_len = torch.sum(batch_data[0][:,:,2]!=0, dim=1).unsqueeze(1)
            max_feat = torch.max(batch_data[0][:,:,4:8], dim=1)[0] # v, a, j, br
            avg_feat = torch.sum(batch_data[0][:,:,3:8], dim=1) / (trip_len+1e-6)
            total_dist = torch.sum(batch_data[0][:,:,3], dim=1).unsqueeze(1)
            total_time = torch.sum(batch_data[0][:,:,2], dim=1).unsqueeze(1)
            avg_dist = avg_feat[:,0].unsqueeze(1)
            avg_speed = avg_feat[:,1].unsqueeze(1)
            
            trip_len = trip_len / config.data.traj_length
            total_time = total_time / 3000.
            head = torch.cat([label, total_dist, total_time, trip_len, avg_dist, avg_speed, sid, eid],dim=1).cuda()

            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, )).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
                
        if (epoch) % 1000 == 0:
            m_path = model_save + f"/unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir + '/results/' + f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))
            # m_path = model_save / f"unet_{epoch}.pt"
            # torch.save(unet.state_dict(), m_path)
            # m_path = exp_dir / 'results' / f"loss_{epoch}.npy"
            # np.save(m_path, np.array(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
    parser.add_argument('--job_name', type=str, default='test', help='activation')
    # parser.add_argument('--epoch', type=int, default=200, help='activation')
    # parser.add_argument('--n_step', type=int, default=500, help='activation')
    # parser.add_argument('--batch_size', type=int, default=1024, help='activation')
    # parser.add_argument('--lr', type=float, default=2e-4, help='activation')
    # parser.add_argument('--mode', type=str, default='label_only', help='activation')
    # parser.add_argument('--filter_nopad', action='store_true', help='whether to output attention in encoder')
    # parser.add_argument('--filter_area', action='store_true', help='whether to output attention in encoder')
    # parser.add_argument('--unnormalize', action='store_true', help='whether to output attention in encoder')
    # parser.add_argument('--interpolated', action='store_true', help='whether to output attention in encoder')
    # parser.add_argument('--guidance_scale', type=float, default=3., help='activation')
    # parser.add_argument('--loss', type=str, default='mse', help='activation')
    # parser.add_argument('--model', type=str, default='unet', help='activation')
    # parser.add_argument('--traj_len', type=int, default=650, help='activation')

    tmp_args = parser.parse_args()
    torch.set_num_threads(8)
    
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    # root_dir = Path(__name__).resolve().parents[0]
    # result_name = '{}_steps={}_len={}_{}_bs={}'.format(
    #     config.data.dataset, config.diffusion.num_diffusion_timesteps,
    #     config.data.traj_length, config.diffusion.beta_end,
    #     config.training.batch_size)
    # exp_dir = root_dir / "DiffTraj" / result_name
    result_name = tmp_args.job_name
    root_dir = "/home/yichen/DiffTraj/results"
    exp_dir = root_dir + "/DiffTraj/" + result_name
    
    for d in ["/results", "/models", "/logs","/Files"]:
        os.makedirs(exp_dir + d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir + '/Files/' + (timestamp + '/')
    # files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy('./utils/config_ori.py', files_save)
    shutil.copy('./utils/Traj_UNet.py', files_save)

    logger = Logger(
        __name__,
        log_path=exp_dir + "/logs/" + (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)
    main(config, logger, exp_dir)