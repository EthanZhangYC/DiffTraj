
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
    
    if config.data.interpolated:
        train_x_ori = dataset[1].squeeze(1)[:,:,2:]
    else:
        train_x_ori = dataset[0].squeeze(1)[:,:,2:]
    train_y_ori = dataset[3]
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    train_x_ori[pad_mask_source_train_ori] = 0.
    
    
    if config.data.filter_area:
        print('filtering area')
        train_x_ori,train_y_ori = filter_area(train_x_ori, train_y_ori, pad_mask_source_train_ori)
        pad_mask_source_train_ori = train_x_ori[:,:,2]==0

    if config.data.traj_length<train_x_ori.shape[1]:
        train_x_ori = train_x_ori[:,:config.data.traj_length,:]
        pad_mask_source_train_ori = pad_mask_source_train_ori[:,:config.data.traj_length]

    if "seid" in config.model.mode:
        sid,eid = generate_posid(train_x_ori, pad_mask_source_train_ori)
        se_id = np.stack([sid, eid]).T

    if config.data.unnormalize:
        print('unnormalizing data')
        minmax_list = [
            (18.249901, 55.975593), (-122.3315333, 126.998528), \
            (0.9999933186918497, 1198.999998648651),
            (0.0, 50118.17550774085),
            (0.0, 49.95356703911097),
            (-9.99348698095659, 9.958323482935628),
            (-39.64566646191948, 1433.3438889109589),
            (0.0, 359.95536847383516)
        ]
        for i in range(7):
            if i==2:
                continue
            train_x_ori[:,:,i] = train_x_ori[:,:,i] * (minmax_list[i][1]-minmax_list[i][0]) + minmax_list[i][0]
    
    if config.data.filter_nopad:
        print('filtering nopadding segments')
        pad_mask_source_incomplete = np.sum(pad_mask_source_train_ori,axis=1) == 0
        train_x_ori = train_x_ori[pad_mask_source_incomplete]
        train_y_ori = train_y_ori[pad_mask_source_incomplete]
        if "seid" in config.model.mode:
            se_id = se_id[pad_mask_source_incomplete]
        # np.sum(pad_mask_source_incomplete)
        
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

        
    # train_x_mtl_ori = train_x_mtl[:,:,2:] 
    # train_x_mtl_ori[pad_mask_target_train] = 0.
    if config.data.interpolated:
        train_x_mtl_ori = dataset_mtl[1].squeeze(1)[:,:,2:]
    else:
        train_x_mtl_ori = dataset_mtl[0].squeeze(1)[:,:,2:]
    pad_mask_target_train_ori = train_x_mtl_ori[:,:,2]==0
    train_x_mtl_ori[pad_mask_target_train_ori] = 0.
    train_y_mtl_ori = dataset_mtl[2]
    
    # if "seid" in config.model.mode:
    #     min_max = []
    #     sid,eid = generate_posid(train_x_mtl_ori, pad_mask_target_train_ori, min_max=min_max)
    #     se_id_mtl = np.stack([sid, eid]).T
    
    if config.data.unnormalize:
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
    
    if config.data.filter_nopad:
        pad_mask_target_incomplete = np.sum(pad_mask_target_train_ori,axis=1) == 0
        train_x_mtl_ori = train_x_mtl_ori[pad_mask_target_incomplete]
        train_y_mtl_ori = train_y_mtl_ori[pad_mask_target_incomplete]

    
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
        
    if "seid" in config.model.mode:
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
    else:
        train_dataset_ori = TensorDataset(
            torch.from_numpy(train_x_ori).to(torch.float),
            torch.from_numpy(train_y_ori)
        )
        train_dataset_mtl_ori = TensorDataset(
            torch.from_numpy(train_x_mtl_ori).to(torch.float),
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


input_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform_standard = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])
}


class create_single_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, trajs, part, transform):
        super(create_single_dataset, self).__init__()
        self.imgs = imgs
        self.trajs = trajs
        self.default_idx = range(len(imgs))
        self.label_dict={'0':0,'1':0,'2':1,'3':1,'4':1}
        self.transform = transform
        # self.dataset = dataset
        # if part=='train':                     
        #     random.shuffle(self.default_idx)
        
    def __getitem__(self, index):
        img_dir = self.imgs[self.default_idx[index]]
        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        traj,label = self.trajs[self.default_idx[index]]
        if len(traj) < 600:
            extra = [[0 for j in range(len(traj[0]))] for i in range(600 - len(traj))]
            traj = traj + extra
        traj = np.array(traj)

        # label = self.label_dict[img_dir.split('/')[-2]]
        return traj, img, label, img_dir
        
    def __len__(self):
        return len(self.imgs)
    
def load_data_img(config):
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
    
    if config.data.interpolated:
        train_x_ori = dataset[1].squeeze(1)[:,:,2:]
    else:
        train_x_ori = dataset[0].squeeze(1)[:,:,2:]
    train_y_ori = dataset[3]
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    train_x_ori[pad_mask_source_train_ori] = 0.
    
    if config.data.unnormalize:
        print('unnormalizing data')
        minmax_list = [
            (18.249901, 55.975593), (-122.3315333, 126.998528), \
            (0.9999933186918497, 1198.999998648651),
            (0.0, 50118.17550774085),
            (0.0, 49.95356703911097),
            (-9.99348698095659, 9.958323482935628),
            (-39.64566646191948, 1433.3438889109589),
            (0.0, 359.95536847383516)
        ]
        for i in range(7):
            if i==2:
                continue
            train_x_ori[:,:,i] = train_x_ori[:,:,i] * (minmax_list[i][1]-minmax_list[i][0]) + minmax_list[i][0]
    
    if config.data.filter_nopad:
        print('filtering nopadding segments')
        pad_mask_source_incomplete = np.sum(pad_mask_source_train_ori,axis=1) == 0
        train_x_ori = train_x_ori[pad_mask_source_incomplete]
        train_y_ori = train_y_ori[pad_mask_source_incomplete]
        # np.sum(pad_mask_source_incomplete)
        
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

        
    # train_x_mtl_ori = train_x_mtl[:,:,2:] 
    # train_x_mtl_ori[pad_mask_target_train] = 0.
    if config.data.interpolated:
        train_x_mtl_ori = dataset_mtl[1].squeeze(1)[:,:,2:]
    else:
        train_x_mtl_ori = dataset_mtl[0].squeeze(1)[:,:,2:]
    pad_mask_target_train_ori = train_x_mtl_ori[:,:,2]==0
    train_x_mtl_ori[pad_mask_target_train_ori] = 0.
    train_y_mtl_ori = dataset_mtl[2]
    
    
    if config.data.unnormalize:
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
    
    if config.data.filter_nopad:
        pad_mask_target_incomplete = np.sum(pad_mask_target_train_ori,axis=1) == 0
        train_x_mtl_ori = train_x_mtl_ori[pad_mask_target_incomplete]
        train_y_mtl_ori = train_y_mtl_ori[pad_mask_target_incomplete]

    
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
    
    train_dataset_ori = TensorDataset(
        torch.from_numpy(train_x_ori).to(torch.float),
        torch.from_numpy(train_y_ori)
    )
    train_dataset_mtl_ori = TensorDataset(
        torch.from_numpy(train_x_mtl_ori).to(torch.float),
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
    
    
    traj_init_filename = '/home/xieyuan/Traj2Image-10.05/datas/cnn_data/traj2image_6class_fixpixel_fixlat3_insert1s_train&test_cnn_0607.pickle'
    traj_init_filename = '/home/yichen/data/traj2image_6class_fixpixel_fixlat3_insert1s_train&test_cnn_0607.pickle'
    # traj_map13_filename = '/home/xieyuan/Traj2Image-10.05/datas/traj_to_map/traj2image_map_index.pickle'
    # traj_map6_filename = '/home/xieyuan/Traj2Image-10.05/datas/traj_to_map/traj2image_rescale_6class_imgs.pickle'

    # traj init dataset
    with open(traj_init_filename, "rb") as f:
        traj_dataset = pickle.load(f)
    train_init_traj, test_init_traj = traj_dataset

    # # traj img dataset
    # with open(traj_map13_filename, "rb") as f:
    #     traj_map13_dataset = pickle.load(f)
    # train_map13, _, train_index, test_map13, _, test_index = traj_map13_dataset

    # with open(traj_map6_filename, "rb") as f:
    #     traj_map6_dataset = pickle.load(f)
    # train_map6, test_map6 = traj_map6_dataset
    
    # pdb.set_trace()
    
    imgs_train = []
    img_dir = "/home/xieyuan/Traj2Image-10.05/datas/OpenStreetMap/global_map_tiles_satellite_zoom18_size50_train_size250/*.png"
    img_dir = "/home/yichen/data/global_map_tiles_satellite_zoom18_size50_test_size250/*.png"
    for file_name in glob.glob(img_dir, recursive=True):
        imgs_train.append(file_name)
    print('train:',len(imgs_train))#,'val:',len(tar_imgs_val))
    
    trainset_mix = create_single_dataset(
        imgs_train, 
        train_init_traj,
        transform=transform_standard['train'],
        part='train'
    )
    train_loader_source_mix = DataLoader(trainset_mix, batch_size=min(batch_sizes, len(trainset_mix)), num_workers=0, shuffle=True, drop_last=False)

        
    # trainset_concat =  ConcatDataset(trainset_tar, trainset_aux)
    # if args.tgt_ratio!=1:
    #     sampler = ImbalancedDatasetSampler(trainset_concat, callback_get_label=get_label_aux, num_samples=self.ori_len)
    # else:
    #     sampler = ImbalancedDatasetSampler(trainset_concat, callback_get_label=get_label_aux)
    # train_loader = torch.utils.data.DataLoader(
    #     trainset_concat,
    #     batch_size=args.batch_size, sampler=sampler,
    #     num_workers=args.n_workers, pin_memory=True
    # )    


    return train_source_iter, train_tgt_iter, test_loader, train_loader_target, train_loader_target_ori, train_loader_source_ori, train_loader_source_mix




# def load_data_traj600_map19(train_init_traj, test_init_traj, train_map_13channel, test_map_13channel, train_map_extra_6channel, test_map_extra_6channel):
#     train_merge_dataset = Dataset_init600_map19(train_init_traj, train_map_13channel, train_map_extra_6channel)
#     test_merge_dataset = Dataset_init600_map19(test_init_traj, test_map_13channel, test_map_extra_6channel)
#     return train_merge_dataset, test_merge_dataset


# class Dataset_init600_map19(Dataset):
#     def __init__(self, traj_init, map_13channel, map_6channel):
#         # traj init dataset
#         self.init_traj = traj_init

#         # map image dataset: 13 channel
#         map_13channel = np.array(map_13channel)
#         self.map_13channel = torch.Tensor(map_13channel)

#         # map extra img: 6 channel
#         map_extra_6channel = np.array(map_6channel)
#         self.map_extra_6channel = torch.Tensor(map_extra_6channel)

#         # sample length
#         self.n_samples = self.map_13channel.shape[0]

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, index):
#         map_img_sample = self.map_13channel[index]
#         map_extra_sample = self.map_extra_6channel[index]
#         label = self.init_traj[index][1]

#         new_traj_sample = []
#         if len(self.init_traj[index][0]) < 600:
#             extra = [[0 for j in range(len(self.init_traj[index][0][0]))] for i in range(600 - len(self.init_traj[index][0]))]
#             new_traj_sample = self.init_traj[index][0] + extra
#         else:
#             new_traj_sample = self.init_traj[index][0]
#         init_traj_sample = torch.tensor(new_traj_sample)

#         return init_traj_sample, map_img_sample, map_extra_sample, label
    
    
# 111
