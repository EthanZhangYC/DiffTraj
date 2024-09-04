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

# This code part from https://github.com/sunlin-ai/diffusion_tutorial


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


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

def load_data(batch_sizes=1024):
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
    
    train_x_ori = dataset[0].squeeze(1)[:,:,2:]
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    train_x_ori[pad_mask_source_train_ori] = 0.
        
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
    train_x_mtl_ori = dataset_mtl[0].squeeze(1)[:,:,2:]
    pad_mask_target_train_ori = train_x_mtl_ori[:,:,2]==0
    train_x_mtl_ori[pad_mask_target_train_ori] = 0.
    
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
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))
    
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
        torch.from_numpy(train_y)
    )
    train_dataset_mtl_ori = TensorDataset(
        torch.from_numpy(train_x_mtl_ori).to(torch.float),
        torch.from_numpy(train_y_mtl)
    )
    train_loader_source_ori = DataLoader(train_dataset_ori, batch_size=min(batch_sizes, len(train_dataset_geolife)), num_workers=0, shuffle=False, drop_last=False)
    train_loader_target_ori = DataLoader(train_dataset_mtl_ori, batch_size=min(batch_sizes, len(train_dataset_mtl)), num_workers=0, shuffle=False, drop_last=False)

    
    test_dataset = TensorDataset(
        torch.from_numpy(test_x).to(torch.float),
        torch.from_numpy(test_y),
    )
    test_loader = DataLoader(test_dataset, batch_size=min(batch_sizes, len(test_dataset)))

    return train_source_iter, train_tgt_iter, test_loader, train_loader_target, train_loader_target_ori, train_loader_source_ori



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
    _,_,_,train_loader_target,train_loader_target_ori,train_loader_source_ori = load_data(args['training']['batch_size'])
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
    model_save = exp_dir + '/models/' + (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        for _, batch_data in enumerate(dataloader):
            x0 = batch_data[0][:,:,:2].cuda() 
            label = batch_data[1].unsqueeze(1)
            head = torch.sum(batch_data[0][:,:,2:8],dim=1) / (torch.sum(batch_data[0][:,:,2]!=0, dim=1)+1e-6).unsqueeze(1)
            head = torch.cat([head,label],dim=1)
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
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        if (epoch) % 10 == 0:
            m_path = model_save + f"/unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir + '/results/' + f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
    # # dataset parameters
    # parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    # parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    # parser.add_argument('--train-resizing', type=str, default='default')
    # parser.add_argument('--val-resizing', type=str, default='default')
    # parser.add_argument('--resize-size', type=int, default=224,
    #                     help='the image size after resizing')
    # parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
    #                     help='Random resize scale (default: 0.08 1.0)')
    # parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
    #                     help='Random resize aspect ratio (default: 0.75 1.33)')
    # parser.add_argument('--no-hflip', action='store_true',
    #                     help='no random horizontal flipping during training')
    # parser.add_argument('--norm-mean', type=float, nargs='+',
    #                     default=(0.485, 0.456, 0.406), help='normalization mean')
    # parser.add_argument('--norm-std', type=float, nargs='+',
    #                     default=(0.229, 0.224, 0.225), help='normalization std')
    # # model parameters
    # parser.add_argument('--bottleneck-dim', default=1024, type=int)
    # parser.add_argument('--no-pool', action='store_true',
    #                     help='no pool layer after the feature extractor.')
    # parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # parser.add_argument('--trade-off', default=1., type=float,
    #                     help='the trade-off hyper-parameter for transfer loss')
    # parser.add_argument('--trade-off-entropy', default=0.01, type=float,
    #                     help='the trade-off hyper-parameter for entropy loss')
    # parser.add_argument('--num-k', type=int, default=4, metavar='K',
    #                     help='how many steps to repeat the generator update')
    # # training parameters
    # parser.add_argument('-b', '--batch-size', default=32, type=int,
    #                     metavar='N',
    #                     help='mini-batch size (default: 32)')
    # parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
    #                     metavar='LR', help='initial learning rate', dest='lr')
    # parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
    #                 metavar='W', help='weight decay (default: 1e-3)',
    #                 dest='weight_decay')
    # parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
    #                     help='number of data loading workers (default: 2)')
    # parser.add_argument('--epochs', default=20, type=int, metavar='N',
    #                     help='number of total epochs to run')
    # parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
    #                     help='Number of iterations per epoch')
    # parser.add_argument('-p', '--print-freq', default=100, type=int,
    #                     metavar='N', help='print frequency (default: 100)')
    # parser.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    # parser.add_argument('--per-class-eval', action='store_true',
    #                     help='whether output per-class accuracy during evaluation')
    # parser.add_argument("--log", type=str, default='mcd',
    #                     help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
    #                     help="When phase is 'test', only test the model."
    #                          "When phase is 'analysis', only analysis the model.")
    
    # parser.add_argument('--use_unlabel', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--interpolated', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--interpolatedlinear', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--trip_time', type=int, default=20, help='')
    
    # parser.add_argument("--cat_mode", type=str, default='cat', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument("--loss_mode", type=str, default='srcce_tgtce_tgtent', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument("--pseudo_mode", type=str, default='threshold', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument('--pseudo_thres', default=0.95, type=float, help='initial learning rate')
    # parser.add_argument('--pseudo_ratio', default=0.666, type=float, help='initial learning rate')
    # parser.add_argument('--nbr_dist_thres', default=30, type=int, help='initial learning rate')
    # parser.add_argument('--nbr_limit', default=10, type=int, help='initial learning rate')
    # parser.add_argument('--trade-off-pseudo', default=1., type=float)
    # parser.add_argument('--trade-off-consis', default=1., type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--mean_tea', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--num_head', default=2, type=int, help='initial learning rate')
    # parser.add_argument("--nbr_data_mode", type=str, default='mergemin5', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument("--nbr_mode", type=str, default='perpt_cat', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument('--nbr_grad', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--nbr_pseudo', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument("--nbr_label_mode", type=str, default='', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument('--nbr_label_embed_dim', default=16, type=int, help='initial learning rate')
    # parser.add_argument('--pseudo_every_epoch', action="store_true", help='Whether to perform evaluation after training')

    # parser.add_argument('--random_mask_nbr_ratio', default=1.0, type=float)
    # parser.add_argument('--mask_early', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--mask_late', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--n_mask_late', default=5, type=int, help='initial learning rate')
    
    # parser.add_argument('--bert_out_dim', default=64, type=int, help='initial learning rate')
    # parser.add_argument('--G_no_frozen', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--token_len', default=0, type=int, help='initial learning rate')
    # parser.add_argument('--token_max_len', default=60, type=int, help='initial learning rate')
    # parser.add_argument('--prompt_id', default=5, type=int, help='initial learning rate')

    # parser.add_argument('--proto_momentum', default=0.9, type=float)
    # parser.add_argument("--update_strategy", type=str, default='iter', help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument('--self_train', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--semi', action="store_true", help='Whether to perform evaluation after training')
    
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    # parser.add_argument('--c_out', type=int, default=7, help='output size')
    # parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    # parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    # parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # parser.add_argument('--embed', type=str, default='timeF',
    #                     help='time features encoding, options:[timeF, fixed, learned]')
    # parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    # parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    # parser.add_argument('--stride', type=int, default=8, help='stride')
    # parser.add_argument('--prompt_domain', type=int, default=0, help='')
    # parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    # parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--job_name', type=str, default='test', help='activation')
    tmp_args = parser.parse_args()
    config.training.job_name = tmp_args.job_name
    
    print(args)
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

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
    main(config, logger, exp_dir, args)
