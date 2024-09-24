
import numpy as np


for i in range(1,21):
    base_dir = '/home/yichen/DiffTraj/results/DiffTraj/0911_label_avgmax_vajb/results/loss_%d.npy'%(i*10)
    array = np.load(base_dir)   
    print(i*10)
    print(array)