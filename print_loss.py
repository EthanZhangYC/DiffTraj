
import numpy as np
import pdb

# for i in range(1,21):
#     base_dir = '/home/yichen/DiffTraj/results/DiffTraj/0911_label_avgmax_vajb/results/loss_%d.npy'%(i*10)
base_dir = '/home/yichen/DiffTraj/results/DiffTraj/1009_ori_filterarea_filterpad_epoch4w/results/loss_31000.npy'
base_dir = '/home/yichen/DiffTraj/results/DiffTraj/1014_ori_filterarea_filterpad_filterclass0/results/loss_1000.npy'
losses = np.load(base_dir)   
# print(i*10)
print(losses)

import matplotlib.pyplot as plt
filename='loss.png'
fig = plt.figure()
# for i in range(len(Gen_traj)):
#     traj=Gen_traj[i]
#     # ax1 = fig.add_subplot(331+i)  
n = losses.shape[0]
pdb.set_trace()
plt.plot(np.arange(n),losses,color='blue',alpha=0.1)
plt.tight_layout()
plt.savefig(filename)
plt.show()
