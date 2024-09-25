
--mode label_only \

--mode label_max_vajb \


CUDA_VISIBLE_DEVICES=6 \
python main.py \
--mode label_avgmax_vajb \
--epoch 500 \
--batch_size 1 \
--job_name 0911_label_avgmax_vajb_epoch500_padmask_bs1



--mode label_avgmax_vajb \
--lr 5e-5 \
--n_step 2000 \
--epoch 500 \
--batch_size 512 \
--filter_nopad \
--unnormalize \
--guidance_scale 0 \
--loss rmse \

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--mode label_oridiff \
--epoch 500 \
--batch_size 512 \
--filter_nopad \
--job_name 0925_label_oridiff_epoch500_nopad_bs512_shuffle_noema










CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=2 \
python main.py \
--mode label_oridiff \
--epoch 200 \
--lr 5e-5 \
--batch_size 128 \
--filter_nopad \
--job_name test

--unnormalize \






--resume /home/yichen/DiffTraj/results/DiffTraj/0911_label_max_vajb_epoch500/models/09-11-16-13-28/unet_500.pt

CUDA_VISIBLE_DEVICES=7 \
python traj_generate.py \
--job_name generate_test \
--mode label_max_vajb \
--resume 

CUDA_VISIBLE_DEVICES=7 \
python traj_generate.py \
--job_name generate_test \
--mode label_avgmax_vajb \
--resume /home/yichen/DiffTraj/results/DiffTraj/0912_label_avgmax_vajb_epoch1000_nopad_unnorm/models/09-12-13-37-06/unet_1000.pt

--resume /home/yichen/DiffTraj/results/DiffTraj/0911_label_avgmax_vajb_epoch500_nopad/models/09-11-20-22-39/unet_500.pt
--resume /home/yichen/DiffTraj/results/DiffTraj/0911_label_avgmax_vajb_epoch500_nopad/models/09-11-20-24-19/unet_500.pt




CUDA_VISIBLE_DEVICES=7 \
python traj_generate.py \
--job_name generate_test \
--mode label_avgmax_vajb \
--resume /home/yichen/DiffTraj/model.pt




CUDA_VISIBLE_DEVICES=6 \
python traj_generate.py \
--job_name generate_test \
--mode label_max_vajb \
--interpolated \
--resume results/DiffTraj/0916_label_max_vajb_epoch500_nopad_unnorm_bs512_nstep50/models/09-16-17-35-29/unet_500.pt



CUDA_VISIBLE_DEVICES=6 \
python traj_generate_ori.py \
--job_name generate_test \
--mode label_oridiff \
--resume /home/yichen/DiffTraj/results/DiffTraj/0924_label_oridiff_epoch500_nopad_bs512/models/09-24-22-05-47/unet_500.pt