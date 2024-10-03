
--mode label_only \
--mode label_max_vajb \
--mode label_avgmax_vajb \
--mode label_oridiff \
--mode label_oridiff_img \
--mode label_oridiff_normlentime_seid \
--lr 5e-5 \
--n_step 2000 \
--epoch 500 \
--batch_size 512 \
--filter_nopad \
--unnormalize \
--guidance_scale 0 \
--loss rmse \
--filter_area \
--model unet_nocond \

CUDA_VISIBLE_DEVICES=2 \
python main.py \
--mode label_oridiff_img \
--epoch 2000 \
--batch_size 512 \
--filter_nopad \
--job_name 0930_label_oridiff_img_epoch2000_nopad_bs512_shuffle

CUDA_VISIBLE_DEVICES=3 \
python main.py \
--mode label_oridiff_normlentime_seid \
--epoch 1000 \
--batch_size 512 \
--filter_area \
--interpolated \
--traj_len 200 \
--lr 1e-5 \
--job_name 1003_label_oridiff_normlentime_seid_epoch1000_bs512_shuffle_filterarea_interlen200_lr1e5


CUDA_VISIBLE_DEVICES=7 \
python main_ori.py \
--job_name 1004_ori_filterarea_interlen300











CUDA_LAUNCH_BLOCKING=1 \

CUDA_VISIBLE_DEVICES=4 \
python main.py \
--mode label_oridiff_normlentime_seid \
--epoch 200 \
--lr 5e-5 \
--batch_size 16 \
--filter_area \
--interpolated \
--traj_len 200 \
--job_name test











############################################################################################################################################################################################################################################


CUDA_VISIBLE_DEVICES=6 \
python traj_generate.py \
--job_name generate_test \
--mode label_oridiff_img 

CUDA_VISIBLE_DEVICES=4 \
python traj_generate.py \
--job_name generate_test \
--mode label_oridiff_normlentime_seid

--mode oridiff_normlentime_seid 





