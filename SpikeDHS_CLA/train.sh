CUDA_VISIBLE_DEVICES=0 python train.py \
--experiment_description "spiking jelly dspike" \
--epoch 100 \
--arch SNN_Darts_s2e0 \
--seed 12345 \
--layers 8 \
--drop_path_prob 0.0 \
--cutout \
--auxiliary \
--save "normal darts. 012-spike" \

# --data "/media/HDD1/personal_files/zkx/datasets/CIFAR10_DVS/dvs-cifar10" \

# --experiment_description "CIFAR100 T=5. ANN FC and pooling. stem SNN with bn Searching epoch 50 validacc 8720% learning rate 0.025 with auxiliary single SNN drop_path_prob 0.0 epoch 100" \
