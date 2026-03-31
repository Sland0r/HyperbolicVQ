#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=train_cifar100_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_cifar100_vqvae_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

cd /home/acolombo/VAEs/egs/MNIST_VQVAE

python3 -u train.py \
        --dataset emnist \
        --N_EPOCHS 50 \
        --BATCH_SIZE 256 \
        --D 2 \
        --n_q 4 \
        --bins 5 \
        --lr_g 3e-4 \
        --lr_manifold 1e-4 \
        --LAMBDA_COM 1.0 \
        --print_freq 10000 \
        --codebook_number 0 \
        --number_of_steps 1000 \
        --c 1.0 \
        #--ema \
        #--kmeans_init \
        
