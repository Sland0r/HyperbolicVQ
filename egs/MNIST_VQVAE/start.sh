#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=train_cifar100_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/train_cifar100_vqvae_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

cd /home/acolombo/VAEs/egs/MNIST_VQVAE

python3 -u train.py \
        --dataset cifar100 \
        --N_EPOCHS 50 \
        --BATCH_SIZE 256 \
        --D 32 \
        --n_q 8 \
        --bins 32 \
        --lr 3e-4 \
        --LAMBDA_COM 1.0 \
        --print_freq 100 \
        --codebook_number 0 \
        --number_of_steps 1000 \
        --ema \
        --kmeans_init \
        #--c 1.0 \
