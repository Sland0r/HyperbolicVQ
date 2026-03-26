#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_mnist_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/eval_mnist_vqvae_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

cd /home/acolombo/VAEs/egs/MNIST_VQVAE

python3 -u eval.py \
        --BATCH_SIZE 128 \
        --D 128 \
        --n_q 4 \
        --bins 256 \
        --checkpoint /home/acolombo/VAEs/checkpoint/mnist_vqvae/REPLACE_WITH_RUN_ID/best_1.pth \
        --ema \
        --kmeans_init \
        #--c 1.0
