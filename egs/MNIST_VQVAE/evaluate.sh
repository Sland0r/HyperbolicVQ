#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=evaluate_mnist_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=evaluations/evaluate_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/evaluate.py \
    --checkpoint h1_dot01/latest.pth \
    --dataset mnist \
    --num_samples 1000