#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=check_codes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/check_codes_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
CHECKPOINT="/home/acolombo/VAEs/checkpoint/mnist_vqvae/21520796/latest.pth"

python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/check_codes.py \
        --checkpoint $CHECKPOINT \
        
python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/check_codes.py \
        --checkpoint $CHECKPOINT \
        --plot_images