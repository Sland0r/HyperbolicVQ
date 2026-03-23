#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ddp_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/acolombo/VAEs/logs/ddp_test_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

python3 /home/acolombo/VAEs/test_ddp_geoopt.py
