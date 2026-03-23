#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/env_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

#conda create -n vaes python=3.10
source activate vaes
pip install scikit-learn