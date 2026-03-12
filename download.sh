#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

python download_dataset.py