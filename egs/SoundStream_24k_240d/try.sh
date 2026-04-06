#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=download_other_500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:30:00
#SBATCH --output=logs/download_other_500_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

python3 /home/acolombo/VAEs/egs/SoundStream_24k_240d/download.py