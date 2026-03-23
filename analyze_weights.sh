#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=analyze_weights
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/acolombo/VAEs/logs/analyze_weights_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

python3 /home/acolombo/VAEs/egs/SoundStream_24k_240d/analyze_weights.py \
    --path /home/acolombo/VAEs/checkpoint/soundstream/21055539/latest.pth
