#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_soundstream_24k_240d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_soundstream_24k_240d_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

python3 /home/acolombo/VAEs/egs/SoundStream_24k_240d/main3_ddp.py \
        --BATCH_SIZE 16 \
        --N_EPOCHS 1 \
        --save_dir /home/acolombo/VAEs/logs \
        --PATH  /home/acolombo/VAEs/checkpoint/soundstream.pth \
        --train_data_path /home/acolombo/VAEs/dataset/LibriTTS/train-clean-100 \
        --valid_data_path /home/acolombo/VAEs/dataset/LibriTTS/dev-clean \
        --sr 24000 \
        --ratios 6 5 4 2 \
        --target_bandwidths 1 2 4 8 12 \
        --print_freq 1000 
