#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hierarchy_discovery
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs_rec_1/discovery_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Remove old processed data to regenerate with leave-one-out split
rm -f /home/acolombo/VAEs/dataset/Amazon/beauty_processed.pt

SAVE_DIR="/home/acolombo/VAEs/checkpoint/rec_1/${SLURM_JOB_ID}"
mkdir -p ${SAVE_DIR}

echo "Starting Hierarchy Discovery Experiments..."

python3 rec_1/train_vae.py \
    --c 0.0 \
    --epochs 500 \
    --embed_dim 32 \
    --bins 256 \
    --lr 3e-4 \
    --batch_size 512 \
    --save_dir ${SAVE_DIR} \
    #--approx
python3 rec_1/train_recommender.py \
    --c 0.0 \
    --epochs 100 \
    --lr 1e-4 \
    --warmup_epochs 5

echo "Done!"
