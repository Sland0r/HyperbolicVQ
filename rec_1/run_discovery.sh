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

# Uniqueness/tie-break token (paper §2.2). Set DEDUP=1 to append a per-item
# disambiguation token so colliding multitokens map to distinct items.
DEDUP=${DEDUP:-0}
DEDUP_FLAG=""
if [ "${DEDUP}" = "1" ]; then
    DEDUP_FLAG="--dedup"
    echo "Uniqueness token enabled (--dedup)."
fi

python3 rec_1/train_vae.py \
    --c 1.0 \
    --epochs 5000 \
    --embed_dim 32 \
    --dedup \
    --bins 128 \
    --lr 1e-4 \
    --batch_size 2048 \
    --save_dir ${SAVE_DIR} \
    --quant 0.01 \
    --recon 1000 \
    --hste_riemannian \
    --approx \
    --hste \
    --new_method \
    ${DEDUP_FLAG}
    #--new_method \
    #--hste \
    
python3 rec_1/train_recommender.py \
    --c 1.0 \
    --dedup \
    --epochs 100 \
    --lr 1e-4 \
    --warmup_epochs 0 \
    ${DEDUP_FLAG}

echo "Done!"
