#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp_hrq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs_nlp/nlp_hrq_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# Paper (Appendix C.1) hyperparameters
BATCH_SIZE=1024
EPOCHS=1500
LEARNING_RATE=1.0
WARMUP_LR=0.01
WARMUP_EPOCHS=20
C=1.0

# MODEL
EMBED_DIM=16
N_Q=4
BINS=128

echo "Starting NLP Experiments for HRQ..."

SAVE_DIR="/home/acolombo/VAEs/checkpoint/nlp/${SLURM_JOB_ID}"
mkdir -p ${SAVE_DIR}

echo "=========================================="
echo "Running experiments for C=${C}"
echo "============================================"

echo "1. Training HRQ Model"
echo "Epochs: ${EPOCHS}"
echo "LR: ${LEARNING_RATE}"
echo "Warmup LR: ${WARMUP_LR}"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "Embedding Dim: ${EMBED_DIM}"
echo "Number of Q: ${N_Q}"
echo "Number of Bins: ${BINS}"

python3 -u /home/acolombo/VAEs/NLP/train_hierarchy.py \
    --embed_dim ${EMBED_DIM} \
    --n_q ${N_Q} \
    --bins ${BINS} \
    --c ${C} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --warmup_lr ${WARMUP_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --save_dir ${SAVE_DIR} \
    --approx \
    #--hste \
    #--new_method \
    #--constructive

echo "2. Evaluate Models"

python3 -u /home/acolombo/VAEs/NLP/eval_recall.py \
    --model_path ${SLURM_JOB_ID} \
    --epochs 100 \
    # --teacher_forcing \
    # --beam_search \

echo "Done!"
