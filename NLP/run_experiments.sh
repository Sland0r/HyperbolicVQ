#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp_hrq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs_nlp/nlp_hrq_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# Paper (Appendix C.1) hyperparameters
BATCH_SIZE=1024
EPOCHS=10
LEARNING_RATE=1.0
WARMUP_LR=0.01
WARMUP_EPOCHS=0
C=1.0

# MODEL
EMBED_DIM=16
N_Q=4
BINS=128

echo "Starting NLP Experiments for HRQ..."

mkdir -p checkpoint/nlp

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
    --save_path checkpoint/nlp/hrq_model_c${C}_ep${EPOCHS}.pt

echo "2. Evaluate Models"

python3 -u /home/acolombo/VAEs/NLP/eval_recall.py \
    --c ${C} \
    --model_path /home/acolombo/VAEs/checkpoint/nlp/hrq_model_c${C}_ep${EPOCHS}.pt \
    --embed_dim ${EMBED_DIM} \
    --n_q ${N_Q} \
    --bins ${BINS} \
    --teacher_forcing

echo "Done!"
