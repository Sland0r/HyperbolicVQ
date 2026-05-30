#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp2_hrq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs_nlp_2/nlp2_hrq_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# Paper (Appendix C.1) hyperparameters
BATCH_SIZE=2048
EPOCHS=500
LEARNING_RATE=1.0
WARMUP_LR=0.01
WARMUP_EPOCHS=0

# Paper-faithful transitive-closure split (set to 'direct' for the leak-free variant)
SPLIT_MODE=closure

# Curvature: c=0 -> RQ (Euclidean), c>0 -> HRQ (hyperbolic). Sweep both.
C_VALUES=(1.0)

# Paper Table 1 sweep: token lengths k, codebook sizes |C_i|, embedding dims h
N_Q_VALUES=(4)
BINS_VALUES=(128)
EMBED_DIMS=(16)

echo "Starting NLP_2 (paper-faithful) Hierarchy Modeling sweep..."
echo "split_mode=${SPLIT_MODE}"

for C in "${C_VALUES[@]}"; do
for N_Q in "${N_Q_VALUES[@]}"; do
for BINS in "${BINS_VALUES[@]}"; do
for EMBED_DIM in "${EMBED_DIMS[@]}"; do

    RUN_ID="${SLURM_JOB_ID}_c${C}_k${N_Q}_s${BINS}_h${EMBED_DIM}_${SPLIT_MODE}"
    SAVE_DIR="/home/acolombo/VAEs/checkpoint/nlp_2/${RUN_ID}"
    mkdir -p ${SAVE_DIR}

    echo "=========================================="
    echo "Run ${RUN_ID}: c=${C}, k=${N_Q}, |C_i|=${BINS}, h=${EMBED_DIM}"
    echo "=========================================="
    echo "1. Training HRQ Model"
    echo "Epochs: ${EPOCHS}"
    echo "LR: ${LEARNING_RATE}"
    echo "Warmup LR: ${WARMUP_LR}"
    echo "Warmup Epochs: ${WARMUP_EPOCHS}"
    echo "Embedding Dim: ${EMBED_DIM}"
    echo "Number of Q: ${N_Q}"
    echo "Number of Bins: ${BINS}"

    python3 -u /home/acolombo/VAEs/NLP_2/train_paper.py \
        --embed_dim ${EMBED_DIM} \
        --compile \
        --n_q ${N_Q} \
        --bins ${BINS} \
        --c ${C} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --lr ${LEARNING_RATE} \
        --warmup_lr ${WARMUP_LR} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --split_mode ${SPLIT_MODE} \
        --save_dir ${SAVE_DIR} \
        --num_workers 8 \
        --approx \
        --new_method \
        #--hste \
        #--quant 0.01 \
        #--commitment_weight 0.1 \
        #--ema
        #--test

    # Recall@10 via paper-faithful beam-search generation (downstream: 100 epochs)
    # what about teacher forcing?
    python3 -u /home/acolombo/VAEs/NLP_2/eval_paper.py \
        --model_path ${RUN_ID} \
        --epochs 50 \
        --teacher_forcing \

done
done
done
done

echo "Done!"
