#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ss_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/ss_debug_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# DEBUG POINCARE
C=1.0

# TRAINING (short debug)
BATCH_SIZE=32
N_EPOCHS=1
WARMUP_EPOCHS_G=0
NUMBER_OF_STEPS=100
PRINT_FREQ=50
DATASET=100

# MODEL
RATIOS="6 5 4 2"
TARGET_BANDWIDTHS="1 2 4 8 12"
EXPONENTIAL_LAMBDA=0.0
SR=24000

# LEARNING RATES AND LOSSES
LR_G=3e-4
LR_MANIFOLD=1e-4
LAMBDA_COM=0.1
LAMBDA_FEAT=1
LAMBDA_ADV=1
LAMBDA_RECON=1
LAMBDA_SEP=0

# CODEBOOK
THRESHOLD_EMA_DEAD_CODE=2
CODEBOOK_WEIGHT=1.0
COMMITMENT_WEIGHT=0.25
DOT_PRODUCT_WEIGHT=0.0
ENTAILMENT_CONE_WEIGHT=0.0
CODEBOOK_DIM=512
DECAY=0.99

python3 -u /home/acolombo/VAEs/egs/SoundStream_24k_240d/main3_ddp.py \
        --BATCH_SIZE ${BATCH_SIZE} \
        --N_EPOCHS ${N_EPOCHS} \
        --c ${C} \
        --PATH  /home/acolombo/VAEs/checkpoint/soundstream/h1_u_debug \
        --train_data_path ${DATASET} \
        --valid_data_path /home/acolombo/VAEs/dataset/LibriTTS/dev-clean \
        --number_of_steps ${NUMBER_OF_STEPS} \
        --sr ${SR} \
        --LAMBDA_COM ${LAMBDA_COM} \
        --LAMBDA_ADV ${LAMBDA_ADV} \
        --LAMBDA_REC ${LAMBDA_RECON} \
        --LAMBDA_FEAT ${LAMBDA_FEAT} \
        --lr_g ${LR_G} \
        --lr_manifold ${LR_MANIFOLD} \
        --ratios ${RATIOS} \
        --decay ${DECAY} \
        --target_bandwidths ${TARGET_BANDWIDTHS} \
        --exponential_lambda ${EXPONENTIAL_LAMBDA} \
        --codebook_weight ${CODEBOOK_WEIGHT} \
        --commitment_weight ${COMMITMENT_WEIGHT} \
        --dot_product_weight ${DOT_PRODUCT_WEIGHT} \
        --entailment_cone_weight ${ENTAILMENT_CONE_WEIGHT} \
        --LAMBDA_SEP ${LAMBDA_SEP} \
        --threshold_ema_dead_code ${THRESHOLD_EMA_DEAD_CODE} \
        --codebook_dim ${CODEBOOK_DIM} \
        --print_freq ${PRINT_FREQ} \
        --warmup_epochs_g ${WARMUP_EPOCHS_G} \
        --uniform \
        --constructive

python3 /home/acolombo/VAEs/egs/SoundStream_24k_240d/extract_ppls.py logs/ss_debug_${SLURM_JOB_ID}.out
