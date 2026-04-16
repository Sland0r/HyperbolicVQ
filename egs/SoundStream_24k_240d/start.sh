#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=h1_bw_u_cm5cb5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:30:00
#SBATCH --output=logs/h1_bw_u_cm5cb5_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
NAME="h1_bw_u_cm5cb5"

# POINCARE
C=1.0

# TRAINING
BATCH_SIZE=32
N_EPOCHS=10
WARMUP_EPOCHS_G=0
NUMBER_OF_STEPS=1000 # plot codebook every n batches
PRINT_FREQ=1000 # log every n batches
DATASET=100

# MODEL
RATIOS="6 5 4 2"
TARGET_BANDWIDTHS="1 2 4 8 12"
EXPONENTIAL_LAMBDA=0.0
SR=24000

# LEARNING RATES AND LOSSES
LR_G=3e-4
LR_MANIFOLD=1e-3
LAMBDA_COM=1
LAMBDA_FEAT=1
LAMBDA_ADV=1
LAMBDA_RECON=1

# CODEBOOK
CODEBOOK_WEIGHT=0.5 # codes towards encoder output
COMMITMENT_WEIGHT=0.5 # encoder outputs towards codes


python3 -u /home/acolombo/VAEs/egs/SoundStream_24k_240d/main3_ddp.py \
        --BATCH_SIZE ${BATCH_SIZE} \
        --N_EPOCHS ${N_EPOCHS} \
        --c ${C} \
        --PATH  /home/acolombo/VAEs/checkpoint/soundstream \
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
        --target_bandwidths ${TARGET_BANDWIDTHS} \
        --exponential_lambda ${EXPONENTIAL_LAMBDA} \
        --codebook_weight ${CODEBOOK_WEIGHT} \
        --commitment_weight ${COMMITMENT_WEIGHT} \
        --print_freq ${PRINT_FREQ} \
        --warmup_epochs_g ${WARMUP_EPOCHS_G} \
        #--pre_quant_batchnorm \
        #--use_spec_augment \
        #--ema \
        #--kmeans_init

# Extract PPLs from the generated log file automatically
python3 /home/acolombo/VAEs/egs/SoundStream_24k_240d/extract_ppls.py logs/${NAME}_${SLURM_JOB_ID}.out