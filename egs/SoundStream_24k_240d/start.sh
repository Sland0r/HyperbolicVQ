#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_soundstream_24k_240d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --output=logs/vanilla_no_ema_24k_240d_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# POINCARE
C=1.0

# TRAINING
BATCH_SIZE=32
N_EPOCHS=10
WARMUP_EPOCHS_G=0
NUMBER_OF_STEPS=500 # plot codebook every n batches
PRINT_FREQ=1000 # log every n batches

# MODEL
RATIOS="6 5 4 2"
TARGET_BANDWIDTHS="1 2 4 8 12"
SR=24000

# LEARNING RATES AND LOSSES
LR_G=3e-4
LR_MANIFOLD=1e-3
LAMBDA_COM=1
LAMBDA_ADV=1
LAMBDA_RECON=1


python3 -u /home/acolombo/VAEs/egs/SoundStream_24k_240d/main3_ddp.py \
        --BATCH_SIZE ${BATCH_SIZE} \
        --N_EPOCHS ${N_EPOCHS} \
        --c ${C} \
        --PATH  /home/acolombo/VAEs/checkpoint/soundstream \
        --train_data_path /home/acolombo/VAEs/dataset/LibriTTS/train-clean-100 \
        --valid_data_path /home/acolombo/VAEs/dataset/LibriTTS/dev-clean \
        --number_of_steps ${NUMBER_OF_STEPS} \
        --sr ${SR} \
        --LAMBDA_COM ${LAMBDA_COM} \
        --LAMBDA_ADV ${LAMBDA_ADV} \
        --LAMBDA_REC ${LAMBDA_RECON} \
        --lr_g ${LR_G} \
        --lr_manifold ${LR_MANIFOLD} \
        --ratios ${RATIOS} \
        --target_bandwidths ${TARGET_BANDWIDTHS} \
        --print_freq ${PRINT_FREQ} \
        --warmup_epochs_g ${WARMUP_EPOCHS_G} \
        #--ema \
        #--pre_quant_batchnorm \
        #--kmeans_init