#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=train_mnist_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs_mnist/train_mnist_vqvae_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

cd /home/acolombo/VAEs/egs/MNIST_VQVAE

#POINCARE 
C=1.0

#TRAINING
BATCH_SIZE=256
N_EPOCHS=50
WARMUP_EPOCHS_G=0
NUMBER_OF_STEPS=1000 # plot codebook every n batches
PRINT_FREQ=1000 # log every n batches
DATASET=mnist

# MODEL
EXPONENTIAL_LAMBDA=0.0
D=2
N_Q=3
BINS=10

# LEARNING RATES AND LOSSES
LR_G=3e-4
LR_MANIFOLD=1e-4
LAMBDA_LAT=0.1 # applied on top of all weights below
LAMBDA_SEP=0.0

# CODEBOOK
THRESHOLD_EMA_DEAD_CODE=2
CODEBOOK_WEIGHT=1.0     # codes towards encoder output
COMMITMENT_WEIGHT=0.25  # encoder outputs towards codes
DOT_PRODUCT_WEIGHT=0.0
ENTAILMENT_CONE_WEIGHT=0.25


python3 -u train.py \
        --dataset $DATASET \
        --N_EPOCHS $N_EPOCHS \
        --BATCH_SIZE $BATCH_SIZE \
        --D $D \
        --n_q $N_Q \
        --bins $BINS \
        --lr_g $LR_G \
        --lr_manifold $LR_MANIFOLD \
        --LAMBDA_LAT $LAMBDA_LAT \
        --LAMBDA_SEP $LAMBDA_SEP \
        --threshold_ema_dead_code $THRESHOLD_EMA_DEAD_CODE \
        --codebook_weight $CODEBOOK_WEIGHT \
        --commitment_weight $COMMITMENT_WEIGHT \
        --dot_product_weight $DOT_PRODUCT_WEIGHT \
        --entailment_cone_weight $ENTAILMENT_CONE_WEIGHT \
        --print_freq $PRINT_FREQ \
        --codebook_number 0 \
        --number_of_steps $NUMBER_OF_STEPS \
        --c $C \
        --exponential_lambda $EXPONENTIAL_LAMBDA \
        --constructive \
        #--uniform \
        #--ema \
        #--kmeans_init \


CHECKPOINT="/home/acolombo/VAEs/checkpoint/mnist_vqvae/$SLURM_JOB_ID/latest.pth"
python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/check_codes.py \
        --checkpoint $CHECKPOINT

python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/check_codes.py \
        --checkpoint $CHECKPOINT \
        --plot_images

python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/check_codes.py \
        --checkpoint $CHECKPOINT \
        --val_scatter