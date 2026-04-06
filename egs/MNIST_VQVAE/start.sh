#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=train_mnist_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
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
NUMBER_OF_STEPS=10000 # plot codebook every n batches
PRINT_FREQ=10000 # log every n batches
DATASET=emnist

# MODEL
EXPONENTIAL_LAMBDA=0.0
D=2
N_Q=4
BINS=5

# LEARNING RATES AND LOSSES
LR_G=3e-4
LR_MANIFOLD=1e-4
LAMBDA_COM=1.0


python3 -u train.py \
        --dataset $DATASET \
        --N_EPOCHS $N_EPOCHS \
        --BATCH_SIZE $BATCH_SIZE \
        --D $D \
        --n_q $N_Q \
        --bins $BINS \
        --lr_g $LR_G \
        --lr_manifold $LR_MANIFOLD \
        --LAMBDA_COM $LAMBDA_COM \
        --print_freq $PRINT_FREQ \
        --codebook_number 0 \
        --number_of_steps $NUMBER_OF_STEPS \
        --c $C \
        --exponential_lambda $EXPONENTIAL_LAMBDA \
        #--ema \
        #--kmeans_init \


CHECKPOINT="/home/acolombo/VAEs/checkpoint/mnist_vqvae/$SLURM_JOB_ID/latest.pth"
python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/check_codes.py \
        --checkpoint $CHECKPOINT \
        #--plot_images