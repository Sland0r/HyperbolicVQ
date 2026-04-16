#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=extract_rec_loss
#SBATCH --output=rankings/extract_rec_loss_%j.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

cd /home/acolombo/VAEs
python3 extract_rec_loss.py --folder mnist_vqvae