#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/download_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes
pip install "sentence-transformers<3.0.0"

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

echo "Starting download..."

python3 rec_1/amazon_dataset.py

echo "Done!"
