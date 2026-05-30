#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=logs_nlp/nlp_sim_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# Install gensim for word embeddings if not present
# pip install --user gensim

echo "Starting NLP cluster similarity evaluation..."

python3 -u /home/acolombo/VAEs/NLP/check_cluster_similarity.py
    
echo "Done!"
