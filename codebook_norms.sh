#!/bin/bash
#SBATCH --partition=staging
#SBATCH --job-name=codebook_norms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=logs/codebook_norms_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

if [ -z "$1" ]; then
    echo "Usage: bash codebook_norms.sh <checkpoint_folder_or_file>"
    exit 1
fi

python3 codebook_norms.py "$1"
