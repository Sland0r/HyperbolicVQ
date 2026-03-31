#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=extract_ppls
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/extract_ppls_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# Usage: sbatch extract_ppls.sh <input_log>
INPUT_LOG=$1

if [ -z "$INPUT_LOG" ]; then
    echo "Error: Need to provide an input log file path."
    echo "Usage: sbatch extract_ppls.sh <input_log>"
    exit 1
fi

python3 /home/acolombo/VAEs/egs/SoundStream_24k_240d/extract_ppls.py "$INPUT_LOG"
