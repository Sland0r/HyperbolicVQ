#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs_nlp/nlp_eval_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

MODEL=new_hste
EPOCHS=10
BATCH_SIZE=512

echo "Starting NLP evaluation..."

echo "=========================================="
echo "Running evaluation for model: ${MODEL}"
echo "============================================"

python3 -u /home/acolombo/VAEs/NLP/eval_recall.py \
    --model_path ${MODEL} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --beam_search \
    # --teacher_forcing \
    # --beam_search
    
echo "Done!"
