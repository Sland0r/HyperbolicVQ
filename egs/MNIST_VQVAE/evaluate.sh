#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=evaluate_emnist_vqvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:30:00
#SBATCH --output=evaluations/evaluate_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

# Pass test-selection flags through to evaluate.py. Examples:
#   sbatch evaluate.sh                                  # run the default suite
#   sbatch evaluate.sh --robustness                     # only robustness
#   sbatch evaluate.sh --generation --interpretability  # only those two
#   sbatch evaluate.sh --rq_transformer                 # only the RQ-Transformer
# If no flag is given, evaluate.py runs robustness + generation + interpretability.
EVAL_FLAGS=("$@")

CHECKPOINTS_DIR="/home/acolombo/VAEs/checkpoint/cifar100/"

for model_dir in "$CHECKPOINTS_DIR"/*/; do
    model_dir=${model_dir%/}
    model_name=$(basename "$model_dir")
    
    # Find the best_*.pth with the highest epoch number
    latest_best=$(ls "$model_dir"/best_*.pth 2>/dev/null | sort -V | tail -n 1)
    
    if [ -z "$latest_best" ]; then
        echo "No best_*.pth found in $model_name, skipping..."
        continue
    fi
    
    best_file=$(basename "$latest_best")
    echo "Evaluating $model_name using $best_file"
    
    dataset=$(python3 -c "import sys; sys.path.insert(0,'$model_dir'); import config; print(config.dataset)")
    echo "  dataset=$dataset"

    python3 /home/acolombo/VAEs/egs/MNIST_VQVAE/evaluate.py \
        --checkpoint "$CHECKPOINTS_DIR/$model_name/$best_file" \
        --dataset "$dataset" \
        --num_samples 10000 \
        --rq_transformer \
        "${EVAL_FLAGS[@]}"
done