#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp2_hrq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs_nlp_3/nlp2_hrq_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

# Paper (Appendix C.1) hyperparameters
BATCH_SIZE=2048
EPOCHS="${EPOCHS:-50}"          # override via env: `sbatch --export=ALL,EPOCHS=500 ...`
LEARNING_RATE=1.0
WARMUP_LR=0.01
WARMUP_EPOCHS=0
# Optional extra flags appended to training (env), e.g. EXTRA_TEST="--test" to
# turn on the per-quantizer residual-radius / grad-norm diagnostics.
EXTRA_TEST="${EXTRA_TEST:-}"

# Paper-faithful transitive-closure split (set to 'direct' for the leak-free variant)
SPLIT_MODE=closure
N_Q=4
BINS=128
EMBED_DIM=16

# Config selected by the first argument: `sbatch run_paper.sh <config>`.
#   euclidean  : c=0  RQ baseline (Euclidean STE)
#   vanilla    : c=1  hyperbolic, standard STE (no --hste / --new_method)
#   hste_new   : c=1  hyperbolic + --hste + --new_method + embedding-init rescale
#                (the rescale keeps exp_map0(emb) off the boundary, where the
#                 HSTE backward's conformal factor λx² would otherwise explode)
CONFIG="${1:-hste_new}"
case "${CONFIG}" in
    euclidean)        C=0.0; EXTRA="" ;;
    vanilla)          C=1.0; EXTRA="" ;;
    hste_new)         C=1.0; EXTRA="--new_method --hste --approx --embed_init_scale 0.05" ;;
    euclidean_scaled) C=0.0; EXTRA="--embed_init_scale 0.05" ;;
    vanilla_scaled)   C=1.0; EXTRA="--embed_init_scale 0.05" ;;
    init_scaled)      C=1.0; EXTRA="--new_method --hste --approx --embed_init_scale 0.05 --init" ;;
    *) echo "Unknown config '${CONFIG}' (use: euclidean | vanilla | hste_new | euclidean_scaled | vanilla_scaled | init_scaled)"; exit 1 ;;
esac

echo "Starting NLP_2 (paper-faithful) HRQ training — config=${CONFIG} (c=${C})"
echo "split_mode=${SPLIT_MODE}  extra-args: ${EXTRA:-<none>}"

RUN_ID="${SLURM_JOB_ID}_${CONFIG}_c${C}_k${N_Q}_s${BINS}_h${EMBED_DIM}_${SPLIT_MODE}"
SAVE_DIR="/home/acolombo/VAEs/checkpoint/nlp_2/${RUN_ID}"
mkdir -p ${SAVE_DIR}

echo "=========================================="
echo "Run ${RUN_ID}: c=${C}, k=${N_Q}, |C_i|=${BINS}, h=${EMBED_DIM}, epochs=${EPOCHS}"
echo "=========================================="

# --compile OFF: the HyperbolicSTE custom autograd Function breaks graph compilation.
python3 -u /home/acolombo/VAEs/NLP_2/train_paper.py \
    --embed_dim ${EMBED_DIM} \
    --n_q ${N_Q} \
    --bins ${BINS} \
    --c ${C} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --warmup_lr ${WARMUP_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --split_mode ${SPLIT_MODE} \
    --save_dir ${SAVE_DIR} \
    --num_workers 8 \
    ${EXTRA} ${EXTRA_TEST}

# Recall@10 via paper-faithful beam-search generation
python3 -u /home/acolombo/VAEs/NLP_2/eval_paper.py \
    --model_path ${RUN_ID} \
    --epochs 10 \
    --teacher_forcing

echo "Done!"
