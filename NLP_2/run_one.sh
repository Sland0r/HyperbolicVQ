#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=nlp2_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs_nlp_4/nlp2_sweep_%A.out

# Generic single-run launcher for the (method x embed_init_scale) grid sweep.
# Parametrised entirely by env vars (set via `sbatch --export=ALL,...`):
#   METHOD : none | new | hste | new_hste   (which residual-transport flags)
#   SCALE  : embed_init_scale (float)
#   C      : curvature        (default 1.0; all sweep runs are hyperbolic)
#   EPOCHS : default 50
# --approx is kept ON for every c>0 run: it only computes the logged ApproxDist
# diagnostic and never enters the loss, so it is not a confound.

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"

BATCH_SIZE=2048
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE=1.0
WARMUP_LR=0.01
WARMUP_EPOCHS=0
SPLIT_MODE=closure
N_Q=4
BINS=128
EMBED_DIM=16

C="${C:-1.0}"
METHOD="${METHOD:-none}"
SCALE="${SCALE:-1.0}"

case "${METHOD}" in
    none)     METHOD_FLAGS="--approx" ;;
    new)      METHOD_FLAGS="--new_method --approx" ;;
    hste)     METHOD_FLAGS="--hste --approx" ;;
    new_hste) METHOD_FLAGS="--new_method --hste --approx" ;;
    *) echo "Unknown METHOD '${METHOD}' (use: none | new | hste | new_hste)"; exit 1 ;;
esac

# EXTRA_FLAGS: optional extra training flags (e.g. "--grad_clip 1.0" or
# "--grad_norm"), appended verbatim. TAG: optional path suffix to keep RUN_IDs
# unique when sweeping these.
EXTRA="${METHOD_FLAGS} --embed_init_scale ${SCALE} ${EXTRA_FLAGS:-}"

# Sanitise SCALE for use in a path (0.05 -> 0p05)
SCALE_TAG="${SCALE//./p}"
RUN_ID="${SLURM_JOB_ID}_m-${METHOD}_s-${SCALE_TAG}_c${C}_k${N_Q}_b${BINS}_h${EMBED_DIM}_${SPLIT_MODE}${TAG:+_${TAG}}"
SAVE_DIR="/home/acolombo/VAEs/checkpoint/nlp_2/${RUN_ID}"
mkdir -p ${SAVE_DIR}

echo "=========================================="
echo "SWEEP run ${RUN_ID}"
echo "  method=${METHOD}  embed_init_scale=${SCALE}  c=${C}  epochs=${EPOCHS}"
echo "  extra-args: ${EXTRA}"
echo "=========================================="

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
    --test \
    ${EXTRA}

python3 -u /home/acolombo/VAEs/NLP_2/eval_paper.py \
    --model_path ${RUN_ID} \
    --epochs 10 \
    --teacher_forcing

echo "Done!"
