#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hierarchy_discovery
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs_rec_1/discovery_%x_%A.out

# Parametrized rec discovery pipeline (VAE -> recommender) for any Amazon
# Reviews 2014 category and quantizer variant.
#
#   DATASET  SNAP category name, passed straight to prepare_data / train_*.py
#            (default Beauty). e.g. Toys_and_Games (TaG), Sports_and_Outdoors (SaO)
#   METHOD   quantizer variant (default a4gc):
#              euclidean  c=0 standard RVQ
#              vanilla    c=1 plain hyperbolic RVQ (--approx)
#              vanillagc  c=1 plain hyperbolic RVQ + gradient_correction
#                         (detaches the residual each stage -> cuts the leaked
#                          per-stage recon gradient; only layer-0 STE reaches enc)
#              riem       c=1 per-layer HSTE, riemannian grad
#              a4         c=1 block-PT-STE, riemannian (no gc)
#              a4gc       c=1 block-PT-STE, riemannian + gradient_correction
#   DEDUP=0  disable the uniqueness/tie-break token (paper §2.2; on by default)
#   REGEN=1  rebuild the leave-one-out split from raw reviews (keeps embeddings)
#
# Examples:
#   sbatch rec_1/run_discovery.sh                                      # Beauty, a4gc
#   DATASET=Toys_and_Games METHOD=euclidean sbatch rec_1/run_discovery.sh
#   DATASET=Sports_and_Outdoors METHOD=a4gc sbatch rec_1/run_discovery.sh

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-Beauty}
METHOD=${METHOD:-a4gc}
# Tag the codes filename with the method so variants that share a curvature
# (e.g. vanilla and a4gc both run at c=1.0) don't clobber each other's codes.
CODES_TAG="${CODES_TAG:-_${METHOD}}"
DATASET_DIR=/scratch-shared/acolombo/VAEs/dataset/Amazon
DATASET_LC=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')

# Map METHOD -> curvature + VAE quantizer flags. The recommender only needs --c
# (it loads the codes file keyed by curvature).
case "${METHOD}" in
    euclidean) C=0.0; VAE_FLAGS="" ;;
    vanilla)   C=1.0; VAE_FLAGS="--approx" ;;
    vanillagc) C=1.0; VAE_FLAGS="--approx --gradient_correction" ;;
    riem)      C=1.0; VAE_FLAGS="--new_method --approx --hste --hste_riemannian" ;;
    a4)        C=1.0; VAE_FLAGS="--new_method --approx --hste_riemannian --block_hste_pt" ;;
    a4gc)      C=1.0; VAE_FLAGS="--new_method --approx --hste_riemannian --block_hste_pt --gradient_correction" ;;
    # Component ablations of A4+gc (thesis-1): drop exactly one GHRQ component.
    #   hra_nodhste  HRA forward, NO d-HSTE: new_method + gc, no block hop / no
    #                riem -> per-layer identity STE (A4+gc minus d-HSTE).
    #   dhste_nohra  d-HSTE backward, NO HRA: block-PT-STE + riem + gc on the
    #                vanilla Möbius aggregation (A4+gc minus new_method).
    hra_nodhste)  C=1.0; VAE_FLAGS="--new_method --approx --gradient_correction" ;;
    hra_nodhste_nogc)  C=1.0; VAE_FLAGS="--new_method --approx" ;;
    dhste_nohra)  C=1.0; VAE_FLAGS="--approx --hste_riemannian --block_hste_pt --gradient_correction" ;;
    *) echo "Unknown METHOD='${METHOD}' (euclidean|vanilla|vanillagc|riem|a4|a4gc|hra_nodhste|hra_nodhste_nogc|dhste_nohra)"; exit 1 ;;
esac

# Uniqueness/tie-break token (paper §2.2), on by default. DEDUP=0 turns it off.
DEDUP=${DEDUP:-1}
DEDUP_FLAG=""
[ "${DEDUP}" = "1" ] && DEDUP_FLAG="--dedup"

# Optional: force the leave-one-out split to be regenerated from raw reviews.
if [ "${REGEN:-0}" = "1" ]; then
    echo "REGEN=1: removing ${DATASET_DIR}/${DATASET_LC}_processed.pt"
    rm -f "${DATASET_DIR}/${DATASET_LC}_processed.pt"
fi

SAVE_DIR="/home/acolombo/VAEs/checkpoint/rec_1/${SLURM_JOB_ID}"
mkdir -p ${SAVE_DIR}

echo "Discovery: dataset=${DATASET} method=${METHOD} c=${C} dedup=${DEDUP} codes_tag=${CODES_TAG} flags=[${VAE_FLAGS}] save_dir=${SAVE_DIR}"

python3 rec_1/train_vae.py \
    --dataset ${DATASET} \
    --c ${C} \
    --epochs 5000 \
    --embed_dim 32 \
    --bins 128 \
    --lr 1e-4 \
    --batch_size ${BATCH:-2048} \
    --save_dir ${SAVE_DIR} \
    --quant 0.01 \
    --recon 1000 \
    --codes_tag "${CODES_TAG}" \
    ${VAE_FLAGS} \
    ${DEDUP_FLAG}

python3 rec_1/train_recommender.py \
    --dataset ${DATASET} \
    --c ${C} \
    --epochs 100 \
    --lr 1e-4 \
    --warmup_epochs 0 \
    --codes_tag "${CODES_TAG}" \
    ${DEDUP_FLAG}

echo "Done!"
