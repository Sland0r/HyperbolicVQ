#!/bin/bash
set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH:-}"

CHECKPOINT_ROOT="/home/acolombo/VAEs/checkpoint/nlp_2"

python3 -u /home/acolombo/VAEs/NLP_2/backfill_plots.py \
  --checkpoint_root "${CHECKPOINT_ROOT}" "$@"
