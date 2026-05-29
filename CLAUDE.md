# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

This project extends [AcademiCodec](https://arxiv.org/pdf/2305.02765.pdf) to explore **Hyperbolic Residual Quantization (HRQ)**: VQ-VAE/codec models whose codebooks live on the Poincaré ball rather than Euclidean space. The curvature parameter `c` controls the geometry — `c=0` recovers standard RVQ, `c>0` gives hyperbolic HRQ. The hypothesis is that hierarchical structure in audio/image/language can be captured more naturally in hyperbolic space.

## Environment Setup

This runs on a SLURM HPC cluster. All GPU jobs must be submitted via `sbatch`, not run interactively.

```bash
# Activate environment (inside a SLURM job or interactive session)
module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes

# Install / update dependencies
pip install -r requirements.txt
```

Every training script must set:
```bash
export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
```

**Note**: `academicodec/quantization/core_vq.py` has a hard-coded `sys.path.insert(0, '/home/acolombo/music')` to import `HyperbolicEntailmentConeLoss` from `~/music/hyp_modules.py`. That file lives outside this repo and must be present on the machine.

## Running Experiments

Each experiment lives in `egs/` or a top-level subfolder and has its own SLURM script.

| Experiment | Submit |
|---|---|
| Audio codec (SoundStream 24kHz) | `sbatch egs/SoundStream_24k_240d/start.sh` |
| Image VQ-VAE (MNIST/CIFAR/EMNIST) | `sbatch egs/MNIST_VQVAE/start.sh` |
| NLP hierarchy (WordNet) | `sbatch NLP/run_experiments.sh` |
| Recommendation hierarchy discovery | `sbatch rec_1/run_discovery.sh` |

After the audio codec training run finishes, perplexity values are extracted automatically:
```bash
python3 egs/SoundStream_24k_240d/extract_ppls.py logs/<job_name>_<job_id>.out
```

## Architecture

```
academicodec/               # core package
├── quantization/
│   ├── core_vq.py          # ALL hyperbolic math + VQ forward pass
│   └── vq.py               # ResidualVectorQuantizer (thin wrapper, exposes QuantizedResult)
├── modules/
│   ├── seanet.py           # SEANetEncoder / SEANetDecoder (1-D conv backbone)
│   ├── conv.py             # causal / weight-normed convolution primitives
│   └── transformer.py      # optional transformer block
└── models/
    ├── encodec/
    │   ├── net3.py         # SoundStream generator (encoder + RVQ + decoder)
    │   ├── msstftd.py      # Multi-Scale STFT Discriminator
    │   └── loss.py         # generator + discriminator losses
    └── soundstream/
        └── models.py       # MultiPeriodDiscriminator, MultiScaleDiscriminator

egs/
├── SoundStream_24k_240d/
│   └── main3_ddp.py        # main training loop (DDP, adversarial, all losses)
└── MNIST_VQVAE/
    └── train.py            # image VQ-VAE training loop

NLP/
├── train_hierarchy.py      # InfoNCE contrastive hierarchy learning on WordNet
└── eval_recall.py          # recall@k evaluation

rec_1/
├── train_vae.py            # RQ-VAE / HRQ-VAE on Amazon Review embeddings
└── train_recommender.py    # seq2seq recommender on generated tokens
```

## Key Implementation Details

### Hyperbolic geometry (`core_vq.py`)

All Poincaré ball operations are implemented from scratch (no geoopt dependency for the core math):
- `exp_map0` / `log_map0`: exponential/logarithmic maps at the origin
- `mobius_add` / `mobius_sub`: Möbius addition
- `hyperbolic_distance_sq` / `pairwise_hyperbolic_distance_sq`: geodesic distances
- `einstein_midpoint`: for computing codebook centroids in hyperbolic space
- `project`: clips points to stay strictly inside the ball (radius `(1-ε)/√c`)

The `parallel_transport` function at the top of `core_vq.py` is a conformal-factor approximation — intentionally not exact PT — because geoopt's internal API changed between versions.

### Quantization loss weights

Controlled by CLI flags in each training script:
- `--codebook_weight` — commitment: codes toward encoder output (default 1.0)
- `--commitment_weight` — encoder outputs toward codes (default 0.25)
- `--dot_product_weight` — dot-product alignment loss
- `--entailment_cone_weight` — hyperbolic entailment cone loss (uses `HyperbolicEntailmentConeLoss` from `~/music/hyp_modules.py`)
- `--constructive` / `--gyration` — flag-only options that switch residual transport mode

### Dead code revival
`--threshold_ema_dead_code` (default 2): codebook entries with EMA cluster size below this are replaced with random encoder outputs from the current batch.

## Checkpoints

Checkpoints are saved under `checkpoint/<experiment>/<slurm_job_id>/` and are gitignored. The latest checkpoint is always `latest.pth`.

## Evaluation

- **Audio**: `egs/SoundStream_24k_240d/eval_ddp.py` — reconstruction metrics; `extract_ppls.py` parses perplexity from logs
- **Image VQ-VAE**: `egs/MNIST_VQVAE/evaluate.py` — FID, codebook usage, reconstruction quality; `check_codes.py` — visualise code assignments
- **NLP**: `NLP/eval_recall.py` — recall@k on WordNet hypernymy
- **Rec**: `rec_1/train_recommender.py` — hit-rate / NDCG on Amazon Reviews
