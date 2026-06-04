# Hierarchy Discovery (Section 4.2)
This folder contains the materials to reproduce the Hierarchy Discovery portion of the paper using the Amazon Reviews 2014 dataset.

## Structure
- `train_vae.py`: Trains the RQ-VAE and HRQ-VAE autoencoders to produce multitokens from MPNet continuous embeddings without explicit hierarchical supervision.
- `train_recommender.py`: Trains a sequence-to-sequence recommender system on the generated tokens to predict the next item in a user's purchase history.
- `run_discovery.sh`: SLURM script to execute the pipeline.

## Uniqueness token (paper §2.2)
RQ appends an extra **tie-break token** so items that collide on the same
`n_q`-token multitoken still get a unique id (the paper's "additional token that
distinguishes conflicts"). This is **off by default** and gated behind `--dedup`:

```bash
# 1. produce codes + a uniqueness-token variant (item_codes_c<c>_dedup.pt)
python3 rec_1/train_vae.py --c 1.0 --dedup ...
# 2. train/eval the recommender on the unique ids (n_q+1 levels)
python3 rec_1/train_recommender.py --c 1.0 --dedup ...
```

Or via SLURM: `DEDUP=1 sbatch rec_1/run_discovery.sh`.

Without `--dedup` the pipeline keeps its previous behaviour (`n_q` tokens,
colliding items share an id). With it, `train_vae.py` appends a per-item counter
column, and `train_recommender.py` drives the model / trie / beam search off the
actual code width so every full id maps to exactly one item. The dedup token's
range must be `< --bins` (a warning/error fires otherwise).

## Excluded Content
As requested, any experiments requiring text generation using LLMs (e.g., Claude for the MovieLens dataset) have been completely removed.
