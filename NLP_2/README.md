# NLP_2 — Paper-faithful Hierarchy Modeling

This folder reimplements the **Hierarchy Modeling** experiment (paper §4.1,
Appendix A.1, B, C.1) so it lines up with the paper as closely as possible. It
is a sibling to `NLP/`; the original files were copied here unchanged for
reference, and the paper-faithful logic lives in **new standalone scripts**:

| New script | Purpose | Replaces |
|---|---|---|
| `dataset_paper.py` | WordNet noun dataset, **transitive-closure split** by default | `wordnet_dataset.py` |
| `train_paper.py` | Contrastive (H)RQ training, paper optimiser schedule | `train_hierarchy.py` |
| `eval_paper.py` | Downstream seq2seq + **beam-search Recall@10** | `eval_recall.py` |
| `run_paper.sh` | SLURM sweep over the full Table 1 grid + eval | `run_experiments.sh` |

The original copies (`wordnet_dataset.py`, `train_hierarchy.py`,
`eval_recall.py`, `run_experiments.sh`, `run_eval.sh`, `check_cluster_similarity.py`)
are left untouched and still import from the `NLP.` package.

## What was made paper-faithful

1. **Transitive-closure train/test split** (`dataset_paper.py`). The paper uses
   the transitive closure of the WordNet noun hypernymy relation (743,241
   relations) and randomly holds out 15% as the test set (§4.1 / A.1, following
   Nickel & Kiela). This is now the default (`--split_mode closure`).
   - The original `NLP/` split on *direct* edges only. That is still available
     here as `--split_mode direct` — it is leak-free but **not** comparable to
     the paper's Table 1. The closure split *does* leak (held-out pairs are
     re-derivable by composing train edges); `eval_paper.py` prints
     graph-composition and popularity baselines so you can see the leakage floor.

2. **Recall@10 by generation** (`eval_paper.py`). Recall@10 is now measured by
   generating ranked multitoken sequences with **beam search (beam_size=10)**,
   matching the paper's "generate k ranked guesses". The old per-position
   top-k diagnostic is only used if you pass `--greedy_positionwise`.

3. **Full Table 1 sweep** (`run_paper.sh`) over c∈{0,1} (RQ/HRQ),
   k∈{3,4}, |Cᵢ|∈{64,128,256}, h∈{4,8,16,32}, instead of a single config.

Unchanged (already matched the paper): contrastive loss + L_RQ(u) + L_RQ(v),
50 negatives with `H'(u) ∪ {u}`, SGD / Riemannian SGD, lr=1.0, 1500 epochs with
20 warm-up epochs at 0.01, and the seq2seq model (4+4 layers, d=256, ff=1024,
8 heads, tied embeddings, 100 epochs, Adam lr=1e-3).

## Running

```bash
# Full paper sweep (train + beam-search eval for every grid cell)
sbatch NLP_2/run_paper.sh

# Single config manually
python3 NLP_2/train_paper.py --c 1.0 --n_q 4 --bins 256 --embed_dim 16 \
    --split_mode closure --save_dir checkpoint/nlp_2/myrun
python3 NLP_2/eval_paper.py --model_path myrun --epochs 100
```

Set `SPLIT_MODE=direct` in `run_paper.sh` (or `--split_mode direct`) for the
leak-free variant.
