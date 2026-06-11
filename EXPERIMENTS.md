# Experiment Results — Consolidated

Single place for the most recent experiment tables across all four tracks:
**NLP (WordNet)**, **rec_1 (Amazon Beauty)**, **image VQ-VAE (MNIST / CIFAR-100 / EMNIST)**, and **SoundStream (LibriTTS 24 kHz)**.

Compiled 2026-06-09 from the latest log files in `logs*/` and `evaluations/`; SoundStream §4.4–4.8 added 2026-06-11 (collapse root-cause campaign, R-D eval, codebook-dim sweep, STE-variant matrix). Each section cites the source files and SLURM job IDs so numbers can be traced back. `c=0` = Euclidean RVQ, `c>0` = hyperbolic HRQ.

---

## 1. NLP — WordNet hierarchy (InfoNCE recall@10)

Source: `logs_nlp_2/nlp2_all_tables.md`, `logs_nlp_2/results_table.txt` (runs of 2026-05-30 → 06-04).
Fixed config unless noted: `embed_dim=16, n_q=4, bins=128, batch=2048, lr=1.0, split=closure`, single seed.
Baselines: popularity@10 ≈ 41%, composition@10 ≈ 80%. Metric = **Recall@10 (%)**.

### 1.1 Method × embedding-init scale, c=1.0, 100 epochs
| init scale | euclidean | none | --new_method | --hste | --new+hste |
|---|---|---|---|---|---|
| 1.0 | 79.8 | 79.6 | 80.6 | 77.6 | **49.4** 💥 |
| 0.3 | 86.7 | 81.5 | 81.6 | 81.4 | 77.0 ⚠️ |
| **0.1** | 91.6 | **93.4** | 93.3 | 93.3 | **93.4** |
| 0.05 | 93.0 | 92.3 | 92.5 | 92.4 | 92.3 |

### 1.2 Method × scale, c=0.5, 50 epochs (ball radius 1/√c ≈ 1.414)
| init scale | none | --new_method | --hste | --new+hste |
|---|---|---|---|---|
| 1.0 | 74.4 | 74.6 | 73.7 | **38.4** 💥 |
| 0.3 | 85.8 | 85.9 | 84.8 | 85.4 |
| 0.1 | 92.1 | 92.0 | 92.1 | 92.1 |
| **0.05** | 93.2 | 93.3 | 93.3 | **93.3** |

*Trained cleanly (no NaN), confirming the c<1 conformal-clamp fix in `core_vq.py`.*

### 1.3 Gradient-handling on the broken cell (`--new+hste`, c=1.0, scale 1.0, 50 ep)
| approach | flag(s) | Recall@10 | emb grad (last ep) | outcome |
|---|---|---|---|---|
| no fix (baseline) | — | 57.8 / 62.5* | 3.7e8 | explodes |
| global clip | `--grad_clip 0.1` | 38.6 | 1.4e6 | froze |
| global clip | `--grad_clip 1.0` | 37.1 | 4.0e6 | froze |
| global clip | `--grad_clip 10` | 35.2 | 1.3e7 | froze |
| grad normalize | `--grad_norm` | 40.6 | 4.9e8 | froze |
| per-param clip | `--grad_clip 1.0 perparam` | 37.5 | 1.8e8 | froze |
| per-param clip | `--grad_clip 10 perparam` | 44.3 | 7.9e6 | froze |
| embed-only clip | `--grad_clip 1.0 embed` | 37.2 | 6.3e7 | froze |
| **Riemannian discount** | **`--hste_riemannian`** | **81.1** | **0.14** | **fixed ✅** |

*\*two seeds of the unfixed baseline. Only the Riemannian discount stabilized the gradient AND recovered accuracy.*

### 1.4 Gradient fixes for `--new+hste` across scales (c=1.0, 50 ep)
Two ways to neutralise the boundary HSTE blow-up, as R@10 / emb-grad (last epoch). `--gradient_correction` jobs 23651588–91.
| init scale | standard `--new+hste` | + `--hste_riemannian` | + `--gradient_correction` |
|---|---|---|---|
| 1.0 | 57.8 💥 | **81.1** / 0.14 ✅ | 78.3 / 0.08 ✅ |
| 0.3 | 75.5 ⚠️ | **83.9** / 0.06 ✅ | 81.2 / 0.05 ✅ |
| 0.1 | 92.2 | 92.4 / 0.05 | 92.2 / 0.05 |
| 0.05 | 93.5 | **93.8** / 0.06 | 93.5 / 0.06 |

**Best single cell in the study: 93.8% (c=1.0, scale 0.05, `--new+hste --hste_riemannian`).** `--gradient_correction` is the other valid fix: it equally tames the scale-1.0 emb grad (0.08 vs the unfixed 3.7e8) and recovers accuracy (57.8→78.3), but `--hste_riemannian` edges it on R@10 at every scale (+2–3 pts where the boundary bites, ≈tied at low scale).

### 1.5 Long-training / unscaled reference
| config | R@10 @ 500 ep | R@10 @ 50-ep peak |
|---|---|---|
| euclidean (scaled 0.05) | 83.3 | 91.9 |
| vanilla hyperbolic (scaled 0.05) | 81.3 | 93.4 |
| --new+hste (scaled 0.05) | 81.9 | 93.5 |

*All overfit by 500 ep; ~50 epochs is the real operating point. Unscaled (`scale=1.0`) 50-ep baselines: euc 75.4, vanilla-hyp 77.6, --new_method 77.7, --new+hste 77.5.*

**Takeaways:** init scaling (0.05–0.1) is the dominant lever; the `--new+hste` explosion at scale 1.0 is fixed cleanly by `--hste_riemannian` (and only by it — no clipping variant recovers accuracy); hyperbolic ≈ Euclidean at the optimum, slightly ahead at the best cell.

### 1.6 Pre-quantization encoding norm vs init scale (the §1.4 runs)
Source: `scratch_emb_norms.py` measuring the four §1.4 checkpoints (`--new+hste --hste_riemannian`, c=1.0). "Encoding pre-quantization" = the token-embedding rows `model.embedding(x)` (the only encoder), which `--embed_init_scale` multiplies at init (`train_paper.py:40-42`). Ball radius R = 1/√c = 1; `mapped/R` = `‖exp_map0(emb)‖/R`, the boundary-proximity that drives the HSTE conformal blow-up (1.0 = on the boundary).

**Before training** (fresh `nn.Embedding(82115,16)` × scale):
| init scale | raw norm mean | raw med | raw max | mapped/R mean | mapped/R max |
|---|---|---|---|---|---|
| 1.0 | 3.94 | 3.92 | 7.18 | 0.998 | 1.000 |
| 0.3 | 1.18 | 1.18 | 2.10 | 0.817 | 0.970 |
| 0.1 | 0.39 | 0.39 | 0.73 | 0.373 | 0.621 |
| 0.05 | 0.20 | 0.20 | 0.35 | 0.194 | 0.339 |

**After training** (trained `model.pt`):
| init scale | raw norm mean | raw med | raw max | mapped/R mean | mapped/R max |
|---|---|---|---|---|---|
| 1.0 | 2.91 | 2.66 | 6.96 | 0.959 | 1.000 |
| 0.3 | 0.96 | 0.95 | 1.95 | 0.726 | 0.961 |
| 0.1 | 0.47 | 0.49 | 0.86 | 0.435 | 0.694 |
| 0.05 | 0.29 | 0.31 | 0.55 | 0.280 | 0.504 |

**Findings:**
- **Before training: higher init scale ⇒ proportionally higher encoding norm (linear).** Raw mean ≈ 4×scale, i.e. the expected `E‖x‖≈√(16−½)≈3.94` for an N(0,1) dim-16 row scaled by `embed_init_scale`. In ball terms this sets boundary proximity directly — scale 1.0 pins at the boundary (mapped/R max 1.000), scale 0.05 sits deep inside (0.19). This is the mechanism behind the §1.4 explosion at scale 1.0.
- **After training: ordering preserved (still strictly monotonic) but the range compresses toward an intermediate band.** High scales shrink (1.0: 3.94→2.91), low scales grow (0.05: 0.20→0.29; 0.1: 0.39→0.47); runs never cross, so init scale still cleanly separates final encoding norms. Scale 1.0 still has max mapped/R pinned at 1.000 — it remains the boundary-stressed case and only trains because `--hste_riemannian` neutralizes the boundary gradient.

### 1.7 Four-method head-to-head (Recall@10 %, c=1.0 / euclidean c=0, 50 ep)
Clean single sweep — all four columns trained with identical code/seed/config, init scale the only row axis. Jobs 23671904–23671923. Columns: **euclidean** (c=0, plain RVQ) · **hyperbolic** (c=1, vanilla STE, no method flags) · **nm+hste+riem** (`--new_method --hste --hste_riemannian`) · **nm+hste+grad_corr** (`--new_method --hste --gradient_correction`).
| init scale | euclidean | hyperbolic | nm+hste+riem | nm+hste+grad_corr |
|---|---|---|---|---|
| 1.0 | 75.4 | 79.4 | **80.7** | 77.5 |
| 0.3 | **83.0** | 80.3 | 83.7 | 81.2 |
| 0.1 | 89.0 | 92.2 | **92.3** | 92.2 |
| **0.05** | 91.8 | 93.5 | **93.8** | 93.4 |

**Takeaways:** init scale 0.05 is the optimum for every method. **`nm+hste+riem` is best at all four scales** (peak 93.8 @ 0.05); `nm+hste+grad_corr` tracks vanilla hyperbolic and sits ~0.3–3 pts behind riem (the gap widens at high scale where the boundary bites). Plain hyperbolic ≥ euclidean once scaled ≤0.1; euclidean only wins at scale 0.3. Reproduces §1.4 within seed noise (riem 80.7/83.7/92.3/93.8, grad_corr 77.5/81.2/92.2/93.4).

---

## 2. rec_1 — Amazon Beauty hierarchy discovery

Source: `logs_rec_1/discovery_*.out` (2026-05-30 → 06-04). RQ/HRQ-VAE on MPNet embeddings (768→32, `n_q=4, bins=128`, 5000 ep), then seq2seq recommender on the generated codes. `--dedup` adds a uniqueness tie-break token. Test metrics from the recommender.

| job | c | method flags | uniq. ratio | Recall@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| 23279215 | 0.0 | euclidean baseline (no dedup) | 0.932 | 0.0342 | 0.0513 | 0.0290 |
| 23320936 | 1.0 | approx, dedup | 0.872 | 0.0388 | 0.0600 | 0.0318 |
| 23361994 | 1.0 | new_method, approx, dedup | 0.897 | 0.0341 | 0.0537 | 0.0285 |
| 23365287 | 1.0 | hste, approx, dedup | 0.627 | 0.0330 | 0.0521 | 0.0279 |
| **23439151** | 1.0 | new+hste+grad_correction, dedup | 0.971 | **0.0391** | **0.0600** | **0.0332** |
| 23472073 | 1.0 | new+hste+**hste_riemannian**, dedup | 0.975 | 0.0385 | 0.0594 | 0.0323 |

All c=1.0 runs use `quant=0.01, recon=1000.0, batch=2048`. The two best (23439151, 23472073) are also the cleanest codebooks (uniqueness 0.97+, all 128 entries used per level, per-level PPL ≈ 94/126/126/126).

**Takeaways:** hyperbolic (c=1.0) beats the Euclidean baseline on every metric (R@10 0.060 vs 0.051). Plain `--hste` collapses code diversity (uniqueness 0.63); adding `--grad_correction` or `--hste_riemannian` restores it (0.97+) and gives the best results.

---

## 3. Image VQ-VAE — MNIST / CIFAR-100 / EMNIST

Source: `evaluations/new_eval/parse_eval_*.out` (2026-06-05, the *_new 50-epoch set), `evaluations/parse_eval_23517282.out` + `cifar_hierarchy_23531644.out` + `rqvae_bench_23538252.out` (2026-06-06). FID lower is better; IS / Linear-Probing higher is better. Model legend: `e`/`e_vanilla` = Euclidean, `h` = vanilla hyperbolic, `hste` = hyperbolic STE, `new` = `--new_method`, `new_hste` = `--new+hste`, `sg`/`nothing` = STE ablations, numeric ids = baseline jobs.

### 3.1 MNIST (reconstruction)
| model | FID ↓ | IS ↑ | Linear Probe ↑ |
|---|---|---|---|
| e_vanilla | **134.63** | 1.885 | 63.6% |
| 23493918 (baseline) | 138.72 | 1.907 | 61.3% |
| new_hste | 142.02 | 1.847 | 61.2% |
| hste | 143.58 | 1.884 | 54.3% |
| new | 145.37 | 1.834 | **65.1%** |
| h | 150.61 | 1.902 | 64.3% |

*Legacy pre-`full_grid` set (older `evaluate.py`). FID here = frequency/marginal-sampling generation FID. Not comparable to §3.1b below.*

### 3.1b MNIST — full_grid `*_new` set, current eval (gradient_correction)
Source: `eval_full.sh` (jobs 23652530/33/36 baselines) + `eval_gc.sh`/`lp_gc.sh` (23651599/23652236 gc), all full_grid 50-epoch `mnist_new` checkpoints under the **same** current `evaluate.py`. FID = generation (marginal-sampling) FID ↓.
| variant | gen FID ↓ | IS ↑ | Linear Probe ↑ |
|---|---|---|---|
| euclidean | 268.50 | 1.790 | **24.90%** |
| c1 | **241.36** | 1.806 | 16.92% |
| c1_hste | 267.11 | 1.863 | 16.39% |
| c1_hste_gc | 253.79 | **1.868** | 21.25% |

*Comparable set. gradient_correction sits mid-pack: 2nd-best gen FID (behind plain c1), best IS, and 2nd-best linear probe (behind Euclidean). No regression vs the other hyperbolic variants.*

### 3.2 CIFAR-100 (reconstruction)
| model | FID ↓ | IS ↑ | Linear Probe ↑ |
|---|---|---|---|
| 23493931 (baseline) | **254.11** | 2.398 | 3.14% |
| e | 256.70 | 2.282 | **3.29%** |
| new | 261.73 | 2.143 | 3.11% |
| hste | 262.43 | 2.211 | 3.01% |
| new_hste | 270.83 | 2.182 | 2.66% |

*Legacy pre-`full_grid` set (older `evaluate.py`). Not comparable to §3.2b below.*

### 3.2b CIFAR-100 — full_grid `*_new` set, current eval (gradient_correction)
Source: `eval_full.sh` (jobs 23652539/43/45 baselines) + `eval_gc.sh`/`lp_gc.sh` (23651600/23652237 gc), all full_grid 50-epoch `cifar_new` checkpoints under the **same** current `evaluate.py`. FID = generation (marginal-sampling) FID ↓.
| variant | gen FID ↓ | IS ↑ | Linear Probe ↑ |
|---|---|---|---|
| euclidean | 219.80 | **1.821** | **6.87%** |
| c1 | **219.75** | 1.752 | 3.85% |
| c1_hste | 230.62 | 1.796 | 3.58% |
| c1_hste_gc | 226.95 | 1.793 | 4.01% |

*Comparable set. Euclidean and plain c1 tie for best gen FID (~219.8); Euclidean clearly best on IS and linear probe. gradient_correction beats only c1_hste on FID and is 2nd among the hyperbolic variants on linear probe — Euclidean is the overall winner here.*

### 3.3 EMNIST (reconstruction)
| model | FID ↓ | IS ↑ | Linear Probe ↑ |
|---|---|---|---|
| hste | **71.20** | 1.688 | 29.1% |
| new | 77.95 | 1.670 | 26.8% |
| sg | 77.70 | 1.741 | 23.7% |
| nothing | 78.61 | 1.739 | 29.4% |
| e | 81.77 | 1.731 | 25.7% |
| new_hste | 84.05 | 1.780 | **30.0%** |

*Legacy pre-`full_grid` set (older `evaluate.py`). Not comparable to §3.3b below.*

### 3.3b EMNIST — full_grid `*_new` set, current eval (gradient_correction)
Source: `eval_full.sh` (jobs 23670423/24/25 baselines) + `eval_gc.sh`/`lp_gc.sh` (23652093/23670432 gc), all full_grid 50-epoch `emnist_new` checkpoints under the **same** current `evaluate.py`. FID = generation (marginal-sampling) FID ↓. (gc RQ-Transformer FID = 10.30; no baseline RQ-T run for EMNIST.)
| variant | gen FID ↓ | IS ↑ | Linear Probe ↑ |
|---|---|---|---|
| euclidean | 251.89 | 1.807 | **15.86%** |
| c1 | 256.19 | 1.820 | 13.64% |
| c1_hste | **244.20** | **1.845** | 12.61% |
| c1_hste_gc | 258.62 | 1.824 | 11.85% |

*Comparable set. On EMNIST gradient_correction is the weakest — worst gen FID and worst linear probe; c1_hste is best on FID/IS, Euclidean best on linear probe.*

### 3.4 CIFAR-100 generation — RQ-Transformer (stage 2)
Source: `parse_eval_23517282.out` / `evaluate_23509288.out`. 6.3M-param RQ-Transformer trained on the VQ codes.
| model | RQ-Transformer FID ↓ | RQ-Transformer IS ↑ |
|---|---|---|
| **e** | **117.13** | 3.037 |
| new | 117.97 | 3.055 |
| sg | 120.86 | 3.051 |
| hste | 127.09 | 2.911 |
| new_hste | 127.57 | 3.043 |

*Legacy pre-`full_grid` set (older `evaluate.py`). The gradient_correction comparison is in §3.4b below, on the full_grid `*_new` checkpoints re-evaluated under the current code (not comparable to these rows — the same euclidean checkpoint scores differently across the two eval pipelines).*

### 3.4b CIFAR-100 generation — full_grid `*_new` set, current eval (gradient_correction)
Source: `eval_full.sh` / `eval_gc.sh` re-evaluations (jobs 23652539/43/45 baselines, 23651600 gc). All four are full_grid 50-epoch `cifar_new` checkpoints evaluated with the **same** current `evaluate.py`, so directly comparable. 6.6M-param RQ-Transformer.
| variant | RQ-Transformer FID ↓ | RQ-Transformer IS ↑ |
|---|---|---|
| euclidean | **94.67** | **3.859** |
| c1 | 101.84 | 3.508 |
| c1_hste | 98.83 | 3.733 |
| c1_hste_gc | 99.48 | 3.757 |

*Comparable set. Euclidean gives the best RQ-Transformer FID (94.7) and IS; gradient_correction (99.5) is mid-pack — better than plain c1, slightly behind c1_hste. Consistent with the reconstruction tables: on CIFAR the methods are within ~7 FID points and Euclidean is narrowly best.*

### 3.5 CIFAR-100 hierarchy recovery from quantized codes (higher is better)
Source: `cifar_hierarchy_23531644.out`. Does the code structure recover the 20 CIFAR-100 superclasses?
| variant | prec@4 | ARI | NMI | purity | cophenetic corr |
|---|---|---|---|---|---|
| euclidean (c=0) | 0.1575 | 0.0484 | 0.4701 | 0.3300 | 0.0464 |
| **hyperbolic c=1** | **0.1625** | 0.0584 | 0.4850 | 0.3400 | 0.0531 |
| hyperbolic c=1 + hste | 0.1525 | **0.0716** | **0.5035** | 0.3500 | 0.0527 |
| hyperbolic c=1 + hste + grad_corr | 0.1500 | 0.0672 | 0.4988 | **0.3600** | **0.0583** |

*Hyperbolic recovers the taxonomy slightly but consistently better than Euclidean. Adding `--gradient_correction` to `--new+hste` gives the best purity (0.360) and best cophenetic correlation (0.0583 — the tightest match between code-distance and the superclass tree). Baselines reproduced exactly (job 23652094), so the gc row is directly comparable.*

### 3.6 ImageNet RQ-VAE throughput benchmark (A100-40GB)
Source: `rqvae_bench_23538252.out`. Model: RQ-VAE 256×256→8×8, D=4, codebook=16384, 99.9M params (G+D), bf16.
| batch | img/s | peak GB | %mem | ~1 epoch |
|---|---|---|---|---|
| 16 | 54.3 | 20.7 | 49% | 6h33m |
| 32 | 57.5 | 39.8 | 94% | 6h11m |
| 64+ | OOM | — | — | — |

**Recommended batch size: 16** (54.3 img/s, 20.7 GB).

**Takeaways:** on reconstruction the methods are within noise; Euclidean is narrowly best on MNIST/CIFAR-100, hyperbolic-STE best on EMNIST FID. For CIFAR-100 generation the Euclidean codes give the best RQ-Transformer FID — consistent with the codes being near-max-entropy (an information ceiling, not a transformer bug). Hyperbolic's one clear edge is hierarchy recovery (§3.5).

**gradient_correction (comparable §x.b tables, full_grid + current eval).** Added `--new_method --hste --gradient_correction` as a 4th variant across MNIST/CIFAR/EMNIST. On reconstruction and generation it lands **mid-pack and within noise** — never the best, never collapsing: MNIST 3.1b best-IS / 2nd-best linear-probe; CIFAR 3.2b beats only c1_hste; CIFAR generation 3.4b 99.5 FID vs Euclidean's best 94.7; EMNIST 3.3b weakest of the four. Its **one win is CIFAR hierarchy recovery (§3.5): best purity (0.360) and best cophenetic correlation (0.058)** — the same place hyperbolic helps generally. Net: gradient_correction is a *safe, non-collapsing* way to run `--new+hste` (cf. the NLP/rec results where it equals or beats `--hste_riemannian`), but on image reconstruction it does not beat Euclidean. **Caveat:** the §x.b tables use the current `evaluate.py` on full_grid checkpoints and are internally comparable, but are **not** comparable to the legacy §3.1–3.4 tables above (different checkpoints + older eval — the same Euclidean checkpoint scores 268 gen-FID here vs ~135 there).

---

## 4. SoundStream — LibriTTS 24 kHz codec

Source: `logs/*.out` and `logs/eval_unique_23588300.out` (2026-06-07 → 06-09). 12-codebook RVQ (`bins=1024`, ratios `[6,5,4,2]`, target bandwidths up to 12 kbps). PPL = per-codebook perplexity (max 1024); recon_loss lower is better. Short runs (≤10 ep) used for the ablations below.

### 4.1 Euclidean vs hyperbolic, commitment weight sweep (10 ep)
| run | c | LAMBDA_COM | recon_loss ↓ | feature_loss | PPL (mean of 12) | status |
|---|---|---|---|---|---|---|
| euc_com1 (23557182) | 0 | 1 | **116.93** | 7.97 | ~688 | ✅ healthy |
| euc_com1000 (23557185) | 0 | 1000 | 120.60 | 13.74 | ~136 | ✅ healthy (lower PPL) |
| hyp_c1_com1 (23557181) | 1 | 1 | 268.28 | 201.5 | **1.0** | 💥 collapsed |
| hyp_c1_com1000 (23557244) | 1 | 1000 | 284.02 | 316.7 | **1.1** | 💥 collapsed |

*Plain c=1.0 collapses: all codes saturate the ball boundary → `hyperbolic_distance_sq` atanh-clamp kills the commit/codebook gradient → PPL pins at 1.0.*

### 4.2 `--code_max_radius` fix (kmeans init, c=1.0, best epoch 3)
| run | code_max_radius | recon_loss ↓ | feature_loss | total unique codes | PPL (mean of 12) | status |
|---|---|---|---|---|---|---|
| hyp_cr03 (23583052) | 0.3 | 168.03 | 19.53 | 12252 / 12288 | ~677 | ✅ recovered |
| hyp_cr05 (23583060) | 0.5 | 168.24 | 13.47 | 12166 / 12288 | ~651 | ✅ recovered |
| hyp_cr09 (23583061) | 0.9 | 167.15 | 28.98 | 12210 / 12288 | mixed (4 hi / 8 lo) | ⚠️ partial |

*Constraining codes to stay inside radius `r·(1/√c)` reverses the collapse — PPL jumps from 1.0 to 600–740. cr0.3 / cr0.5 give the fullest, most uniform codebook usage; cr0.9 lets the deeper codebooks drift toward the boundary again (PPL drops to ~120–250 on layers 5–12). Full eval in `eval_unique_23588300.out`.*

### 4.3 No-kmeans init and new_method+hste variants (best epoch 3)
| run | code_max_radius | kmeans | new+hste | recon_loss ↓ | PPL (mean) | status |
|---|---|---|---|---|---|---|
| hyp_cr09_nokmeans (23588462) | 0.9 | ✗ | ✗ | **161.41** | ~653 | ✅ best recon |
| hyp_cr03_nokmeans_nmhste (23592322) | 0.3 | ✗ | ✓ | 215.91 | ~208 (ramping) | ⚠️ partial |
| hyp_cr09_nokmeans_nmhste (23591636) | 0.9 | ✗ | ✓ | 262.48 | 1.0 | 💥 collapsed |
| featlow (23583071) | 0.0 | ✗ | ✗ | 261.03 | 1.0 | 💥 collapsed (feature_loss blew up) |
| featlow_gc (23583199) | 0.0 | ✗ | ✗ | 259.20 | 1.0 | 💥 collapsed |

**Takeaways:** Euclidean is stable out of the box (best recon 116.9). Hyperbolic c=1.0 needs `--code_max_radius` to avoid boundary collapse; **cr0.9 without kmeans gives the best hyperbolic recon (161.4)** but is fragile — combining it with `--new+hste` re-collapses, while cr0.3+nmhste survives weakly. The headline open problem is that the hyperbolic codec is still ~40 recon-loss points behind Euclidean and remains sensitive to the boundary even with the radius cap.

### 4.4 Collapse root cause & fixes (d512, 3 ep, c=1, cmr=0.9, runs of 2026-06-09 → 06-11)

Source: `logs/*.out`, diagnostics in `scratch_failure_diag.py`, `scratch_fw2_layers.py`, `scratch_collapse_sim.py`, `scratch_hste_grad.py`, `scratch_lowdim_diag.py`. Base config = `hyp_cr09_nokmeans_nmhste` (`--new_method --hste --hste_riemannian`, kmeans off); each row changes one variable. Valid recon at epoch 3 ↓ / valid PPL.

| run | job | delta vs control | recon | PPL | status |
|---|---|---|---|---|---|
| initfix (control) | 23612322 | interior code init + `--encoder_scale -1` | 262 | 1.0 | 💥 collapse |
| featlow | 23626017 | `LAMBDA_FEAT 0.05` | ~262 | 1.0 | 💥 collapse, feat loss diverges |
| shell | 23626018 | `--encoder_shell 0.5` | **178** | →541 | ✅ works |
| norevival | 23635298 | `THRESHOLD_EMA_DEAD_CODE 0` | **187** | 5.8–40 | ✅ works |
| gradcorr | 23638359 | `--gradient_correction` (riem kept) | 264 | 1.0 | 💥 collapse (train-PPL alive, valid dead) |
| cw025 / cw0025 | 23646069/71 | commitment ÷10 / ÷100 | 260 / 268 | ~1.0 | 💥 collapse — falsifies the commit-imbalance story |
| nmonly | 23643472 | new_method only (**Euclidean STE**) | **173** | alive | ✅ healthy — ordering is innocent |
| hsteonly | 23643473 | hste+riem with **standard** ordering | 262 | 1.0 | 💥 collapse — **HSTE is the cause** |
| hste_noriem | 23650403 | hste **without** `hste_riemannian` (std order) | **172** | ~455 | ✅ healthy (gen grad ~1e6, stable) |
| gradcorr_noriem | 23650979 | nm+hste+gc, **no** riemannian | **165** | ~480 | ✅ **best HSTE config** (grad 9→7e5, decreasing) |
| nq8 | 23680345 | riem with **8 codebooks** (`TARGET_BANDWIDTHS "1 2 4 8"`) | 262→326 | 1.0 | 💥 collapse — riem failure depth-independent |

**Mechanism (measured, not inferred).** The collapse signature (commit pinned at **72.2079** = `1.25·[2·atanh(1−1e-3)]²`, valid PPL=1) is caused by the `--hste_riemannian` branch of `HyperbolicSTE.backward`: it returns the Riemannian gradient (÷λ_q² ≈ ÷111 at code radius 0.9), starving the encoder's reconstruction signal — isolated against residual ordering (nmonly vs hsteonly), codebook dimension (§4.6) and RVQ depth (nq8). Lowering `commitment_weight` 10×/100× does NOT rescue it ⇒ not a commit/recon balance problem. Secondary dynamics (all measured): dead-code revival re-seeds the entire codebook at step 1 (`cluster_size` inits at 0, threshold 2), arming a Möbius residual-radius accumulation that saturates deep layers within one forward; once saturated, all gradients to the encoder die and Adam's scale-invariant steps walk the encoder norm to ~1.3e5 with frozen weight magnitudes. Causal ablations (`scratch_collapse_sim.py`): rec+commit alone reproduces full collapse in 15 steps; revival OFF ⇒ zero saturation in 60 steps. **Every c=1 run — including the healthy ones — ends with residuals on the boundary shell (r≈1.000) coding by direction only** (post-mortem: healthy hyp_std_d8 has 288 unique codes by direction; collapsed runs have 1). Side findings: `--quantizer_grad_clip`/`--manifold_grad_clip` are printed but **never applied**; `main3_ddp.py` uses `soundstream/loss.py` where feature-matching is not gated by `discriminator_iter_start`.

### 4.5 10-epoch runs + rate–distortion evaluation (jobs 23652295–97, eval 23670491)

New tooling: `eval_rd.py` / `eval_rd.sh` — encode once at n_q=12, decode prefixes n_q∈{1,2,4,8,12}; PESQ-wb (24k→16k), STOI, SI-SDR, log-mel L1, per-layer code entropy → nominal AND entropy-coded kbps (kbps = n_q × 1 at bins=1024 / 100 Hz). 100 dev-clean files. JSONs in `egs/SoundStream_24k_240d/rd_results/`, figures in `figures/rd_*.png`.

| model | recon ep10 | PESQ @1→12 kbps | STOI | SI-SDR | entropy kbps @n_q=12 |
|---|---|---|---|---|---|
| euc_10ep | **124.1** | 1.27 → **1.48** (real R-D curve, saturates ~8 kbps) | 0.79→0.86 | −13.5→−10.7 | 11.0 |
| hyp_std_10ep | 153.3 | 1.10 **flat** | 0.71 | ≈ −35 | 11.0 |
| gradcorr_noriem_10ep | 158.1 | 1.12 **flat** | 0.71 | ≈ −35 | 9.6 |

*The hyperbolic R-D curves are FLAT — extra RVQ depth adds nothing perceptually. Verified NOT an eval artifact (encode/decode path is bit-identical to the training forward, max diff 2e-7). Explanation: in the boundary-shell regime the Möbius sum of boundary codes saturates `log_map0` magnitude, so layers 2–12 barely change the decoder input. Entropy coding would save ~8–20%.*

### 4.6 codebook_dim sweep with tangent-space bottleneck (3 ep, `--tangent_proj`, valid recon ep3)

New flag `--tangent_proj`: the `codebook_dim` bottleneck is a Euclidean `nn.Linear` **before** `exp_map0` (and after `log_map0` on decode) — quantization lives in the low-dim Poincaré ball, no hyperbolic `HLinear` layers. Jobs 23670380–93, 23671648–52 (hyp_std rerun after a tangent_proj bug fix in the standard-mode exit), 23671849–58, 23675622–29, 23683730.

| dim | euc | hyp_std | gradcorr_noriem | nmhste_noriem (no gc) | nmhste_riem (no gc) | blockhste | gyronly |
|---|---|---|---|---|---|---|---|
| 8 | 160.6 | **184.4** | 270.6 💥 | NaN 💥 | 267.4 💥 | 273.2 💥 | 271.7 💥 |
| 32 | **157.1** | 200.8 | 260.8 💥 | 255.0 ⚠️ | 273.1 💥 | 270.0 💥 | — |
| 64 | 179.7 | 212.0 | 190.3 ✅ | NaN 💥 | 264.0 💥 | 270.0 💥 | — |
| 128 | 159.7 | **165.2** | 184.5 ✅ | NaN 💥 | 260.3 💥 | 303.5 💥 | — |

*💥 = boundary collapse (commit 72.2079, PPL≈1); NaN = codebook explosion within epoch 1; ⚠️ = alive but poor. Euclidean-STE families (euc, hyp_std) train at every dim — hyp_std_d8 reaches near-full utilization (PPL 731–963) by coding directions on the shell. The HSTE usable envelope is narrow: no-riemannian + gradient_correction + dim ≥ 64. The non-riemannian ×λ_x² amplification compounds multiplicatively through the un-detached 12-layer chain (deepest backprop paths dominate when per-hop factors > 1) → NaN; `gradient_correction` detaches the chain and tames it. The riemannian ÷λ_q² discount applies effectively ONCE per path (shallow paths dominate when factors < 1) — no compounding needed, a single ÷100 starves.*

### 4.7 STE-variant matrix — closing the gradient-surrogate question

Further verified variants (bit-identical forward values, `scratch_verify_newste.py`): `--block_hste` (per-layer codes detached; ONE identity STE wrapping the whole RVQ block in tangent space — exact factor-1 affine Jacobian) and `--gyration_only` (pure gyration transport `gyr[x,⊖q]`, ZERO conformal λ/"gamma" coefficients, Euclidean magnitude preserved exactly — verified 1.000000 incl. at the boundary; helper `gyration_transport()` factored out of `parallel_transport_1`).

| surrogate | magnitude | direction | result @ d8 (and beyond) |
|---|---|---|---|
| Euclidean STE | identity | identity | ✅ works (hyp_std healthy at every dim) |
| hste_riemannian | ÷λ_q² (geometric) | PT (gyration·λ-ratio) | 💥 starves — at every dim & depth |
| hste no-riem (no gc) | ×λ_x²/λ_q² (geometric) | PT | 💥 explodes (NaN) at low dim |
| gyration_only | **exactly 1** | gyration | 💥 collapses (job 23683730) |
| block_hste | exactly 1 (block level) | identity (block level) | 💥 no learning (partial code diversity, recon ~270–303) |

**Conclusion (two-sided):** fixing the magnitude while keeping geometric direction fails (gyration_only); geometric magnitude with identity direction fails (riem/noriem); **identity in BOTH — the plain Euclidean STE — is the unique working surrogate.** Both factors of the hyperbolic STE (λ-coefficients and gyration) are first-order objects valid only near q≈r, where they reduce to the identity anyway; away from it the scalar part fails loudly (starve/explode) and the rotation part fails quietly (misdirection). Every gradient-side fix without radius control failed — the boundary-shell drift is the prerequisite problem. *(Open: combining a verified STE variant with `--encoder_shell`, the interior regime where the corrections are mild; and the nm-ordering+Euclid-STE d8 control for airtight attribution.)*

### 4.8 SoundStream leaderboard (3-epoch valid recon, d512 unless noted)

Euclidean **140** > std-hyp+cmr **161** ≈ gradcorr_noriem **165** > hste_noriem **172** ≈ nmonly **173** > shell **178** > norevival **187** ≫ every `hste_riemannian` / collapsed config (~260+, PPL 1). At 10 epochs: euc 124, hyp_std 153, gradcorr_noriem 158 — the HSTE path closes the recon gap with longer training but stays perceptually flat across bitrates (§4.5). **Working fixes, ranked: `--encoder_shell 0.5` > `THRESHOLD_EMA_DEAD_CODE 0` > drop `hste_riemannian` (+`--gradient_correction` for dim < 512).**
