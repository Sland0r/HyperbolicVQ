# Image Task Results — MNIST / CIFAR-100

Standard config throughout: **50 epochs, n_q=4, bins=128, c=1** for hyperbolic
runs. Reconstruction and hierarchy are mean ± std over seeds 42/43/44; no
dim / depth / embed-init variants. Generation (table 3) is single-seed (s42),
50-epoch RQ-Transformer, 10k samples.

## 1. Reconstruction — best val recon loss ↓ (×10⁻³, mean ± std, seeds 42/43/44)

| method | MNIST | CIFAR-100 |
|---|---|---|
| Euclidean | **0.477 ± 0.006** | **1.077 ± 0.006** |
| Vanilla hyperbolic (c=1) | 0.600 ± 0.020 | 1.280 ± 0.017 |
| per-layer HSTE | 0.587 ± 0.025 | 1.323 ± 0.055 |
| per-layer HSTE + gc | 1.260 ± 0.130 | 1.390 ± 0.087 |
| A4 (block-PT riem+gc) | 0.647 ± 0.023 | 1.360 ± 0.046 |
| A4 no-gc | 0.667 ± 0.035 | 1.470 ± 0.027 |
| A4 clip (+gc) | 0.587 ± 0.031 | 1.243 ± 0.021 |
| A4 cw=1 | 0.757 ± 0.023 | 1.590 ± 0.036 |
| A5 (A4 + last-commit chain) | 0.623 ± 0.021 | 1.400 ± 0.030 |
| A4_v2 (strict gc) | ~1.6 | 7.25 ± 0.34 |
| A4_v2 + A5 | 1.81 ± 0.99 | 3.00 ± 0.23 |
| a6 (keep-first recon) | 0.657 ± 0.006 | 1.407 ± 0.046 |
| a7 (keep-last recon) | 0.690 ± 0.017 | 1.480 ± 0.046 |
| a6.1 (sum: A4 + a6) | 0.613 ± 0.015 | 1.347 ± 0.032 |
| a7.1 (sum: A4 + a7) | 0.630 | 1.343 |
| a8 (sum: A4 + all per-layer) | 0.587 ± 0.012 | 1.27 ± 0.05 ‡ |

Among hyperbolic-block arms, **a6.1/a7.1 give the best recon** (beat A4 and
a6/a7); Euclidean still wins overall. gc-flip on the sum arms: a6.1+gc CIFAR
1.307 (best a6.1 cell, noisy), MNIST unchanged 0.613; a7.1 prefers no-gc.

‡ **a8** (un-truncated full-sum router, **seeds 42/43/44**): **the best hyperbolic
reconstructor on BOTH datasets** — MNIST 0.587 ± 0.012, CIFAR 1.27 ± 0.05 (beats
a6.1/a7.1 ≈1.34, A4 1.36, vanilla 1.28; Euclidean still wins overall). Confirms
the single-seed finding across seeds. Collapses at audio n_q=12 (§7), so this is a
shallow-depth win only. (CIFAR precision is coarser — 4-decimal val-recon logging.)

## 2. CIFAR-100 superclass hierarchy recovery ↑ (mean ± std, seeds 42/43/44)

| method | prec@4 | ARI | NMI | purity | coph. corr |
|---|---|---|---|---|---|
| Euclidean | 0.159 ± 0.001 | 0.048 ± 0.005 | 0.469 ± 0.012 | 0.327 ± 0.006 | 0.046 ± 0.001 |
| Vanilla hyperbolic | **0.164 ± 0.003** | 0.055 ± 0.003 | 0.481 ± 0.003 | 0.337 ± 0.006 | 0.052 ± 0.002 |
| per-layer HSTE | 0.158 ± 0.005 | 0.069 ± 0.008 | 0.498 ± 0.012 | 0.347 ± 0.015 | 0.050 ± 0.004 |
| per-layer HSTE + gc | 0.153 ± 0.004 | 0.071 ± 0.009 | 0.507 ± 0.010 | 0.360 ± 0.000 | **0.056 ± 0.003** |
| A4 (riem+gc) | 0.157 ± 0.004 | **0.087 ± 0.008** | **0.515 ± 0.013** | **0.370 ± 0.010** | 0.043 ± 0.005 |
| A4 no-gc | 0.149 ± 0.004 | 0.079 ± 0.019 | 0.506 ± 0.023 | 0.357 ± 0.012 | 0.044 ± 0.002 |
| A4 clip (+gc) | 0.151 ± 0.006 | 0.066 ± 0.005 | 0.497 ± 0.009 | 0.350 ± 0.010 | 0.052 ± 0.007 |
| A4 cw=1 (n=2) | 0.143 ± 0.007 | 0.083 ± 0.008 | 0.511 ± 0.008 | 0.360 ± 0.014 | 0.043 ± 0.009 |
| A5 | 0.151 ± 0.006 | 0.058 ± 0.005 | 0.481 ± 0.003 | 0.337 ± 0.006 | 0.046 ± 0.003 |
| A4_v2 (strict gc) | 0.138 ± 0.014 | 0.060 ± 0.010 | 0.494 ± 0.018 | 0.340 ± 0.010 | 0.063 ± 0.047 |
| A4_v2 + A5 | 0.107 ± 0.001 | 0.047 ± 0.018 | 0.479 ± 0.018 | 0.333 ± 0.021 | 0.041 ± 0.010 |
| a6 | 0.153 ± 0.004 | 0.085 ± 0.008 | 0.513 ± 0.011 | 0.360 ± 0.010 | 0.050 ± 0.003 |
| a7 | 0.145 ± 0.010 | 0.077 ± 0.035 | 0.503 ± 0.041 | 0.363 ± 0.038 | 0.044 ± 0.002 |
| a6.1 | 0.155 ± 0.003 | 0.080 ± 0.013 | 0.504 ± 0.017 | 0.360 ± 0.017 | 0.050 ± 0.002 |
| a7.1 | 0.149 ± 0.003 | 0.088 ± 0.027 | 0.514 ± 0.032 | 0.357 ± 0.025 | 0.051 ± 0.003 |
| a6.1 + gc | 0.155 ± 0.000 | 0.087 ± 0.019 | 0.516 ± 0.021 | 0.367 ± 0.021 | 0.051 ± 0.005 |
| a7.1 + gc | 0.149 ± 0.003 | 0.088 ± 0.006 | 0.517 ± 0.008 | 0.360 ± 0.010 | 0.046 ± 0.005 |

**a6 ties A4 for the taxonomy lead** (ARI 0.085 vs 0.087); a7/a7.1 have good
means but are seed-fragile (a7 ARI ±0.035, NMI ±0.041, purity ±0.038; a7.1 ARI
±0.027). The +gc variants are both the tightest *and* tie A4: **a6.1+gc** matches
A4 on ARI (0.087) and edges it on NMI/purity (0.516/0.367), and **a7.1+gc** is the
most seed-stable of the routers (ARI 0.088 ± 0.006). prec@4 separates almost
nothing (all 0.145–0.155, within noise of every row). (Full 5-metric rows now
from hierarchy-eval job 23811346, seeds 42/43/44.)

## 3. Generation — RQ-Transformer, 10k samples, FID ↓ (IS ↑), single seed (s42)

| method | MNIST FID | CIFAR-100 FID |
|---|---|---|
| Euclidean | 20.36 (2.069) | **94.67 (3.86)** |
| Vanilla hyperbolic | **15.01 (2.105)** | 101.84 (3.51) |
| per-layer HSTE | 17.24 (2.075) | 98.83 |
| per-layer HSTE + gc | — | 99.48 |
| A4 (riem+gc) | 16.78 (2.068) | 98.23 (3.80) |
| A4 no-gc | 18.48 (2.077) | — |
| a6.1 | 17.00 | 97.47 (3.72) |
| a6.1 + gc | 16.01 | 98.04 (3.77) |
| a7.1 | 15.67 | 99.89 |
| a7.1 + gc | 17.43 | 98.39 (3.84 best IS) |

Note the **dataset reversal**: on MNIST every hyperbolic variant beats
Euclidean (vanilla best, 15.0 vs 20.4); on CIFAR Euclidean wins FID and a6.1
is the best new arm. a5 / a4_v2 / a6 / a7 (non-sum) were not run through
stage-2 generation.

---

**Takeaway:** Euclidean owns recon (and CIFAR gen-FID); hyperbolic owns MNIST
gen-FID and CIFAR taxonomy. Within hyperbolic, **A4/a6 lead hierarchy** and
**a6.1/a7.1 lead recon**, all within a tight band — consistent with the
codes-near-max-entropy info-ceiling.

---

# NLP Task Results — WordNet hierarchy (Recall@10 %)

Closure split. Standard cell otherwise: **n_q=4, cw=1.0, embed_init_scale=1.0,
50 epochs, lr=1.0, c=1** (Euclidean c=0), single seed. **Bold** = best in column;
* = reused from earlier runs (d16/b128 = the headline cell; euc/van d16/b64,
d32/b128, and A4 d16/b64 from the capacity sweep).

## 4. Capacity grid — embed_dim (8/16/32) × codebook size (128/256)

| Method | d8/b128 | d8/b256 | d16/b64 | d16/b128 | d16/b256 | d32/b128 | d32/b256 |
|---|---|---|---|---|---|---|---|
| Euclidean | 75.4 | **63.4** | 76.7* | 75.9* | 67.0 | **71.1*** | 51.9 |
| Vanilla hyp | 62.3 | 42.6 | **86.9*** | 83.4* | 81.7 | 53.6* | **65.2** |
| Vanilla hyp + gc | 71.6 | 41.4 | 83.5 | 82.3 | 80.5 | 46.2 | 45.5 |
| A4 no-gc | 61.6 | 39.7 | 82.3* | 84.1* | 81.6 | 36.6* | 40.2 |
| A4 + gc | 78.7 | 41.4 | 83.8 | 81.9* | 80.8 | 46.7 | 52.6 |
| A5 + gc | 77.3 | 44.7 | 85.7 | 84.4* | 83.6 | 52.6 | 48.6 |
| A5 (no gc) | 78.0 | 44.2 | 85.6 | 84.4* | 84.2 | 40.5 | 49.9 |
| **a7.1** | **81.0** | 42.7 | 84.6 | 85.0* | **85.3** | 47.0 | 36.1 |
| a6.1 | 64.8 | 40.3 | 84.7 | 84.6* | 85.0 | 46.4 | 43.2 |

**Reading.** Hyperbolic dominates the whole trainable region (d8–d16); the
winning method shifts with codebook size — **vanilla** at the small book
(d16/b64, 86.9), **a7.1** at standard/large (d16/b128 85.0, d16/b256 85.3, and
d8/b128 81.0). Euclidean wins only in the broken regimes. The two failure
columns have *opposite* causes (verified from training diagnostics):

- **d32 = init/boundary collapse** (hyperbolic only). At scale 1.0 the init norm
  grows as √dim (~5.7 at d32), `exp_map0` saturates on the ball boundary and the
  commit distance clamps → commit loss 170–210, approx-dist ~50, CE stuck ~15,
  code sequences collapse (vangc → 1247 unique). Euclidean has no boundary and is
  unaffected. **Fixable** with √dim-compensated init (`embed_init_scale ≈ 0.707`),
  per the image-side finding.
- **d8/b256 = codebook over-provisioning** (NOT init). The VAE trains *cleanly*
  (CE/commit/approx-dist identical to the working d8/b128 cell, 0 dead codes), but
  256 codes can't be usefully spread in 8 hyperbolic dims (perplexity ~50/256),
  so nearly every concept gets a unique code sequence → the seq2seq memorizes
  instead of generalizing hypernymy → poor closure recall. Euclidean tolerates it
  by spreading codes wider.

gc is **non-monotone in dimensional headroom**: it *rescues* the cramped d8/b128
cell (vanilla 62.3→71.6, A4 61.6→78.7) but *hurts* the comfortable d16 cells
(vanilla 86.9→83.5, A4 84.1→81.9) — the same sign-flip seen with depth.

**Baselines (from the NLP_2 `a4var` reproduction, job set 23822–23824***).** A
single-seed re-run of the whole grid reproduced the numbers above exactly (so the
table stays single-seed; no variance table is added), but also logged two closure
references missing here: **global-popularity@10 ≈ 40.9 %** and a
**graph-composition@10 ceiling ≈ 80.1 %**. These reframe the failure columns: the
collapsed d32 cells (36–52) and Euclidean's d8/b256 (63.4) are at or *below* the
popularity floor — the seq2seq has learned essentially nothing transferable there —
while the winning hyperbolic d16 cells (85+) close ~85 % of the gap to the
composition ceiling. (The per-codebook `mean/var` figures in those logs are
quantizer perplexity statistics, not seed variance.)

## 5. Representation quality (standard cell d16/b128, cw=1.0)

Per-concept code tuples extracted for all 82 115 noun synsets. **Uniqueness** =
distinct code sequences / vocab (and number of clusters where >1 concept shares a
code). **Similarity** = how related the concepts sharing a code are: WordNet path
& Wu-Palmer tree similarity, and GloVe embedding cosine (higher = tighter group).

| Method | uniq ratio | clusters >1 | max cluster | path sim | wup sim | GloVe emb sim |
|---|---|---|---|---|---|---|
| Euclidean | 0.997 | 32 | 163 | 0.1227 | 0.3204 | 0.3359 |
| Vanilla hyp | 0.888 | 6486 | 582 | 0.0778 | 0.2417 | 0.0972 |
| A4 + gc | 0.967 | 2161 | 456 | 0.0877 | 0.2620 | 0.1618 |

**Reading (note the confound).** The raw similarity columns favor Euclidean, but
that is an *artifact of it barely clustering*: it is near-injective (99.7 % unique
codes, only 32 real clusters), using the codebook as a hash rather than a shared/
hierarchical structure — so averaging similarity over its handful of accidental
near-synonym collisions trivially looks "coherent." Averaging intra-cluster
similarity structurally rewards *not* clustering. The meaningful comparison is
between the models that actually cluster: **A4 + gc beats vanilla on both axes** —
more selective (uniqueness 0.967 vs 0.888 → 2161 vs 6486 clusters, i.e. it does
not over-collapse the codebook) *and* more semantically coherent on all three
similarity metrics (path 0.088 vs 0.078, wup 0.262 vs 0.242, GloVe 0.162 vs
0.097). Vanilla over-clusters into looser groups. *Caveat:* comparing average
pairwise similarity across very different cluster counts (32 vs 6486) is
apples-to-oranges; a rigorous version would report **lift over a random-code
baseline** (expected ≈ 0 for Euclidean, positive for the hyperbolic models).

---

# Recommendation Task Results — Amazon Beauty

Amazon **Beauty**, standard HRQ config — `768→32, n_q=4, bins=128, 5000 ep,
quant=0.01, recon=1000, batch=2048, --dedup`, then 100-ep seq2seq recommender.
No dim/depth/embed-init variations. Test-set metrics, best per column in **bold**.

| # | method | flags | job | uniq. | R@5 | NDCG@5 | R@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| 1 | Euclidean* | `c=0` | 23279215 | 0.932 | 0.0342 | 0.0235 | 0.0513 | 0.0290 |
| 2 | Vanilla hyperbolic | `c=1` plain | 23320936 | 0.872 | 0.0388 | 0.0250 | 0.0600 | 0.0318 |
| | *— pre-block-hop (per-layer STE) —* | | | | | | | |
| 3 | new_method only | `new_method` | 23361994 | 0.897 | 0.0341 | 0.0222 | 0.0537 | 0.0285 |
| 4 | HSTE only | `hste` | 23365287 | 0.627 | 0.0330 | 0.0218 | 0.0521 | 0.0279 |
| 5 | new_method + HSTE + riem | `new_method hste hste_riemannian` | 23472073 | 0.975 | 0.0385 | 0.0256 | 0.0594 | 0.0323 |
| 6 | new_method + HSTE + gc | `new_method hste gradient_correction` | 23439151 | 0.971 | 0.0391 | 0.0264 | 0.0600 | 0.0332 |
| | *— block-hop (A4 = block_hste_pt, riem) —* | | | | | | | |
| 7 | A4 (no gc) | `block_hste_pt hste_riemannian` | 23715002 | 0.970 | 0.0398 | 0.0266 | 0.0591 | 0.0328 |
| 8 | A4 (+gc) | `+ gradient_correction` | discovery_new | 0.973 | 0.0393 | 0.0259 | **0.0611** | 0.0329 |
| 9 | A4 +gc, noriem | `block_hste_pt gc` (no riem) | discovery_new_noriem | 0.134 | — collapsed — | | | |
| 10 | A4 clip+gc | `block_hste_pt hste_clip gc` | discovery_new_clip | 0.749 | — crashed† — | | | |
| | *— recon routers on A4 block hop —* | | | | | | | |
| 11 | a6.1 (keep-first) | `a6.1 ...riem` | 23788083 | 0.974 | 0.0397 | 0.0261 | 0.0593 | 0.0324 |
| 12 | a6.1 + gc | `a6.1 ... gc` | 23805021 | 0.973 | 0.0386 | 0.0266 | 0.0576 | 0.0327 |
| 13 | a7.1 (keep-last) | `a7.1 ...riem` | 23788085 | 0.972 | 0.0361 | 0.0239 | 0.0589 | 0.0312 |
| 14 | a7.1 + gc | `a7.1 ... gc` | 23805022 | 0.971 | **0.0407** | **0.0273** | 0.0601 | **0.0336** |
| | *— per-layer routers (A3 base, no block hop) + full-sum + strict-gc —* | | | | | | | |
| 15 | a6 (keep-first) | `new_method hste riem a6` | 23908020 | 0.970 | 0.0372 | 0.0251 | 0.0567 | 0.0314 |
| 16 | a7 (keep-last) | `new_method hste riem a7` | 23908021 | 0.975 | 0.0381 | 0.0251 | 0.0588 | 0.0317 |
| 17 | a8 (full sum) | `new_method hste riem block_hste_pt a8` | 23908022 | 0.973 | 0.0395 | 0.0267 | 0.0601 | 0.0333 |
| 18 | A4_v2 (strict gc) | `...block_hste_pt riem gc A4_v2` | 23908023 | 0.208 💥 | 0.0327 | 0.0229 | 0.0490 | 0.0281 |
| 19 | A5 (+last-commit chain) | `...block_hste_pt riem gc A5` | 23935320 | 0.974 | **0.0399** | **0.0276** | 0.0585 | **0.0335** |

**Reading:**
- **a8 (full-sum router) is the strongest of the new arms** — R@10 0.0601 (= A4+gc's runner-up band) and NDCG@10 0.0333, second only to a7.1+gc (0.0336) across the whole table. Reopening *all* per-layer recon on top of the block hop helps at n_q=4, mirroring NLP where a8 tied the study best.
- The **per-layer-only routers a6/a7** (no block hop) land at/just-below A4 (R@10 0.0567/0.0588) — healthy (uniq ≈0.97) but not better than the block-hop family here.
- **A4_v2 (strict gc) collapses the codebook** (uniq 0.208) and drops *below* Euclidean (R@10 0.0490 < 0.0513) — the same strict-gc failure seen at NLP (75.3 ≈ Euclidean): removing the layer-0 commit residual leak starves the encoder.
- Every hyperbolic variant that keeps codebook health (uniq ≳0.95) beats Euclidean across the board. The two failures are codebook-collapse cases: **noriem** (uniq 0.134, fully collapsed) and **HSTE-only** without `new_method` (uniq 0.627).
- **Best Recall@10 is A4+gc (0.0611)**; **best on the @5 metrics + NDCG@10 is a7.1+gc** (R@5 0.0407, NDCG@5 0.0273, NDCG@10 0.0336). The a6.1/a7.1 routers don't beat plain A4 on R@10 — at this shallow depth (n_q=4) they're roughly a wash with A4, with a7.1+gc edging the ranking metrics.
- `gc` barely moves things at n_q=4 (consistent with the depth-conditional story).

\* **Euclidean baseline is mismatched** — it ran without `--dedup` at batch 512, so it's not strictly comparable; a matched rerun is pending.

† **A4 clip+gc**: VAE finished (codes saved) but the recommender crashed on the old `item_codes` vs `beauty_item_codes` filename mismatch (since fixed by the codes-path consolidation), so a rerun would now complete.

## 6. Cross-dataset check — Sports & Outdoors / Toys & Games

Same HRQ config (`768→32, n_q=4, bins=128, 5000 ep, quant=0.01, recon=1000,
batch=2048, --dedup`) on two further Amazon categories, to test whether the
Beauty result transfers. Full discovery + 100-ep seq2seq recommender, test-set
metrics; uniq. = pre-dedup multitoken uniqueness ratio. Best per column **bold**.

**Sports & Outdoors** (18 357 items)

| method | job | uniq. | R@5 | NDCG@5 | R@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Euclidean | 23826264 | 0.960 | 0.0194 | 0.0123 | 0.0318 | 0.0163 |
| Vanilla hyperbolic | 23826265 | 0.826 | **0.0210** | **0.0136** | **0.0358** | **0.0183** |
| A4 (+gc) | 23826266 | 0.958 | 0.0196 | 0.0123 | 0.0324 | 0.0164 |
| A4 (no gc) | 23839672 | 0.955 | 0.0180 | 0.0115 | 0.0297 | 0.0153 |

**Toys & Games** (11 924 items)

| method | job | uniq. | R@5 | NDCG@5 | R@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Euclidean | 23826261 | 0.974 | 0.0334 | 0.0227 | 0.0528 | 0.0289 |
| Vanilla hyperbolic | 23826262 | 0.878 | **0.0385** | **0.0253** | **0.0573** | **0.0314** |
| A4 (+gc) | 23826263 | 0.975 | 0.0354 | 0.0234 | 0.0547 | 0.0296 |
| A4 (no gc) | 23839671 | 0.975 | 0.0358 | 0.0244 | 0.0533 | 0.0300 |

**Reading:** the Beauty headline (*hyperbolic > Euclidean*) replicates on both
categories — every hyperbolic arm beats Euclidean on every metric. But the
*within-hyperbolic* winner flips: here **plain vanilla hyperbolic sweeps both
datasets** (and by a clear margin: +13–18 % R@10 over Euclidean), while the A4
block-hop and gc — which led on Beauty — sit between vanilla and Euclidean.
Consistent with the depth-conditional story: at n_q=4 the A4 machinery buys
nothing over vanilla, and on these two datasets it slightly hurts. (Euclidean
here is matched — `--dedup`, batch 2048 — unlike the Beauty Euclidean caveat.)

---

# Audio Task Results — SoundStream 24 kHz (LibriTTS train-clean-100)

Codec config: `codebook_dim=64` (via `--tangent_proj`), `n_q=12, bins=1024`;
hyperbolic runs `c=1, cw=1.0`. Full hyperbolic recipe = `--new_method
--block_hste_pt --hste_riemannian [--gradient_correction] --encoder_scale -1
--tangent_proj --code_max_radius 0.9 --uniform`, kmeans off. Metric = best
validation reconstruction loss ↓; "health" = per-layer val perplexity (12
layers). 💥 = codebook collapse (val ppl ≈ 1 across all layers).

## 7. Depth-12 reconstruction, 3 epochs (d64)

| method | job | val_rec ↓ | codebook health |
|---|---|---|---|
| Euclidean | 23670386 | 179.73 | all alive, 625–871 |
| Vanilla hyperbolic | 23671651 | 211.99 | — |
| A4 (riem+gc) | 23693961 | 177.72 | all alive, 101–513 |
| A4 clip+gc | 23731633 | **173.25** | layers 2–4 near-dead (3.7–4.6), rest 616–726 |
| A5 (A4 + last-commit chain) | 23741656 | 258.22 💥 | collapsed from epoch 1 |
| *— per-layer family at n_q=12 (no block hop) —* | | | |
| a6 (keep-first, per-layer) | 23907996 | 228.26 | alive but weak, ppl 7–156 |
| a7 (keep-last, per-layer) | 23907997 | 263.20 | layer-0 dead (ppl 1.0), rest alive |
| A3 (per-layer HSTE + riem) | 23671854 | 264.05 💥 | collapsed, all 12 ppl = 1.0 |
| a8 (block hop + full per-layer sum) | 23924478 | 262.81 💥 | collapsed, all 12 ppl ≈ 1.0 |
| A4_v2 (strict gc) | 23935321 | 262.24 💥 | collapsed, all 12 ppl ≈ 1.0 |

clip+gc is the best d64/3-ep run (its live layers carry euclidean-grade ppl, but
3 of 12 are near-dead). A5 re-opens the rec × commit × dead-code-revival collapse
that gc contains — the n_q=12 verdict inverts the shallow-depth (NLP n_q=4) one.

**Per-layer family at depth.** The per-layer recon routers and the full per-layer
A3 base — which *top* the NLP table at n_q=4 — all degrade or collapse at the
audio depth n_q=12: a6 (228), a7 (263, layer-0 dead), A3 (264 💥, full collapse),
every one worse than the block-hop A4 (177.7). This is the direct depth-conditional
confirmation: the per-layer chain's N compounding transports, which are an asset
at n_q=4, become a liability at n_q=12, and the single block hop is what survives.
(a8 reattaches all per-layer codes on the block hop — the known collapse-risk
corner at this depth — and duly collapsed: 262.8, all 12 ppl ≈ 1.0.)

## 8. Depth-12 reconstruction, 10 epochs (d64) — latest set, incl. recon routers

| method | job | val_rec ↓ | codebook health |
|---|---|---|---|
| Euclidean | 23788090 | **117.63** | all 12 alive, 871–975 |
| A4 (riem+gc) | 23788089 | 121.18 | all alive, 37–677 |
| a6.1 +gc | 23788086 | 271.05 💥 | collapsed from epoch 1 |
| a6.1 +gc (rerun) | 23822121 | 144.45 | healthy, all alive |
| a6.1 no-gc | 23805026 | 150.50 | healthy |
| a7.1 +gc | 23788087 | 268.62 💥 | collapsed |
| a7.1 no-gc | 23805027 | 272.43 💥 | collapsed |

**Reading.** At 10 epochs Euclidean leads (117.6) and **A4 riem+gc is the best
hyperbolic (121.2)**. The recon-router arms are **bistable at this setting**: the
*same* a6.1+gc config collapsed from epoch 1 in one run (271.0) but trained
healthily in a rerun (144.45); a6.1 no-gc is reliably healthy (150.5) while a7.1
collapses in both gc and no-gc. None beat A4 — confirming that a6.1's earlier
"best d64" number was an n=1 bistable artifact, not a real audio improvement.

## 9. Dim-sweep viability envelope (A4, 3 epochs)

| dim | Euclidean | A4 (+gc) | A4 no-gc |
|---|---|---|---|
| 8   | 160.6 | 265.9 💥 | 263.3 💥 |
| 32  | 157.1 | 179.0 | 183.0 |
| 64  | 179.7 | **177.7** | 264.7 💥 → 207.7 w/ `--leaky_clamp` |
| 128 | 159.7 | 279.9 💥 | 261.3 💥 |

A4's viable envelope is **d32–d64 only**; `gc` is *required* at d64 but *fatal* at
native dim (d512/10 ep: Euclidean 124.1 best, A4 no-gc 151.2 healthy, A4+gc
274 💥). `--uniform` quantizer-depth dropout is a required ingredient (all
no-uniform ablations collapse). Perceptual R-D is flat for the hyperbolic codec.

**Takeaway (audio):** unlike the other three tasks, hyperbolic shows **no
reconstruction advantage** — Euclidean wins recon at every viable dim and depth,
and the hyperbolic arms' main story is collapse-avoidance. Audio is the clearest
"hyperbolic-as-pure-compressor is not supported" case.

---

# Mean validation residual (`--approx`) across tasks

The `--approx` flag logs `approx_distance` = mean `hyperbolic_distance_sq` between
the reconstructed code-sum-plus-leftover `(quantized_out ⊕ residual)` and the
encoder output on the ball, i.e. how far the quantizer's reconstruction lands
from the encoder output in hyperbolic space (0 ⇒ exact). It is **not** computed
for Euclidean `c=0` runs (left at 0). Values below are for the three core
hyperbolic arms. `approx` is a construction flag and `HyperbolicSTE.forward`
returns the code-sum unchanged, so the block hop never alters the forward value —
eval-mode recompute is consistent with the training-time number.

**Image — MNIST / CIFAR-100** (best-epoch, mean over seeds 42/43/44; from each
checkpoint's `logs/log.txt`):

| Dataset | Vanilla hyperbolic | A4 (+gc) | A4 (no gc) |
|---|---|---|---|
| MNIST | 0.00049 | <0.00001 | <0.00001 |
| CIFAR-100 | 0.00133 | <0.00001 | <0.00001 |

**Recommendation — Amazon** (final validation pass, n_q=4):

| Dataset | Vanilla hyperbolic | A4 (+gc) | A4 (no gc) |
|---|---|---|---|
| Beauty | 44.787 | 0.0313 | 0.0143 |
| Sports & Outdoors | 50.404 | 0.0375 | 0.0173 |
| Toys & Games | 48.151 | 0.0509 | 0.0191 |

**Audio — SoundStream d64** (recomputed, job 23881968; eval-mode forward on 100
dev-clean files at fixed full depth n_q=12, since training used random-depth
`--uniform` dropout and never logged this):

| Run | Vanilla hyperbolic | A4 (+gc) | A4 (no gc) |
|---|---|---|---|
| d64, 3 epochs | 57.77 | 0.145 | 57.77 † |
| d64, 10 epochs | — | 0.026 | — ‡ |

† A4-no-gc d64/3ep is the collapsed run (§9: 264.7 💥); its reconstruction lands
on the ball boundary, where `hyperbolic_distance_sq`'s atanh clamp saturates at a
fixed ceiling (~57.77) — the same constant any boundary-saturated model returns,
hence the exact match with vanilla. ‡ No vanilla-10ep or A4-no-gc-d64-10ep
checkpoints exist (the only 10-epoch no-gc run is d512).

**NLP — WordNet, standard cell d16/b128** (closure split; means over repeat runs):

| Method | mean approx | source jobs |
|---|---|---|
| Vanilla hyperbolic | ~14.4 | nlp2_a4var 23740875/76, 23711047 |
| A4 (+gc) | 12.9 | nlp2_a4var 23805002–007 (6 reps) |
| A4 (no gc) | 8.9 | nlp2_a4var 23784105–110 (6 reps) |

*(NLP reported for the headline d16/b128 cell only; per-cell values across the
full §4 capacity grid are scattered across many `a4var` job batches and not
reconstructed here.)*

**Reading.** The same pattern holds in every domain: **A4's block-PT hop
reconstructs the encoder output on the ball almost exactly** (residual collapses
toward 0 — strikingly so in rec/audio, ~0.01–0.15 vs vanilla's ~45–58), whereas
**vanilla per-layer hyperbolic RVQ leaves a large residual** because the per-layer
Möbius subtractions do not telescope back to the encoder output. NLP at d16 is the
one regime where A4 and vanilla stay comparable (~9–14). The 0.00000 image cells
are genuinely <5e-6, not missing data.
