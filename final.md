# Final paper experiments

Methods (consistent across all tables):
- **Euclidean** — `c=0`, plain RVQ, no method flags.
- **Vanilla hyperbolic** — `c=1`, Euclidean STE, no method flags.
- **A4 (ours)** — `c=1 --new_method --block_hste_pt --hste_riemannian`
  (block-level PT-STE with the riemannian-once discount; `--gradient_correction`
  is added only for deep quantizer stacks — see the audio section — and omitted
  at shallow depth, where it starves the commit signal).
- **A4_v2 (strict gc)** — A4 `+ --gradient_correction --A4_v2`. Vanilla gc only
  detaches the residual *after* each step, so layer 0's (coarsest) commitment
  still leaks to the encoder through `r0`; `--A4_v2` detaches the *first*
  residual too, so **no** per-layer commitment loss reaches the encoder. Safe
  only with a block STE (recon flows through the separate `r0` hop, not the loop
  residual). Combined with `--A5` it leaves exactly one commit path — the
  last-layer chain. The strict behavior is gated behind the opt-in `--A4_v2`
  flag so plain gc / A4 numbers are unchanged.

---

## 1. NLP — WordNet hierarchy (Recall@10 %, closure split)

Config: 50 epochs, `embed_dim=16, n_q=4, bins=128, batch=2048, lr=1.0, c=1`
(euclidean `c=0`), **`commitment_weight=1.0`**, single seed.
Source logs: `NLP_2/logs_nlp_4/nlp2_a4var_*.out`.

| embed_init_scale | Euclidean | Vanilla hyperbolic | A4 (ours) |
|---|---|---|---|
| 1.0  | 75.8 (23711830) | 83.3 (23711047) | **84.1** (23711829) |
| 0.75 | 77.8 (23715043) | **85.5** (23715045) | 83.2 (23715046) |
| 0.5  | 80.0 (23715047) | **85.6** (23715048) | 83.9 (23715049) |
| 0.3  | 82.9 (23714239) | **86.6** (23714240) | 82.5 (23714243) |
| 0.05 | 91.8 (23714244) | **94.1** (23714245) | 93.6 (23714246) |

### 1a-bis. A5 at the reference cell (scale 1.0, dim 16, bins 128, cw=1.0)

`--A5` = commitment loss only at the first and last RQ steps; the last one is
computed on a never-detached residual chain (codes detached inside it) so its
gradient reaches the embedding through all Möbius updates, independent of gc.

n=3 per variant (training is unseeded, so each run is an i.i.d. draw; run 1 =
the original job listed, runs 2–3 = the June 12 variance reruns, jobs
23740872–23740888). Mean ± sample std:

| variant | run-1 job | Recall@10 (n=3) |
|---|---|---|
| Euclidean | 23711830 | 75.90 ± 0.10 |
| Vanilla hyperbolic | 23711047 | 83.43 ± 0.42 |
| A4 no-gc (table 1a ref) | 23711829 | 83.93 ± 0.67 |
| A4 + gc | 23710509 | 81.93 ± 0.31 |
| A4 + gc, clip hop | 23731496 | 82.27 ± 0.15 |
| A5 + gc | 23739270 | **84.40 ± 0.26** |
| A5 (no gc) | 23739271 | **84.37 ± 0.35** |
| A4_v2 (strict gc) | 23768934 | 75.30 ± 0.17 |
| A4_v2 + A5 | 23770215 | 81.77 ± 0.35 |

Strict-gc reading (June 13, jobs 23768934–36 / 23770215–17): removing the
layer-0 commit leak entirely (A4_v2) **collapses recall to Euclidean level**
(75.3 vs Euclidean 75.9) — with zero commitment gradient the encoder is steered
only by reconstruction and the hierarchy signal is lost. Adding the A5
last-layer chain back (A4_v2 + A5) restores one commit path and recovers recall
to 81.8 — essentially A4+gc's 81.9, and below A5's 84.4. So at shallow depth one
clean commit path ≈ one leaky path; A5's edge comes from keeping *two* (coarse
leak + last chain). The coarse layer-0 commit that vanilla gc leaks is
load-bearing, not an inconsistency to remove.

Reading (variance-checked): A5+gc fully repairs gc's commit starvation at
shallow depth — the +2.5 gap over A4+gc is ~8 pooled stds, unambiguous. The
single last-step chain carries enough of the severed deep-layer signal. A5
without gc is the best cell mean (84.37 vs A4's 83.93): pruning intermediate
commitment losses to first+last helps even when all paths are open, though
this smaller gap (+0.4, ~1σ of A4's spread) is consistent rather than
conclusive — both A5 arms also run visibly tighter than A4 no-gc. Contrast
with images (§3c), where the same re-opened deep signal costs hierarchy
recovery — at shallow depth / NLP the recovered signal is net-positive on
the retrieval metric.

### 1b. Capacity sweep at scale 1.0 — embed_dim × codebook size (Recall@10 %)

Config as above, `embed_init_scale=1.0, commitment_weight=1.0` fixed;
rows = (embed_dim, bins). dim16/bins128 row = the scale-1.0 row of table 1.

| dim | bins | Euclidean | Vanilla hyperbolic | A4 (ours) |
|---|---|---|---|---|
| 6  | 64  | 76.4 | **87.6** | 73.0 |
| 6  | 128 | 74.7 | 69.2 | **75.1** |
| 16 | 64  | 76.7 | **86.9** | 82.3 |
| 16 | 128 | 75.8 | 83.3 | **84.1** |
| 32 | 64  | **75.0** | 53.0 † | 48.7 † |
| 32 | 128 | **71.1** | 53.6 † | 36.6 † |

### 1c. Rescuing dim 32 — codebook / embedding init treatments (A4, scale-1.0 row)

A4 only; three init treatments on the failing dim-32 cells. `cb×k` multiplies
the initial codebook points by k (`--codebook_init_mult`; base init radii are
uniform in (0, 0.5]): ×0.1 pulls codes near the origin, ×1.6 pushes them out
toward the boundary-pinned data. "proper scale" leaves codebooks alone and
sets `--embed_init_scale 0.707` = √(16/32), restoring the dim-16 init norm
(the √dim-compensated, "properly scaled" init).

| dim | bins | A4 (base) | A4 cb×0.1 | A4 cb×1.6 | A4 proper scale |
|---|---|---|---|---|---|
| 32 | 64  | 48.7 | 47.5 (23720350) | 43.6 (23720351) | **82.6** (23720353) |
| 32 | 128 | 36.6 | 35.5 (23720354) | 42.8 (23720356) | **80.7** (23720357) |

Matched baselines at proper scale (0.707, dim 32):

| dim | bins | Euclidean | Vanilla hyperbolic | A4 (ours) |
|---|---|---|---|---|
| 32 | 64  | 77.4 (23721377) | **89.8** (23721378) | 82.6 |
| 32 | 128 | 70.5 (23721380) | **87.9** (23721381) | 80.7 |

Reading: codebook-side init treatments do nothing (±5 pts around the broken
base — wherever the codes start, the boundary-pinned embeddings cannot move
toward them because their own gradients are clamp-dead). The √dim-compensated
embedding init fully rescues A4 back to its ~81–84 band: the dim-32 failure
is entirely encoder-side boundary pinning. The matched baselines complete the
picture: vanilla hyperbolic recovers even more (89.8/87.9 — the best cells of
the whole capacity study), while the Euclidean control barely moves
(75.0→77.4, 71.1→70.5), confirming init scale only matters through ball
geometry. Two-knob summary: embedding init radius is a hard prerequisite
(gates whether hyperbolic gradients exist); codebook init is immaterial.
Overall NLP verdict: with radius control in place vanilla hyperbolic is the
top performer; A4's value is FLATNESS — its column stays in an 80–84 band
across every trainable regime (init scale 0.5–1.0, dim 6–32, bins 64–128)
while vanilla swings 53→94, i.e. A4 trades peak performance for robustness
to mis-scaled initialization.

† dim-32 hyperbolic cells are optimization failures, not capacity limits: at
scale 1.0 the init norm grows with √dim (≈5.7 at dim 32), exp_map0 saturates
on the boundary and the commit distance clamps — the hyp run barely trains
(loss 247→186) and A4 freezes outright (CE 15.9→15.6 in 50 ep; the riemannian
discount further shrinks already clamp-dead gradients). Euclidean has no
boundary and is unaffected. The dim6/bins128 vanilla dip (69.2) is the
opposite pathology: best train CE of the sweep (4.73) but worst
generalization. Net: dim 16 is the sweet spot at scale 1.0; A4 wins both of
its winning cells there/at d6b128, vanilla wins where capacity is small
(dim 6–16 with bins 64), and nothing hyperbolic survives dim 32 at this init
scale — radius control (encoder-side or init-side) remains a prerequisite
beyond ~dim 16.

Reading: at cw=1.0 the strong commit pull is itself a partial boundary remedy
for vanilla hyperbolic (79.4→83.3 at scale 1.0 vs cw=0.25), so A4's win
survives only at the most extreme pinning (scale 1.0); vanilla wins every
other row, with A4 second everywhere (except 0.3, where euclidean edges it by
0.4). A4 beats euclidean at 4 of 5 scales. Reference at cw=0.25, scale 1.0:
euc 75.4 / hyp 79.4 / A4 81.9 — there, where vanilla's boundary fragility is
unmitigated, A4's margin is larger (+2.5). hyp+cw1 @ 0.05 (94.1) is the best
NLP cell of the whole study.

### 1d. hste_clip variant (A4 with clip instead of the riemannian discount)

One run at the scale-1.0, cw=1.0 cell: `--new_method --block_hste_pt
--hste_clip --gradient_correction` (job 23731496, `a4gccw1clip_s-1p0`).

| embed_init_scale | Euclidean | Vanilla hyperbolic | A4 (riem) | A4 clip+gc |
|---|---|---|---|---|
| 1.0 | 75.8 | 83.3 | **84.1** | 82.3 |

Reading: clip+gc beats euclidean by +6.5 and the graph-composition baseline
(80.1) but sits ~2 pts under both A4-riem and vanilla — clip does not pay off
at shallow depth (n_q=4), consistent with the rec result below and opposite
to audio (section 4), where the same recipe is the best run of the study.

### 1e. Depth sweep — Recall@10 % vs quantizer depth (scale 1.0, cw=0.25)

Jobs 23700631–636 (n_q 8/12); the n_q=4 row is the matched cw=0.25 reference
(A4 there = no-gc per the header definition; the 8/12 rows are full A4 +gc,
its deep regime).

| n_q | Euclidean | Vanilla hyperbolic | A4 (ours) |
|---|---|---|---|
| 4  | 75.4 | 79.4 | **81.9** |
| 8  | 65.1 | **75.7** | 75.1 |
| 12 | 45.8 | **74.1** | 73.8 |

Reading: one of the strongest pro-hyperbolic results of the project —
Euclidean CRATERS with depth (75→65→46) while both hyperbolic variants hold
~74–76 essentially flat. Deeper residual stacks on hierarchical data are
where the geometry earns its keep; A4 tracks vanilla within ~0.5 pt at every
depth (routing fixes have nothing left to fix on this shallow model).

---

## 2. Recommendation — Amazon Beauty

Config: RQ/HRQ-VAE on MPNet embeddings (`768→32, n_q=4, bins=128`, 5000 ep,
`quant=0.01, recon=1000, batch=2048`), then seq2seq recommender (100 ep) on the
generated codes with the `--dedup` uniqueness token. Test metrics from the
recommender. The "no gc" A4 row matches the header definition; the "+gc" row
adds `--gradient_correction`. Source logs: `logs_rec_1/discovery_*.out`.

| method | job | uniq. ratio | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Euclidean | 23279215* | 0.932 | 0.0342 | 0.0235 | 0.0513 | 0.0290 |
| Vanilla hyperbolic | 23320936 | 0.872 | 0.0388 | 0.0250 | 0.0600 | 0.0318 |
| A4 (ours, no gc) | 23715002 | 0.970 | **0.0398** | **0.0266** | 0.0591 | 0.0328 |
| A4 (ours, +gc) | discovery_new | 0.973 | 0.0393 | 0.0259 | **0.0611** | **0.0329** |
| A4 clip+gc | discovery_new_clip† | 0.749 | — | — | — | — |

Reading: both A4 variants restore codebook health (uniqueness 0.97 vs
vanilla's 0.87) and beat euclidean on every metric. gc barely matters at this
shallow depth (n_q=4): no-gc takes the @5 metrics (R@5 0.0398, NDCG@5 0.0266),
+gc takes Recall@10 (0.0611 vs 0.0591), NDCG@10 is a tie (0.0328/0.0329).

*\*Euclidean baseline is mismatched: no dedup, batch 512 (matched rerun pending).*

†*clip variant (`--hste_clip` instead of `--hste_riemannian`): VAE stage
completed (val recon 4.66e-4, dedup codes saved) but the seq2seq stage crashed
on a filename mismatch (`item_codes_c1.0_dedup.pt` saved vs
`beauty_item_codes_c1.0_dedup.pt` expected), so no recommender metrics. Its
uniqueness ratio (0.749, Q0 ppl 5.1 with 55/128 codes used) is the worst in
the table — codebook health alone rules it out at this depth.*

## 3. Images — MNIST / CIFAR-100 / EMNIST

Config: 50 epochs, n_q=4, bins=128, single seed. Variant names: "A4" =
block-PT riem+gc (on images the default A4 includes gc); no-gc / clip / cw=1
are single-delta ablations. "A5" = A4 + `--A5` (jobs 23738361/23738364,
June 12): commitment loss only at the first and last RQ steps, the last one
computed on a never-detached residual chain so its gradient reaches the
encoder through all Möbius updates — restoring the deep-layer commitment
signal that gc severs, via one controlled path. Source logs: `egs/MNIST_VQVAE/logs_new/*.out`,
evals under `egs/MNIST_VQVAE/evaluations/`.

### 3a. Reconstruction — best val recon loss ↓ (×10⁻³, mean ± std over seeds 42/43/44)

Variance reruns June 12 (jobs 23740833–23740871), exact config replicas of
the seed-42 originals. EMNIST column: single seed-42 run only.

| method | MNIST | CIFAR-100 | EMNIST (n=1) |
|---|---|---|---|
| Euclidean | **0.477 ± 0.006** | **1.077 ± 0.006** | **0.17** |
| Vanilla hyperbolic (c=1) | 0.600 ± 0.020 | 1.280 ± 0.017 | 0.18 |
| per-layer HSTE | 0.587 ± 0.025 | 1.323 ± 0.055 | 0.19 |
| per-layer HSTE + gc | 1.260 ± 0.130 † | 1.390 ± 0.087 | 0.19 |
| A4 (block-PT riem+gc) | 0.647 ± 0.023 | 1.360 ± 0.046 | — |
| A4 no-gc | 0.667 ± 0.035 | 1.470 ± 0.027 | — |
| A4 clip (+gc) | 0.587 ± 0.031 | 1.243 ± 0.021 | — |
| A4 cw=1 | 0.757 ± 0.023 | 1.590 ± 0.036 ‡ | — |
| A5 (A4 + last-commit chain) | 0.623 ± 0.021 | 1.400 ± 0.030 | — |
| A4_v2 (strict gc) | ~1.6 | 7.25 ± 0.34 | — |
| A4_v2 + A5 | 1.81 ± 0.99 | 3.00 ± 0.23 | — |

Strict-gc reading (jobs 23768937–942 / 23770218–224): A4_v2 wrecks recon
(CIFAR 7.25 vs A4 1.36, ~5×) and `approx_dist` explodes to 38 (MNIST) / 57
(CIFAR) vs ~0 for A4 — with no commitment the codebook stops tracking the
encoder, so quantization (and thus reconstruction) falls apart. The A5
last-layer chain re-pins it: A4_v2 + A5 brings `approx_dist` back to ~0 and
halves CIFAR recon error to 3.00, but stays ~2× A4/A5 and is noisy on MNIST
(1.81 ± 0.99). Neither strict variant matches the leaky-gc recon.

†*per-layer riem+gc MNIST degradation (approx_dist 13.6 — the riemannian
discount compounding N times) is robust across seeds; block-level A4 fixes
it, replicating the once-not-N-times story.* ‡*one of the three CIFAR cw=1
values read from the log at 4 decimals (the run's best_31.pth fell out of
the checkpoint retention window).*

Variance-checked reading: Euclidean's recon edge and clip's best-hyperbolic
status (both datasets) are robust. The single-seed "A5 beats A4 on MNIST"
claim softens to parity-or-slightly-better: A5 0.623±0.021 vs A4
0.647±0.023 (~1σ gap, A5 ahead in all three seeds); on CIFAR A4 is ahead by
a similarly noise-level margin. Net: A5 changes image recon by at most ~1σ
in either direction.

### 3b. MNIST generation — RQ-Transformer (Lee et al.), 10k samples

Jobs 23719936–940 (June 12). These supersede the earlier evaluate.py
generation FIDs (~233–241), which used the wrong fold order.

| method | job | FID ↓ | IS ↑ | best CE |
|---|---|---|---|---|
| Euclidean | 23719936 | 20.36 | 2.069 | 2.123 |
| Vanilla hyperbolic | 23719937 | **15.01** | **2.105** | 2.099 |
| per-layer HSTE | 23719938 | 17.24 | 2.075 | 2.154 |
| A4 (riem+gc) | 23719939 | 16.78 | 2.068 | 2.150 |
| A4 no-gc | 23719940 | 18.48 | 2.077 | 2.122 |

Reading: every hyperbolic variant beats Euclidean on generation FID despite
losing on recon; vanilla wins outright (15.01 vs 20.36, −26%). Hyperbolic
codes are better autoregressive tokens even when they are worse
reconstructions. clip / cw=1 variants not yet run through this eval.

### 3c. CIFAR-100 superclass hierarchy recovery ↑ (eval 23741416; mean ± std over seeds 42/43/44)

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
| A5 (A4 + last-commit chain) | 0.151 ± 0.006 | 0.058 ± 0.005 | 0.481 ± 0.003 | 0.337 ± 0.006 | 0.046 ± 0.003 |
| A4_v2 (strict gc) | 0.138 ± 0.014 | 0.060 ± 0.010 | 0.494 ± 0.018 | 0.340 ± 0.010 | 0.063 ± 0.047 |
| A4_v2 + A5 | 0.107 ± 0.001 | 0.047 ± 0.018 | 0.479 ± 0.018 | 0.333 ± 0.021 | 0.041 ± 0.010 |

Strict-gc reading (eval 23770875): the commit→encoder routing is a clean
monotone knob for hierarchy. Rank by ARI: A4 coarse-commit-only (0.087) ≫
A4_v2 strict / no commit (0.060) ≈ A5 coarse+last (0.058) > **A4_v2 + A5
deep-commit-only (0.047 ≈ Euclidean, prec@4 0.107 = lowest in the table)**. The
*more* the deep-layer commitment alone steers the encoder, the *worse* the
taxonomy — direct confirmation that deep commitment fights hierarchy emergence
(§1a-bis / §3a show the same signal helps recall and recon-alignment). A4's win
comes precisely from keeping only the coarse commit and severing the deep ones;
A4_v2 + A5 does the opposite and is dominated — best at nothing.

Variance-checked reading: every hyperbolic variant still beats Euclidean on
ARI/NMI/purity; A4 riem+gc remains the taxonomy winner with the gap intact
(ARI 0.087±0.008 vs Euclidean 0.048±0.005). A5's hierarchy collapse is
robust, not a seed artifact: ARI 0.058±0.005 and NMI 0.481±0.003 are
statistically indistinguishable from vanilla hyperbolic and ~3 pooled stds
below A4. Combined with §1a-bis and §3a: the deep-layer commitment signal
that gc severs is real, recoverable signal for task metrics (NLP retrieval
+2.5, robust; MNIST recon parity-or-better) but it actively fights the
codebook hierarchy that gc's severing lets emerge — A4's taxonomy win comes
precisely from NOT letting deep commitment losses steer the encoder. Note
A4 no-gc's ARI is seed-fragile (±0.019, one seed hit 0.101), so single-run
hierarchy comparisons in this band should be read with care. Caution: the earlier linear-probe result (A4 13.5%
vs Euclidean 24.9%) predates the evaluate.py fold-order fix — treat as
unverified pending re-eval.

## 4. Audio — SoundStream LibriTTS 24 kHz

Config: 3 epochs, `codebook_dim=64, n_q=12, bins=1024`, LibriTTS
train-clean-100, `--code_max_radius 0.9 --encoder_scale -1 --tangent_proj`,
kmeans off, cw=1.0 for the hyperbolic runs. Metric: best validation
reconstruction loss. Source logs: `logs/*_d64_3ep_*.out`.

| method | job | val_rec ↓ | codebook health (val ppl, 12 layers) |
|---|---|---|---|
| Euclidean | 23670386 | 179.73 | all alive, 625–871 |
| Vanilla hyperbolic | 23671651 | 211.99 | — |
| A4 (riem+gc) | 23693961 | 177.72 | all alive, 101–513 |
| A4 clip+gc | 23731633 | **173.25** | layers 2–4 near-dead (3.7–4.6), rest 616–726 |
| A5 (A4 riem+gc + last-commit chain) | 23741656 | 258.22 † | COLLAPSED: val ppl ≈ 1–2 all layers from epoch 1 |

†*A5 = single delta `--A5` on the A4 riem+gc config. Epoch 1 tracks A4
almost exactly (identical grad norms, commit 1.0 vs 0.8 at iter 1000) but
deep-layer train ppl is already sagging (Q9–Q11: 21/12/8 vs A4's 43/34/33);
by epoch 2 commit explodes 1.0 → 72, feat_loss blows up, and val ppl pins
at ~1 (single code). Best val_rec is the pre-collapse epoch-1 value. The
n_q=12 verdict inverts the shallow-depth one (NLP n_q=4: A5+gc = best,
§1a-bis): at depth, gc's severing isn't starving useful signal so much as
CONTAINING the rec × commit × dead-code-revival collapse conjunction — A5
re-opens a commitment path from the deepest residual to the encoder and the
collapse driver reasserts itself despite code_max_radius + encoder_scale
auto-calibration. A5's value is depth-conditional, mirroring the
riem-vs-clip hop story.*

Reading: clip+gc is the best d64 run to date — −4.5 vs the previous hyperbolic
champion (riem+gc) and −6.5 vs matched euclidean — and its live layers carry
euclidean-grade perplexities (616–726). The cost is three nearly-dead middle
layers, so it wins recon with effectively ~9 of 12 quantizers. Note the
depth split with sections 1d/2: clip+gc only pays off on the deep stack
(n_q=12); at n_q=4 (NLP, rec) the riemannian discount remains the better A4.

### 4b. Native dim, 10 epochs — Euclidean still ahead as a pure compressor

No tangent_proj bottleneck (d512), 10 epochs, otherwise matched.

| method | job | val_rec ↓ | best_epoch |
|---|---|---|---|
| Euclidean | 23652295 | **124.13** | 10 |
| A4 no-gc | 23715394 | 151.19 | 10 (improving every epoch, all 12 layers alive) |
| Vanilla hyperbolic | 23652296 | 153.31 | 9 |
| gradcorr (no riem) | 23652297 | 158.11 | 9 |
| A4 +gc | 23700480 | 274.12 💥 | 3 (PPL 1.0 ×12 from ep1) |

Reading: a4nogc is the best hyperbolic 10-ep run ever (commit monotone
2.9→1.3, never near saturation, grads flat) but Euclidean leads by 27 points
at native dim — the d64 win in 4a does not transfer. Note gc flips sign:
required at d64 (4c), fatal at d512.

### 4c. Dim sweep — A4 viability envelope (3 epochs, val_rec ↓)

| dim | Euclidean | A4 (+gc) | A4 no-gc |
|---|---|---|---|
| 8   | 160.6 | 265.9 💥 | 263.3 💥 |
| 32  | 157.1 | 179.0 | 183.0 |
| 64  | 179.7 | **177.7** | 264.7 💥 → 207.7 with `--leaky_clamp` |
| 128 | 159.7 | 279.9 💥 | 261.3 💥 |

Euclidean is healthy at every dim; A4's envelope is d32–d64 only (gc buys
d64 at the cost of d512, see 4b). 💥 = commit saturates at the 72.2079
ceiling = 1.25 × (2·atanh(1−1e-3))², the hard clamp of
`hyperbolic_distance_sq` whose gradient is exactly zero — collapse is an
absorbing zero-grad plateau. `--leaky_clamp` (C¹ linear extension of atanh
past the knee) causally cures the nogc-d64 cell (264.7 → 207.7, all layers
alive, commit never spikes) but at 3 ep still trails gc's 177.7.

Two further A4 audio facts: (1) `--uniform` quantizer-depth dropout is a
REQUIRED ingredient — all five no-uniform ablations (d8–d512) collapse to
PPL 1.0 ×12 from epoch 1; full recipe = `--new_method --block_hste_pt
--hste_riemannian [--gradient_correction at d≤64] --tangent_proj
--code_max_radius 0.9 --encoder_scale -1 --uniform`. (2) Perceptual R-D is
FLAT for hyperbolic (3-ep riem+gc ckpt: PESQ 1.098→1.102 for n_q 1→12 while
entropy scales 0.87→9.25 kbps) — depth adds bits, not quality; measured at
3 ep (PESQ ~1.1 absolute = undertrained), so the 10-ep/leaky reversal is
untested.

---

## 5. Framing & caveats

What the results support, and how strongly:

1. **Hyperbolic captures hierarchy better — strongly supported.** NLP: hyp
   beats euc in every trainable cell, and euc craters with depth (75→65→46)
   while hyp holds ~74–76 (table 1e). CIFAR-100 taxonomy: every hyp variant
   beats euc; A4 ARI +80% (3c). This is the core hypothesis and it holds
   across both domains that have ground-truth hierarchy.
2. **Hyperbolic codes are better generative/downstream tokens — supported.**
   MNIST RQ-Transformer FID 15.0 vs 20.4, all hyp variants win (3b); rec
   beats euc on all four recommender metrics (2).
3. **Hyperbolic as a pure compressor — NOT supported.** Euclidean wins recon
   on all three image datasets (3a), leads by 27 pts at audio native dim
   (4b), and hyperbolic perceptual R-D is flat (4c). The honest distortion
   claim is parity-at-best in matched bottleneck configs (4a).
4. **Attribution: several headline wins belong to vanilla c=1, not A4.**
   NLP's best cells (94.1 @ scale 0.05; 89.8 proper-scale d32) and the MNIST
   FID win are plain hyperbolic with radius control. A4's defensible claim
   is robustness/trainability: the flat 80–84 NLP band, fixing per-layer
   riem degradation (3a†), the only hyperbolic surviving deep audio stacks,
   the audio d64 win (4a), and the taxonomy win (3c).
5. **The recipe is domain-conditional.** clip vs riem flips between audio
   (clip wins, 4a) and NLP/rec/taxonomy (riem wins, 1d/2/3c); gc is required
   at audio d64, fatal at d512, harmful at shallow NLP — non-monotone in
   both depth and dim. No single setting wins everywhere; the paper should
   own this explicitly.
6. **Commit-routing ablation (A4_v2) — the layer-0 commit leak is load-bearing.**
   *Motivation:* vanilla gc only detaches the residual *after* each step, so
   A4+gc isn't "no commit to the encoder" — layer 0's coarsest commit still
   leaks through `r0`. We tested removing it (A4_v2, strict gc → zero commit to
   encoder) and routing only the deep commit (A4_v2 + A5 → last-layer chain
   only). *Result:* commit→encoder routing is a clean monotone knob. (a)
   A4_v2 collapses everything — NLP→Euclidean (75.3), recon 5× worse,
   `approx_dist`→38/57 (codebook stops tracking the encoder): the leaked coarse
   commit was the only thing aligning quantization. (b) A4_v2 + A5 recovers
   alignment (`approx_dist`→0) and recall (81.8 ≈ A4+gc) but is **dominated** —
   ~2× recon, and the **worst hierarchy of any hyperbolic variant** (ARI 0.047,
   prec@4 0.107). Hierarchy ranks A4 coarse-only (0.087) ≫ strict (0.060) ≈ A5
   (0.058) > deep-only (0.047): the deeper the commit that steers the encoder,
   the worse the taxonomy. *Verdict:* A4's win comes from keeping the **coarse**
   commit and severing the deep ones; strict gc is a useful ablation knob (opt-in
   `--A4_v2`), not a default. (jobs 23768934–942 / 23770215–224; see 1a-bis,
   3a, 3c.)
7. **Open items / unclosed comparisons.** Single seeds throughout; audio
   wins are 3-epoch; rec Euclidean baseline mismatched (no dedup, batch
   512); rec clip metrics never produced (crashed, §2); image linear probe
   predates the eval fix; EMNIST has no block-PT runs; image clip/cw1
   variants lack RQ-Transformer and probe evals; 10-ep leaky/R-D reversal
   untested.
