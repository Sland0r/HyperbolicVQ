# Gradient-Routing Ablation — NLP / WordNet hierarchy

*Standard cell: `n_q=4, bins=128, embed_dim=16, c=1, lr=1.0, 50 ep, closure split`.
Metric = Recall@10 (%), beam-search, teacher-forced. Block-hop arms anchored on
**A4** (`--new_method --block_hste_pt --hste_riemannian`). Blank cells = experiments
not yet run (see §6).*

---

## 1. Framing — the encoder's gradient budget

An HRQ-VAE encoder maps a concept to a point `r0` on the Poincaré ball; the
residual quantiser turns `r0` into a code tuple and a reconstruction. During
training the encoder receives gradient from up to **four distinct routes**, and
every method below is just a choice of which routes are open and how strongly.
The whole ablation is "remove / reroute one channel, hold the rest at the A4
setting."

- **R_layer — per-layer reconstruction.** Reconstruction gradient flowing back
  through the *attached* residual chain, one quantiser layer at a time (the
  standard RVQ straight-through path). Vanilla hyperbolic and A3 keep it fully
  open; **A4 closes it** (`block_ste` detaches the per-layer codes); the routers
  a6/a7/a8 reopen restricted slices of it.
- **R_block — block reconstruction hop.** A single block-level straight-through
  estimator (`HyperbolicSTE`) mapping the whole quantised block back onto `r0`.
  Its backward sends **100 % of ∂recon/∂Q to the encoder and nothing to the
  codes** — an isometric, one-shot recon channel. A4 is defined by having *only*
  this recon route open.
- **C — commitment.** `cw · d²(sg[q], r0)`, pulling the encoder toward its codes.
  Breadth is itself a routing choice: **all layers** (no-gc), **first layer only**
  (gc), **first layer with the residual leak removed** (strict-gc / A4_v2), or
  **off** (cw=0).
- **riem — Riemannian rescaling.** The conformal λ² discount applied once to the
  recon gradient (`hste_riemannian`), preventing the boundary blow-up of the
  hyperbolic STE.

> **A4 = {R_layer off, R_block on, C on, riem on}.**

---

## 2. Master decomposition (cw = 1.0, standard cell)

| Arm | R_layer | R_block | C breadth | riem | flags (besides `--new_method`) | **R@10** |
|---|:--:|:--:|:--:|:--:|---|:--:|
| Euclidean (c=0) | full (Euc-STE) | — | all | — | `--c 0` | 75.9 |
| Vanilla hyp | full (Möbius) | ✗ | all | ✗ | — | 83.4 |
| Vanilla hyp + gc | full | ✗ | layer-0 | ✗ | `gc` | 82.3 |
| **A3** (per-layer HSTE) | full (HSTE) | ✗ | all | ✓ | `hste riem` | **85.2** |
| A3 + gc | full | ✗ | layer-0 | ✓ | `hste riem gc` | 81.5 |
| **A4 (no-gc)** | ✗ | ✓ | all | ✓ | `block_hste_pt riem` | **84.1** |
| A4 + gc | ✗ | ✓ | layer-0 | ✓ | `block_hste_pt riem gc` | 81.9 |
| A4_v2 (strict gc) | ✗ | ✓ | layer-0 *strict* | ✓ | `block_hste_pt riem gc block_recon` | 75.3 |
| A5 (no-gc) | ✗ | ✓ | all + last-chain | ✓ | `block_hste_pt riem A5` | 84.4 |
| A5 + gc | ✗ | ✓ | layer-0 + last-chain | ✓ | `block_hste_pt riem gc A5` | 84.4 |
| A4_v2 + A5 | ✗ | ✓ | strict + last-chain | ✓ | `… A5 block_recon` | 81.8 |
| a6 (keep-first) | first only | ✗ | all | ✓ | `hste riem a6` | 84.8 |
| **a7 (keep-last)** | last only | ✗ | all | ✓ | `hste riem a7` | **85.6** |
| a6.1 (A4 ⊕ a6) | first | ✓ | all | ✓ | `block_hste_pt riem a6_1` | 84.6 |
| a7.1 (A4 ⊕ a7) | last | ✓ | all | ✓ | `block_hste_pt riem a7_1` | 85.0 |
| **a8 (A4 ⊕ full)** | all | ✓ | all | ✓ | `hste riem a8` | **85.6** |
| A4 no-riem | ✗ | ✓ | all | ✗ | `block_hste_pt` | 84.7 |
| A3 no-riem | full | ✗ | all | ✗ | `hste` | 54.7 |

All values are single-cell numbers at the standard config (a6/a7 = mean of 3
reps: a6 84.8/84.2/85.3, a7 85.6/85.9/85.2; A4_v2 = mean 75.2/75.5/75.2).

---

## 3. Focused study A — channel isolation (which route carries the signal)

Hold the **block hop on**, switch R_layer and commitment on/off. Because "commit
off" *is* `cw=0`, this square is run at the **cw = 0.25** operating point (its own
internally-consistent reference: euc 75.4, vanilla 79.4, A3 80.7). The four
corners answer: *given the always-on hop, where does the hyperbolic gain come
from — per-layer recon or commitment?*

| | **C off** (cw=0) | **C on** (all-layer, cw=0.25) |
|---|:--:|:--:|
| **R_layer off** | hop-only — **74.7** (exp3) | a4nogc — **81.8** (exp1) |
| **R_layer on** (a8) | a8 + cw0 — **75.6** (exp2) | _to run — a8 @ cw0.25_ |

Control (remove the hop itself, `block_recon`): R_layer off-ish, hop off, gc
commit → **78.0**.

**Reading.** Commitment is the load-bearing channel. With the hop as the sole
recon route, turning commit on lifts the encoder from **74.7 (below Euclidean's
75.4) to 81.8**; turning per-layer recon on instead (commit off, 75.6) buys
essentially nothing — note exp2 and exp3 share CE ≈ 7.74. The hop *alone* carries
almost no usable encoder signal at this depth; a4nogc's 81.8 is the
all-layer commitment, not the hop. The missing 4th corner tests whether
reopening per-layer recon *on top of* commit+hop (a8) adds anything over a4nogc.

---

## 4. Focused study B — commitment breadth (the `gc` axis)

Block-hop family, riem on, cw = 1.0. Only the commitment routing changes.

| C breadth | flag | R@10 |
|---|---|:--:|
| all layers | `no gc` (A4 no-gc) | **84.1** |
| first layer only | `gc` (A4 + gc) | 81.9 |
| first layer, leak removed | `block_recon` (A4_v2 strict) | 75.3 |
| off | `cw=0` (hop-only, cw0.25) | 74.7 |

**Reading.** Commit breadth is monotone here: every path you close costs Recall.
`gc` (keep only the layer-0 commit) drops 2.2 pts; removing even the residual
*leak* that `gc` still lets through (**A4_v2 strict**) collapses the model all the
way to Euclidean (75.3) — i.e. the layer-0 commit *leak* is load-bearing, not
incidental. At n_q=4 the block hop is invisible under `gc` (the gc and no-gc
training curves coincide), so `gc` here is pure signal removal — the opposite of
its role at depth-12 audio, where `gc` is *required* to contain collapse.

---

## 5. Focused study C — per-layer recon routing (the routers)

How much of R_layer to reopen, and on which base. a6/a7 reopen a single layer of
per-layer recon on the **A3 base** (no block hop); a6.1/a7.1/a8 add the **A4 block
hop** underneath (summed, value-exact). cw = 1.0, riem on.

| Router | base | which layers of R_layer | R@10 |
|---|---|---|:--:|
| a6 (keep-first) | A3 (no hop) | first | 84.8 |
| a7 (keep-last) | A3 (no hop) | last | **85.6** |
| a6.1 (A4 ⊕ a6) | A4 (hop) | first | 84.6 |
| a7.1 (A4 ⊕ a7) | A4 (hop) | last | 85.0 |
| **a8 (A4 ⊕ full)** | A4 (hop) | all | **85.6** |

**Reading.** At n_q=4 the **per-layer recon family tops the table**: a7 (keep-last,
85.6), a8 (full sum, 85.6) and A3 (full per-layer, 85.2) tie for best, all ahead
of the best pure block-hop arm (A4 no-gc, 84.1). Adding the block hop underneath
a6/a7 (→ a6.1/a7.1) does *not* help and slightly hurts a6 (84.8 → 84.6); the
un-truncated a8 recovers the a7 ceiling. All consistent with the
depth-conditional story: the block hop's one-shot isometric recon is worth more
than N compounding per-layer transports only when N is large (n_q=12 audio), not
at n_q=4 — where reopening per-layer recon is what tops the table.

---

## 6. Riemannian-rescaling ablation (the last NLP axis — now complete)

Drop `--hste_riemannian` and hold everything else at the family's best setting
(cw = 1.0, no-gc), to ask whether the conformal λ² discount is necessary.

| family | riem on | riem off | Δ |
|---|:--:|:--:|:--:|
| per-layer (A3) | 85.2 | **54.7** 💥 | −30.5 |
| block-hop (A4 no-gc) | 84.1 | 84.7 | +0.6 |

**Reading — riem is load-bearing only for the per-layer family.** The per-layer
HSTE applies N=4 compounding tangent transports; each amplifies the boundary λ²
blow-up, so without the Riemannian discount the encoder gradient explodes near
the ball boundary and the model craters (85.2 → 54.7). The block hop is a *single*
isometric STE that never compounds, so it does not need the discount at all — A4
is in fact marginally better without it (84.1 → 84.7). This is the gradient-routing
explanation for why A4 trades the per-layer chain for one block hop: it removes
the very compounding that makes riem mandatory. (Note this is the *opposite* of
the rec result, where dropping riem collapsed the block-hop arm to uniq 0.134 —
flagged in §7 as the one cross-task inconsistency to chase.)

The NLP gradient-routing ablation is now complete across all axes. Dropped by
request: the cw = 0.25 channel-isolation 4th corner (a8 @ cw0.25).

Already in hand: Euclidean, vanilla(±gc), A3(±gc), A3 no-riem, A4(±gc),
A4 no-riem, A4_v2 (strict), A5(±gc), A4_v2+A5, a6, a7, a6.1, a7.1, a8.

---

## 7. Takeaways

1. **Commitment is the load-bearing encoder channel.** The recon×commit square
   (§3): commit on/off swings Recall 74.7 → 81.8 with the hop fixed; per-layer
   recon on/off (commit off) barely moves it. Closing commit paths is monotone
   costly (§4), and removing the layer-0 commit *leak* collapses the model to
   Euclidean.
2. **The block hop is depth-conditional, not a free win.** At n_q=4 it carries
   almost no signal on its own and the per-layer-recon arms (a7 85.6, A3 85.2)
   beat every pure block-hop arm. Its isometric one-shot recon pays off only when
   the per-layer transports compound (n_q=12 audio).
3. **`gc` reverses sign with depth.** Here (shallow) it only removes commit signal
   (−2.2 pts, strict-gc −9 pts); at depth it is the collapse firewall.
4. **Riemannian rescaling is load-bearing only where transports compound.** It is
   mandatory for the per-layer family (A3 85.2 → 54.7 without it) and superfluous
   for the single-hop family (A4 84.1 → 84.7). This is the mechanistic reason A4
   swaps N per-layer transports for one block hop — it engineers away the boundary
   blow-up that forces the discount.
