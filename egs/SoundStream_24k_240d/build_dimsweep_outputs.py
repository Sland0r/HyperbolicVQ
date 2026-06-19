#!/usr/bin/env python3
"""Build the 10-epoch audio dimension-sweep deliverables:
  1. recon dimension table   (best val recon_loss_valid per dim x config)  -> LaTeX
  2. perceptual R-D table     (n_q=12 PESQ/STOI/SI-SDR/melL1 + entropy kbps) -> LaTeX
  3. R-D curves               (quality vs entropy-kbps across n_q)            -> PNG/PDF
Reads training logs (logs/<name>_*.out) for recon and rd_results/dimsweep/<name>.rd.json
for perceptual numbers. Robust to missing cells (prints '--').
"""
import glob
import json
import math
import os
import re

ROOT = "/home/acolombo/VAEs"
LOGDIR = f"{ROOT}/logs"
RDDIR = f"{ROOT}/egs/SoundStream_24k_240d/rd_results/dimsweep"
OUTDIR = f"{ROOT}/egs/SoundStream_24k_240d/dimsweep_out"
os.makedirs(OUTDIR, exist_ok=True)

DIMS = [8, 32, 64, 128, 512]
# Dimensions shown in the R-D curves figure (Fig. tab:audio_rd companion). The full
# sweep clutters the plot, so the figure keeps only the low/mid/high trio 8/32/128.
FIG_DIMS = [8, 32, 128]
# (column label, dict dim->run_name)
COLS = [
    ("Euclidean", {8: "euc_d8_10ep", 32: "euc_d32_10ep", 64: "euc_d64_10ep",
                   128: "euc_d128_10ep", 512: "euc_10ep"}),
    ("GHRQ-VAE (ours)", {d: f"blockptste_riem_gc_d{d}_ema_10ep" for d in DIMS}),
    ("GHRQ-VAE (no gc)", {d: f"blockptste_riem_nogc_d{d}_ema_10ep" for d in DIMS}),
    ("Naive hyperbolic", {d: f"hyp_std_nocmr_esema099_d{d}_10ep" for d in DIMS}),
]
# d64 +gc has a capped sibling that is the thesis n_q=12 number; note it separately.

COLLAPSE_RECON = 250.0  # recon at/above this == collapsed

# Cells whose training log was cleaned up but whose best val recon_loss_valid is
# published (thesis Table tab:audio_d64). Used only when no log is found.
KNOWN_RECON = {
    "euc_d64_10ep": 117.63,
}


def newest_log(name):
    cands = sorted(glob.glob(f"{LOGDIR}/{name}_*.out"), key=os.path.getmtime)
    return cands[-1] if cands else None


def best_recon(name):
    """min recon_loss_valid across the training log."""
    lg = newest_log(name)
    if not lg:
        return KNOWN_RECON.get(name)
    vals = []
    with open(lg) as fh:
        for line in fh:
            m = re.search(r"recon_loss_valid:([0-9.]+)", line)
            if m:
                vals.append(float(m.group(1)))
    return min(vals) if vals else None


def load_rd(name):
    p = f"{RDDIR}/{name}.rd.json"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def fmt(x, nd=1, star=False):
    if x is None:
        return "--"
    s = f"{x:.{nd}f}"
    return s + (r"\,$\boldsymbol{\times}$" if star else "")


# ---------- Table 1: recon dimension sweep ----------
recon = {}  # (col, dim) -> (val, collapsed)
for label, mp in COLS:
    for d in DIMS:
        v = best_recon(mp[d])
        recon[(label, d)] = (v, (v is not None and v >= COLLAPSE_RECON))

lines = [r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\hline",
         "dim & " + " & ".join(c for c, _ in COLS) + r" \\", r"\hline"]
for d in DIMS:
    cells = []
    for label, _ in COLS:
        v, coll = recon[(label, d)]
        cells.append(fmt(v, 1, star=coll))
    lines.append(f"${d}$ & " + " & ".join(cells) + r" \\")
lines += [r"\hline", r"\end{tabular}"]
recon_tex = "\n".join(lines)

# ---------- Table 2: perceptual R-D (n_q=12) ----------
plines = [r"\begin{tabular}{llccccc}", r"\hline",
          r"Config & dim & PESQ$\uparrow$ & STOI$\uparrow$ & SI-SDR$\uparrow$ "
          r"& mel-L1$\downarrow$ & kbps(H)$\downarrow$ \\", r"\hline"]
for label, mp in COLS:
    for d in DIMS:
        rd = load_rd(mp[d])
        if rd is None:
            plines.append(f"{label} & ${d}$ & " + " & ".join(["--"] * 5) + r" \\")
            continue
        pt = next((p for p in rd["points"] if p["n_q"] == 12), rd["points"][-1])
        plines.append(
            f"{label} & ${d}$ & {fmt(pt['pesq'],3)} & {fmt(pt['stoi'],3)} & "
            f"{fmt(pt['sisdr'],2)} & {fmt(pt['meldist'],4)} & {fmt(pt['kbps_entropy'],2)} \\\\")
    plines.append(r"\hline")
plines.append(r"\end{tabular}")
perc_tex = "\n".join(plines)

with open(f"{OUTDIR}/table_recon.tex", "w") as fh:
    fh.write(recon_tex + "\n")
with open(f"{OUTDIR}/table_perceptual.tex", "w") as fh:
    fh.write(perc_tex + "\n")

print("===== RECON TABLE =====\n" + recon_tex)
print("\n===== PERCEPTUAL TABLE =====\n" + perc_tex)

# ---------- Figure: R-D curves ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("pesq", "PESQ-wb $\\uparrow$"), ("sisdr", "SI-SDR (dB) $\\uparrow$")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4.3))
    cmap = {"Euclidean": plt.cm.Blues, "GHRQ-VAE (ours)": plt.cm.Oranges,
            "Naive hyperbolic": plt.cm.Greens}
    for ax, (mk, mlabel) in zip(axes, metrics):
        for label, mp in COLS:
            if label not in cmap:
                continue
            shades = cmap[label](
                [0.35 + 0.5 * i / max(1, len(FIG_DIMS) - 1) for i in range(len(FIG_DIMS))])
            for di, d in enumerate(FIG_DIMS):
                rd = load_rd(mp[d])
                if rd is None:
                    continue
                pts = [p for p in rd["points"] if p[mk] is not None]
                if not pts:
                    continue
                xs = [p["kbps_entropy"] for p in pts]
                ys = [p[mk] for p in pts]
                ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=shades[di],
                        label=f"{label} d{d}")
        ax.set_xlabel("entropy rate (kbps)")
        ax.set_ylabel(mlabel)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Audio rate--distortion, 10-epoch dimension sweep ($n_q$=1,2,4,8,12)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(f"{OUTDIR}/rd_curves.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/rd_curves.pdf", bbox_inches="tight")
    print(f"\nwrote {OUTDIR}/rd_curves.png and .pdf")
except Exception as e:
    print(f"figure skipped: {e}")
