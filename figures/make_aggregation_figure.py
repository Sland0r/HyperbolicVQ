"""Figure for thesis section 4.2: aggregation of residual codebook gyrovectors
on the Poincare ball.

A single encoder point z_e is residual-quantized into 3 codebook gyrovectors
q0, q1, q2 that behave like a real RVQ: q0 is the coarse code (largest, oriented
toward z_e), q1, q2 are progressively smaller refinements of the shrinking
residual. The SAME codes are then aggregated in two orders:

  * new_method (right-nested):   q0 + (q1 + q2)
        the exact algebraic inverse of the encoder residual chain
        r_{k+1} = (-q_k) + r_k  (Mobius left-translation), so it reconstructs
        z_e up to the final residual r3 -> d(z_hat_new, z_e) = ||r3||.

  * standard (left-nested):      (q0 + q1) + q2
        Mobius addition is non-associative; the gyration gyr[q0,q1] rotates q2,
        so this drifts away from z_e -> d(z_hat_std, z_e) > d(z_hat_new, z_e).

z_e is defined as the inverse-consistent reconstruction of the codes, so the
new_method endpoint is provably the closer of the two. The effect is modest at
this shallow depth (3 codes) and grows near the boundary / with depth; the
points are placed near the boundary and shown in a zoom inset so the (real)
difference is legible.

c = 1 (Poincare disk), 2-D for visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C = 1.0  # curvature

# ----------------------------------------------------------------------------
# Poincare-ball ops (2-D, mirrors academicodec/quantization/core_vq.py)
# ----------------------------------------------------------------------------
def mobius_add(x, y, c=C):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xy = np.dot(x, y); xx = np.dot(x, x); yy = np.dot(y, y)
    num = (1 + 2 * c * xy + c * yy) * x + (1 - c * xx) * y
    den = 1 + 2 * c * xy + c * c * xx * yy
    return num / den

def project(x, c=C, eps=1e-5):
    x = np.asarray(x, float)
    maxn = (1 - eps) / np.sqrt(c)
    n = np.linalg.norm(x)
    return x * (maxn / n) if n > maxn else x

def hyp_dist(x, y, c=C):
    diff = mobius_add(-np.asarray(x, float), y, c)
    return (2 / np.sqrt(c)) * np.arctanh(min(np.sqrt(c) * np.linalg.norm(diff), 1 - 1e-12))

def geodesic_points(p, q, n=200):
    """Sample points along the Poincare-disk geodesic between p and q."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    x1, y1 = p; x2, y2 = q
    det = 4 * (x1 * y2 - x2 * y1)
    if abs(det) < 1e-9:  # collinear with origin -> straight diameter segment
        t = np.linspace(0, 1, n)
        return np.outer(1 - t, p) + np.outer(t, q)
    b1 = x1 * x1 + y1 * y1 + 1
    b2 = x2 * x2 + y2 * y2 + 1
    cx = (b1 * 2 * y2 - b2 * 2 * y1) / det
    cy = (2 * x1 * b2 - 2 * x2 * b1) / det
    cen = np.array([cx, cy])
    r = np.sqrt(max(cx * cx + cy * cy - 1, 1e-12))
    a1 = np.arctan2(y1 - cy, x1 - cx)
    a2 = np.arctan2(y2 - cy, x2 - cx)
    d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi  # shorter arc (stays in disk)
    th = a1 + np.linspace(0, 1, n) * d
    return cen + r * np.column_stack([np.cos(th), np.sin(th)])

def polar(r, deg):
    a = np.deg2rad(deg)
    return np.array([r * np.cos(a), r * np.sin(a)])

# ----------------------------------------------------------------------------
# RVQ-like codes: q0 coarse (toward z_e), q1/q2 shrinking refinements.
# z_e := inverse-consistent reconstruction  q0 + (q1 + (q2 + r3)),  r3 small.
# ----------------------------------------------------------------------------
q0 = polar(0.55,  55)          # coarse code  (largest, ~ z_e direction)
q1 = polar(0.45, 165)          # refinement 1 (smaller, coarse residual turns)
q2 = polar(0.34,   5)          # refinement 2 (smaller still)
r3 = polar(0.04, 200)          # tiny leftover residual after 3 layers

z_e = project(mobius_add(q0, mobius_add(q1, mobius_add(q2, r3))))

# new_method: right-nested  q0 + (q1 + q2)   -> drawn O -> q2 -> q1 -> q0
v1 = project(q2)
v2 = project(mobius_add(q1, v1))
v3 = project(mobius_add(q0, v2))
new_path, z_new = [np.zeros(2), v1, v2, v3], v3

# standard: left-nested  (q0 + q1) + q2       -> drawn O -> q0 -> q1 -> q2
s1 = project(q0)
s2 = project(mobius_add(s1, q1))
s3 = project(mobius_add(s2, q2))
std_path, z_std = [np.zeros(2), s1, s2, s3], s3

d_new = hyp_dist(z_new, z_e)
d_std = hyp_dist(z_std, z_e)
print(f"d_hyp(z_hat_new, z_e) = {d_new:.4f}")
print(f"d_hyp(z_hat_std, z_e) = {d_std:.4f}")
print(f"standard drifts {d_std / d_new:.1f}x farther than new_method")

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 16, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(7.6, 7.6))

C_NEW = "#1b7837"   # green  - new_method
C_STD = "#d95f02"   # orange - standard

def draw_ball(a):
    th = np.linspace(0, 2 * np.pi, 400)
    a.plot(np.cos(th), np.sin(th), color="0.25", lw=1.6, zorder=1)
    a.add_patch(plt.Circle((0, 0), 1, color="0.955", zorder=0))

def draw_chain(a, path, color, lbls=None, side=1):
    for i in range(len(path) - 1):
        arc = geodesic_points(path[i], path[i + 1])
        a.plot(arc[:, 0], arc[:, 1], color=color, lw=2.6,
               solid_capstyle="round", zorder=3)
        if lbls is not None:
            mid = arc[100]; tang = arc[110] - arc[90]
            perp = np.array([-tang[1], tang[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-9)
            a.annotate(lbls[i], mid + side * 0.085 * perp, color=color,
                       fontsize=18, ha="center", va="center", zorder=6,
                       fontweight="bold")
    pts = np.array(path)
    a.scatter(pts[1:-1, 0], pts[1:-1, 1], s=30, color=color,
              edgecolor="white", lw=0.8, zorder=5)

def draw_points(a, s_ze=320, s_hat=120, labels=True, loff=1.0):
    # error geodesics
    for z, col in [(z_new, C_NEW), (z_std, C_STD)]:
        arc = geodesic_points(z, z_e)
        a.plot(arc[:, 0], arc[:, 1], color=col, lw=1.4, ls=(0, (2, 2)), zorder=4)
    a.scatter(*z_e, s=s_ze, marker="*", color="k", edgecolor="white", lw=1.0, zorder=8)
    a.scatter(*z_new, s=s_hat, marker="o", color=C_NEW, edgecolor="white", lw=1.2, zorder=8)
    a.scatter(*z_std, s=s_hat, marker="s", color=C_STD, edgecolor="white", lw=1.2, zorder=8)
    if labels:
        a.annotate(r"$z_e$", z_e + loff * np.array([0.07, -0.01]), fontsize=20, zorder=9)
        a.annotate(r"$\hat{z}_{\mathrm{HRA}}$", z_new + loff * np.array([0.07, -0.07]),
                   color=C_NEW, fontsize=19, zorder=9)
        a.annotate(r"$\hat{z}_{\mathrm{naive}}$", z_std + loff * np.array([0.05, 0.05]),
                   color=C_STD, fontsize=19, zorder=9)

# --- main ball ---
draw_ball(ax)
draw_chain(ax, new_path, C_NEW, [r"$q_3$", r"$q_2$", r"$q_1$"], side=-1)
draw_chain(ax, std_path, C_STD, [r"$q_1$", r"$q_2$", r"$q_3$"], side=1)
draw_points(ax, labels=True, loff=1.0)
ax.scatter(0, 0, s=45, color="0.25", zorder=6)
ax.annotate(r"$O$", [-0.055, -0.07], fontsize=18)

# --- zoom inset around z_e ---
half = 0.17
xc, yc = z_e
axins = ax.inset_axes([0.60, 0.055, 0.37, 0.37],
                      xlim=(xc - half, xc + half), ylim=(yc - half, yc + half))
draw_chain(axins, new_path, C_NEW)
draw_chain(axins, std_path, C_STD)
draw_points(axins, s_ze=420, s_hat=150, labels=True, loff=0.5)
axins.set_xticks([]); axins.set_yticks([])
for sp in axins.spines.values():
    sp.set_edgecolor("0.4")
axins.set_title("zoom near $z_e$", fontsize=14, pad=3)
ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=1.0, alpha=0.8)

# legend
handles = [
    Line2D([0], [0], color=C_NEW, lw=2.6,
           label=r"HRA:  $q_1 \oplus (q_2 \oplus q_3)$"),
    Line2D([0], [0], color=C_STD, lw=2.6,
           label=r"Naive Approach:  $(q_1 \oplus q_2) \oplus q_3$"),
    Line2D([0], [0], color="k", lw=0, marker="*", markersize=14,
           label=r"$z_e$ (Encoder Output)"),
    Line2D([0], [0], color="0.4", lw=1.4, ls=(0, (2, 2)),
           label=r"Geodesic Error $d(\hat{z}, z_e)$"),
]
ax.legend(handles=handles, loc="lower left", fontsize=15, frameon=True,
          framealpha=0.95, borderpad=0.7)

ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()

out = "figures/poincare_aggregation.png"
fig.savefig(out, dpi=450, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
