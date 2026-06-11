"""Retrieve the CIFAR-100 label hierarchy from quantized representations.

CIFAR-100 ships a fixed 2-level taxonomy: 100 fine classes grouped into 20
coarse superclasses (5 fine classes each). We test whether the quantized latent
of a trained (H)RQ-VAE recovers that taxonomy, and whether hyperbolic geometry
(c>0) recovers it better than Euclidean (c=0).

Method (prototype clustering in each model's NATIVE geometry):
  1. Encode every test image -> quantized latent, reconstruct the per-position
     representation. For c>0 we recover exact Poincare-ball points via
     exp_map0(decode(codes))  (decode applies log_map0, exp_map0 inverts it).
  2. Pool the spatial grid to one point per image:
        c==0 : arithmetic mean        c>0 : Einstein midpoint
  3. Pool per-image points to one prototype per fine class (same rule).
  4. Pairwise distances between the 100 prototypes in the native metric
        c==0 : Euclidean              c>0 : Poincare geodesic
  5. Score recovery of the 20 superclasses:
        - sibling precision@4 (each superclass has 4 other members)
        - agglomerative clustering into 20 -> ARI / NMI / purity
        - cophenetic correlation vs the ground-truth tree
     and save a dendrogram with leaves coloured by superclass.

All recovery metrics are scale-invariant, so comparing them across geometries
is fair even though the underlying distances live in different spaces.
"""
import argparse
import os
import sys

sys.path.insert(0, "/home/acolombo/VAEs")
sys.path.insert(0, "/home/acolombo/VAEs/egs/MNIST_VQVAE")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from academicodec.quantization.core_vq import (
    exp_map0, project, einstein_midpoint, pairwise_hyperbolic_distance_sq,
)

# ---- CIFAR-100 coarse taxonomy (superclass -> 5 fine class names) -----------
SUPERCLASSES = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

# The cifar_new runs (variant -> checkpoint .pth). The gc run resolves its latest
# best_*.pth dynamically (epoch number not known ahead of time).
def _latest_best(run_dir):
    import glob as _glob
    cands = sorted(_glob.glob(os.path.join(run_dir, "best_*.pth")),
                   key=lambda p: int(p.split("best_")[-1].split(".pth")[0]))
    return cands[-1] if cands else None

RUNS = [
    ("euclidean", "/home/acolombo/VAEs/checkpoint/cifar_new/euclidean/23519938/best_47.pth"),
    ("hyperbolic_c1", "/home/acolombo/VAEs/checkpoint/cifar_new/c1/23519939/best_47.pth"),
    ("hyperbolic_c1_hste", "/home/acolombo/VAEs/checkpoint/cifar_new/c1_hste/23519941/best_48.pth"),
    ("hyperbolic_c1_hste_gc", _latest_best("/home/acolombo/VAEs/checkpoint/cifar_new/c1_hste_gc/23651593")),
]
RUNS = [(n, c) for n, c in RUNS if c is not None]


def load_model(checkpoint, device):
    ckpt_dir = os.path.dirname(checkpoint)
    sys.modules.pop("config", None)
    sys.path.insert(0, ckpt_dir)
    import config
    sys.path.pop(0)
    from mnist_vqvae import VQVAE2D

    model = VQVAE2D(
        D=config.D, n_q=config.n_q, bins=config.bins, c=config.c,
        exponential_lambda=getattr(config, "exponential_lambda", 0.0),
        uniform=getattr(config, "uniform", False),
        ema=getattr(config, "ema", False),
        kmeans_init=getattr(config, "kmeans_init", False),
        threshold_ema_dead_code=getattr(config, "threshold_ema_dead_code", 2),
        codebook_weight=getattr(config, "codebook_weight", 1.0),
        commitment_weight=getattr(config, "commitment_weight", 0.25),
        dot_product_weight=getattr(config, "dot_product_weight", 0.0),
        entailment_cone_weight=getattr(config, "entailment_cone_weight", 0.0),
        in_channels=3, img_size=32,
        size=getattr(config, "size", "large"),
        solution=getattr(config, "solution", False),
        gyration=getattr(config, "gyration", False),
        full_grid=getattr(config, "full_grid", False),
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    state = state.get("model", state) if isinstance(state, dict) else state
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strict=False tolerates pre-calibration checkpoints (2026-06-05 baselines)
    # that predate the _enc_scale/_enc_calibrated buffers. Those default to
    # (1.0, 0) -> identity at eval, so the older runs are unaffected; only the
    # new gc checkpoint carries trained values.
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"quantizer.vq._enc_scale", "quantizer.vq._enc_calibrated"}
    leftover = set(missing) - allowed
    assert not leftover and not unexpected, f"unexpected key mismatch: missing={leftover} unexpected={unexpected}"
    model.eval()
    return model, float(config.c)


@torch.no_grad()
def image_points(model, c, loader, device):
    """Return (per-image pooled point [N_img, D], fine label [N_img])."""
    pts, labels = [], []
    for x, y in loader:
        x = x.to(device)
        codes = model.encode(x)                       # (n_q, B, N)
        tangent = model.quantizer.decode(codes)       # (B, D, N) tangent space
        tangent = tangent.permute(0, 2, 1)            # (B, N, D)
        if c > 0:
            ball = project(exp_map0(tangent, c), c)   # exact ball points (B, N, D)
            B = ball.shape[0]
            w = torch.ones(ball.shape[1], 1, device=device)
            pooled = torch.stack([einstein_midpoint(ball[b], w, c)[0] for b in range(B)])
        else:
            pooled = tangent.mean(dim=1)              # (B, D)
        pts.append(pooled.cpu())
        labels.append(y.clone())
    return torch.cat(pts), torch.cat(labels)


def prototypes(pts, labels, c, device):
    """Pool per-image points into one prototype per fine class (0..99)."""
    protos = []
    for cls in range(100):
        z = pts[labels == cls].to(device)
        if c > 0:
            w = torch.ones(z.shape[0], 1, device=device)
            protos.append(einstein_midpoint(z, w, c)[0].cpu())
        else:
            protos.append(z.mean(dim=0).cpu())
    return torch.stack(protos)                         # (100, D)


def distance_matrix(protos, c, device):
    p = protos.to(device)
    if c > 0:
        d = pairwise_hyperbolic_distance_sq(p, p, c).clamp_min(0).sqrt()
    else:
        d = torch.cdist(p, p)
    d = d.cpu().numpy()
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def sibling_precision(D, coarse, k=4):
    """For each fine class, fraction of its k nearest neighbours in same superclass."""
    n = D.shape[0]
    accs = []
    for i in range(n):
        order = np.argsort(D[i])
        order = order[order != i][:k]
        accs.append(np.mean(coarse[order] == coarse[i]))
    return float(np.mean(accs))


def purity(pred, true):
    total = 0
    for cl in np.unique(pred):
        members = true[pred == cl]
        total += np.bincount(members).max()
    return total / len(true)


def evaluate(name, D, coarse, fine_names, coarse_names, out_dir):
    prec = sibling_precision(D, coarse, k=4)

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    pred = AgglomerativeClustering(
        n_clusters=20, metric="precomputed", linkage="average"
    ).fit_predict(D)
    ari = adjusted_rand_score(coarse, pred)
    nmi = normalized_mutual_info_score(coarse, pred)
    pur = purity(pred, coarse)

    # cophenetic correlation vs ground-truth tree (0 same superclass, 1 different)
    gt = (coarse[:, None] != coarse[None, :]).astype(float)
    coph_corr, _ = cophenet(Z, squareform(gt, checks=False))

    # dendrogram coloured by superclass
    fig, ax = plt.subplots(figsize=(22, 7))
    leaf_labels = [f"{fine_names[i]}|{coarse_names[coarse[i]]}" for i in range(len(fine_names))]
    dendrogram(Z, labels=leaf_labels, leaf_font_size=5, ax=ax)
    ax.set_title(f"CIFAR-100 retrieved hierarchy — {name}  "
                 f"(prec@4={prec:.3f}, ARI={ari:.3f}, NMI={nmi:.3f})")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dendrogram_{name}.png"), dpi=120)
    plt.close(fig)

    return dict(prec_at4=prec, ari=ari, nmi=nmi, purity=pur, cophenetic_corr=coph_corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/acolombo/VAEs/dataset/CIFAR100")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", default="/home/acolombo/VAEs/egs/MNIST_VQVAE/evaluations/cifar_hierarchy")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                           transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # fine idx -> coarse idx, using torchvision's own class ordering
    c2i = ds.class_to_idx
    coarse_names = list(SUPERCLASSES.keys())
    fine_to_coarse = {}
    for ci, sc in enumerate(coarse_names):
        for fname in SUPERCLASSES[sc]:
            fine_to_coarse[c2i[fname]] = ci
    assert len(fine_to_coarse) == 100, f"mapping covers {len(fine_to_coarse)} classes"
    coarse = np.array([fine_to_coarse[i] for i in range(100)])
    fine_names = [None] * 100
    for n, i in c2i.items():
        fine_names[i] = n

    results = {}
    for name, ckpt in RUNS:
        print(f"\n{'='*60}\n{name}  ({ckpt})\n{'='*60}", flush=True)
        model, c = load_model(ckpt, device)
        print(f"  curvature c = {c}", flush=True)
        pts, labels = image_points(model, c, loader, device)
        protos = prototypes(pts, labels, c, device)
        D = distance_matrix(protos, c, device)
        res = evaluate(name, D, coarse, fine_names, coarse_names, args.out_dir)
        results[name] = res
        for k, v in res.items():
            print(f"  {k:18s}: {v:.4f}", flush=True)

    # summary table
    print(f"\n{'='*78}\nSUMMARY — hierarchy recovery (higher is better)\n{'='*78}", flush=True)
    cols = ["prec_at4", "ari", "nmi", "purity", "cophenetic_corr"]
    header = f"{'variant':22s}" + "".join(f"{c:>17s}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, _ in RUNS:
        row = f"{name:22s}" + "".join(f"{results[name][c]:>17.4f}" for c in cols)
        print(row)
    print(f"\nDendrograms + this log saved under {args.out_dir}")


if __name__ == "__main__":
    main()
