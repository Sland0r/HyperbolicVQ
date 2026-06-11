"""Diagnose why the RQ-Transformer prior barely beats the uniform baseline.

For each VQ-VAE checkpoint it encodes the dataset into integer codes and reports,
per RQ depth and overall:
  - codebook usage          (# entries used / bins)
  - marginal/unigram entropy  H(c)            -> floor for a context-free prior
  - spatial bigram entropy    H(c_t | c_{t-1}) -> structure across positions
  - depth bigram entropy      H(c_d | c_{d-1}) -> structure across depths

Compare the transformer's *plateaued* cross-entropy (from the eval logs) against
the unigram entropy: if they are close, the prior has already captured all the
structure that exists and the bottleneck is the codes, not the transformer.

All entropies are in nats. Uniform baseline = ln(bins).

Usage:
  python3 code_entropy.py --roots checkpoint/cifar100 checkpoint/cifar100_100
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/acolombo/VAEs")
sys.path.insert(0, "/home/acolombo/VAEs/egs/MNIST_VQVAE")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from evaluate import load_model  # reuse the exact model-loading logic


def _entropy(counts):
    """Shannon entropy (nats) of a 1-D count vector; returns (H, num_used)."""
    import numpy as np
    total = counts.sum()
    used = int((counts > 0).sum())
    if total == 0:
        return 0.0, 0
    p = counts[counts > 0].astype("float64") / total
    return float(-(p * np.log(p)).sum()), used


def _cond_entropy(joint):
    """Conditional entropy H(Y|X) (nats) from a joint count matrix joint[x, y]."""
    import numpy as np
    total = joint.sum()
    if total == 0:
        return 0.0
    px = joint.sum(axis=1, keepdims=True)  # (X, 1)
    # H(Y|X) = sum_x p(x) * H(Y|X=x)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_y_given_x = np.where(px > 0, joint / px, 0.0)
        log_term = np.where(p_y_given_x > 0, np.log(p_y_given_x), 0.0)
    h_per_x = -(p_y_given_x * log_term).sum(axis=1)  # (X,)
    p_x = (px.squeeze(1) / total)
    return float((p_x * h_per_x).sum())


def find_best_checkpoint(model_dir):
    import glob
    bests = sorted(glob.glob(os.path.join(model_dir, "best_*.pth")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    return bests[-1] if bests else None


@torch.no_grad()
def encode_dataset(model, loader, n_q, device):
    """Return codes as a LongTensor (N, T, n_q) on CPU."""
    all_codes = []
    for imgs, _ in tqdm(loader, desc="Encoding", leave=False):
        codes = model.encode(imgs.to(device))     # (n_q, B, T)
        all_codes.append(codes.permute(1, 2, 0).cpu())  # (B, T, n_q)
    return torch.cat(all_codes, dim=0)


def analyse(codes, bins):
    """codes: (N, T, n_q) LongTensor. Returns per-depth + overall stats."""
    import numpy as np
    codes = codes.numpy()
    N, T, n_q = codes.shape
    uniform = math.log(bins)

    per_depth = []
    for d in range(n_q):
        cd = codes[:, :, d]  # (N, T)
        marg_counts = np.bincount(cd.reshape(-1), minlength=bins)
        h_marg, used = _entropy(marg_counts)

        # spatial bigram H(c_t | c_{t-1}) along the flattened position axis
        spatial_joint = np.zeros((bins, bins), dtype="int64")
        if T > 1:
            prev = cd[:, :-1].reshape(-1)
            cur = cd[:, 1:].reshape(-1)
            np.add.at(spatial_joint, (prev, cur), 1)
        h_spatial = _cond_entropy(spatial_joint) if T > 1 else h_marg

        per_depth.append(dict(depth=d, used=used, usage=used / bins,
                              h_marg=h_marg, h_spatial=h_spatial))

    # depth bigram H(c_d | c_{d-1}) pooled over all positions
    depth_cond = []
    for d in range(1, n_q):
        prev = codes[:, :, d - 1].reshape(-1)
        cur = codes[:, :, d].reshape(-1)
        joint = np.zeros((bins, bins), dtype="int64")
        np.add.at(joint, (prev, cur), 1)
        depth_cond.append(_cond_entropy(joint))

    mean_marg = float(np.mean([p["h_marg"] for p in per_depth]))
    mean_spatial = float(np.mean([p["h_spatial"] for p in per_depth]))
    mean_depth_cond = float(np.mean(depth_cond)) if depth_cond else float("nan")
    return dict(uniform=uniform, per_depth=per_depth, mean_marg=mean_marg,
                mean_spatial=mean_spatial, mean_depth_cond=mean_depth_cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="checkpoint root dirs, e.g. checkpoint/cifar100 checkpoint/cifar100_100")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data_dir", default="/home/acolombo/VAEs/dataset/CIFAR100")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    tf = transforms.ToTensor()
    train_data = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=tf)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    for root in args.roots:
        if not os.path.isdir(root):
            print(f"!! skipping missing root: {root}")
            continue
        print("\n" + "#" * 78)
        print(f"# ROOT: {root}")
        print("#" * 78)
        for name in sorted(os.listdir(root)):
            model_dir = os.path.join(root, name)
            if not os.path.isdir(model_dir):
                continue
            ckpt = find_best_checkpoint(model_dir)
            if ckpt is None:
                print(f"\n[{name}] no best_*.pth, skipping")
                continue

            # evaluate.load_model does `import config` from the checkpoint dir;
            # Python caches it in sys.modules, so evict any prior one to force a
            # fresh import for each checkpoint (configs differ across folders).
            sys.modules.pop("config", None)
            margs = SimpleNamespace(checkpoint=ckpt, dataset=args.dataset)
            model, config = load_model(margs, device)
            codes = encode_dataset(model, loader, config.n_q, device)
            stats = analyse(codes, config.bins)

            print(f"\n=== [{name}]  ckpt={os.path.basename(ckpt)}  "
                  f"bins={config.bins}  n_q={config.n_q}  c={config.c} ===")
            print(f"  uniform baseline ln(bins) = {stats['uniform']:.4f} nats")
            print(f"  {'depth':>5} {'usage':>10} {'H_marg':>9} {'H(c_t|c_t-1)':>13}")
            for p in stats["per_depth"]:
                print(f"  {p['depth']:>5} {p['used']:>4}/{config.bins:<4} "
                      f"{p['h_marg']:>9.4f} {p['h_spatial']:>13.4f}")
            print(f"  ---")
            print(f"  mean unigram entropy  H(c)            = {stats['mean_marg']:.4f}")
            print(f"  mean spatial bigram   H(c_t|c_t-1)    = {stats['mean_spatial']:.4f}")
            print(f"  mean depth bigram     H(c_d|c_d-1)    = {stats['mean_depth_cond']:.4f}")
            print(f"  (compare these against the RQ-Transformer's plateaued CE in the eval log)")
            del model, codes
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
