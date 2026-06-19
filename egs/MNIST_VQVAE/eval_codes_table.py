"""Test-set codebook statistics for the 6-experiment image table.

For each of the 6 experiments x {MNIST, CIFAR-100}, load the best checkpoint,
run the test set (val_loader = train=False split) through the encoder/quantizer,
and report:
  * per-codebook perplexity  ppl_q = 2 ** entropy(test-set freq over `bins`)
  * total number of distinct multi-level code tuples (effective vocabulary):
    a "tuple" is the (n_q,) vector of indices at one spatial position.

Routing flags (a6/a7/a7_1/block_hste_pt/gradient_correction/hste_riemannian)
only affect the backward pass; encode() is plain encoder + nearest-code
assignment, so they do not change the codes produced here.
"""
import os
import sys
import math
import importlib

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (display name, checkpoint dir) -- best_*.pth with the largest epoch is picked.
EXPERIMENTS = {
    "cifar100": [
        ("Euclidean",      "/home/acolombo/VAEs/checkpoint/cifar_new/euclidean/23519938"),
        ("Vanilla hyp",    "/home/acolombo/VAEs/checkpoint/cifar_new/c1/23519939"),
        ("A4 (riem+gc)",   "/home/acolombo/VAEs/checkpoint/cifar_new/c1_blockpt/23698257"),
        ("A4 no-gc",       "/home/acolombo/VAEs/checkpoint/cifar_new/c1_blockpt_nogc/23730612"),
        ("a7.1",           "/home/acolombo/VAEs/checkpoint/cifar_new/c1_a71_s42/23784121"),
        ("a7.1 + gc",      "/home/acolombo/VAEs/checkpoint/cifar_new/c1_a71gc_s42/23805018"),
    ],
    "mnist": [
        ("Euclidean",      "/home/acolombo/VAEs/checkpoint/mnist_new/euclidean/23519915"),
        ("Vanilla hyp",    "/home/acolombo/VAEs/checkpoint/mnist_new/c1/23519920"),
        ("A4 (riem+gc)",   "/home/acolombo/VAEs/checkpoint/mnist_new/c1_blockpt/23698256"),
        ("A4 no-gc",       "/home/acolombo/VAEs/checkpoint/mnist_new/c1_blockpt_nogc/23715571"),
        ("a7.1",           "/home/acolombo/VAEs/checkpoint/mnist_new/c1_a71_s42/23784115"),
        ("a7.1 + gc",      "/home/acolombo/VAEs/checkpoint/mnist_new/c1_a71gc_s42/23805011"),
    ],
}


def latest_best(ckpt_dir):
    cands = [f for f in os.listdir(ckpt_dir) if f.startswith("best_") and f.endswith(".pth")]
    if not cands:
        raise FileNotFoundError(f"no best_*.pth in {ckpt_dir}")
    cands.sort(key=lambda f: int(f[len("best_"):-len(".pth")]))
    return os.path.join(ckpt_dir, cands[-1])


def load_config(ckpt_dir):
    sys.modules.pop("config", None)
    sys.path.insert(0, ckpt_dir)
    import config as cfg
    importlib.reload(cfg)
    sys.path.pop(0)
    return cfg


def build_model(cfg, device):
    from mnist_vqvae import VQVAE2D
    ds = cfg.dataset
    if ds == "cifar100":
        in_channels, img_size, size = 3, 32, "large"
    elif ds == "emnist":
        in_channels, img_size, size = 1, 28, "medium"
    else:  # mnist
        in_channels, img_size, size = 1, 28, "small"

    g = lambda k, d: getattr(cfg, k, d)
    model = VQVAE2D(
        D=cfg.D, n_q=cfg.n_q, bins=cfg.bins, c=cfg.c,
        exponential_lambda=g("exponential_lambda", 0.0),
        uniform=g("uniform", False), ema=g("ema", False),
        kmeans_init=g("kmeans_init", False),
        threshold_ema_dead_code=g("threshold_ema_dead_code", 2),
        codebook_weight=g("codebook_weight", 1.0),
        commitment_weight=g("commitment_weight", 0.25),
        dot_product_weight=g("dot_product_weight", 0.0),
        entailment_cone_weight=g("entailment_cone_weight", 0.0),
        gyration_weight=g("gyration_weight", 0.0),
        in_channels=in_channels, img_size=img_size, size=size,
        new_method=g("new_method", True), approx=g("approx", False),
        hste=g("hste", False), hste_riemannian=g("hste_riemannian", False),
        hste_clip=g("hste_clip", False),
        gradient_correction=g("gradient_correction", False),
        block_hste_pt=g("block_hste_pt", False),
        A5=g("A5", False), A4_v2=g("A4_v2", False),
        a6=g("a6", False), a7=g("a7", False),
        a6_1=g("a6_1", False), a7_1=g("a7_1", False),
        a8=g("a8", False), solution=g("solution", False),
        gyration=g("gyration", False), full_grid=g("full_grid", False),
    ).to(device)

    state = torch.load(cfg._ckpt, map_location=device)
    state = state.get("model", state) if isinstance(state, dict) else state
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"quantizer.vq._enc_scale", "quantizer.vq._enc_calibrated"}
    leftover = set(missing) - allowed
    assert not leftover and not unexpected, f"key mismatch: missing={leftover} unexpected={unexpected}"
    model.eval()
    return model


def make_loader(cfg):
    transform = transforms.ToTensor()
    ds = cfg.dataset
    if ds == "mnist":
        data = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)
    elif ds == "emnist":
        data = datasets.EMNIST(root=cfg.data_dir, split=cfg.emnist_split, train=False, download=True, transform=transform)
    elif ds == "cifar100":
        data = datasets.CIFAR100(root=cfg.data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(ds)
    return DataLoader(data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def code_stats(model, loader, n_q, bins, device):
    freq = torch.zeros((n_q, bins), dtype=torch.float64, device=device)
    base = torch.tensor([bins ** i for i in range(n_q)], dtype=torch.long, device=device)  # (n_q,)
    tuple_ids = set()
    n_positions = 0
    for x, _ in loader:
        x = x.to(device)
        codes = model.encode(x)                 # (n_q, B, N)
        nq, B, N = codes.shape
        for q in range(nq):
            freq[q] += torch.bincount(codes[q].reshape(-1), minlength=bins).to(torch.float64)
        flat = codes.permute(1, 2, 0).reshape(-1, nq)   # (B*N, n_q)
        ids = (flat.to(torch.long) * base).sum(dim=1)    # unique id per tuple
        tuple_ids.update(ids.cpu().tolist())
        n_positions += flat.shape[0]

    probs = freq / freq.sum(dim=1, keepdim=True).clamp_min(1)
    ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)   # (n_q,)
    ppl = torch.exp2(ent).tolist()
    used_per_q = (freq > 0).sum(dim=1).tolist()
    return ppl, used_per_q, len(tuple_ids), n_positions


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for ds, exps in EXPERIMENTS.items():
        for name, ckpt_dir in exps:
            cfg = load_config(ckpt_dir)
            cfg._ckpt = latest_best(ckpt_dir)
            model = build_model(cfg, DEVICE)
            loader = make_loader(cfg)
            ppl, used, n_tuples, n_pos = code_stats(model, loader, cfg.n_q, cfg.bins, DEVICE)
            rows.append(dict(dataset=ds, name=name, n_q=cfg.n_q, bins=cfg.bins,
                             ppl=ppl, used=used, n_tuples=n_tuples, n_pos=n_pos,
                             ckpt=os.path.basename(cfg._ckpt)))
            ppl_s = ", ".join(f"{p:.1f}" for p in ppl)
            used_s = "/".join(str(u) for u in used)
            print(f"[{ds:8s}] {name:14s} n_q={cfg.n_q} bins={cfg.bins} "
                  f"ppl=[{ppl_s}] used=[{used_s}] tuples={n_tuples} (of {n_pos} positions) "
                  f"ckpt={os.path.basename(cfg._ckpt)}", flush=True)

    # Markdown tables
    for ds in EXPERIMENTS:
        dr = [r for r in rows if r["dataset"] == ds]
        if not dr:
            continue
        nq = dr[0]["n_q"]
        print(f"\n### {ds}  (n_q={nq}, bins={dr[0]['bins']})\n")
        head = "| experiment | " + " | ".join(f"ppl q{q}" for q in range(nq)) + " | mean ppl | distinct tuples |"
        sep = "|" + "---|" * (nq + 3)
        print(head)
        print(sep)
        for r in dr:
            ppl_cells = " | ".join(f"{p:.1f}" for p in r["ppl"])
            mean_ppl = sum(r["ppl"]) / len(r["ppl"])
            print(f"| {r['name']} | {ppl_cells} | {mean_ppl:.1f} | {r['n_tuples']:,} |")


if __name__ == "__main__":
    main()
