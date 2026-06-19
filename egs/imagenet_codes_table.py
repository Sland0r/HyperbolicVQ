"""Test(val)-set codebook statistics for the ImageNet RQ-VAE checkpoints.

Same table as the MNIST/CIFAR codes table, for the experiments already run:
Euclidean, hyperbolic (vanilla), and A4+gc. For each checkpoint, push the
ImageNet val set through the encoder + RVQ and report:
  * per-codebook perplexity  ppl_q = 2 ** entropy(val-set freq over `bins`)
  * total distinct multi-level code tuples (effective vocabulary): a "tuple" is
    the (n_q,) vector of indices at one 8x8 spatial position.

Variant flags only affect the backward STE; encode() is plain encoder +
nearest-code assignment, so they don't change the codes produced here.
"""
import os, sys, glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, "/home/acolombo/VAEs")
from egs.imagenet_rqvae_bench import RQVAE

VAL_DIR = "/scratch-shared/acolombo/imagenet/ILSVRC/Data/CLS-LOC/val"
CKPTS = [
    ("Euclidean",  "/home/acolombo/VAEs/checkpoint/imagenet_rqvae_euc/latest.pth"),
    ("Hyperbolic", "/home/acolombo/VAEs/checkpoint/imagenet_rqvae_hyp/latest.pth"),
    ("A4 + gc",    "/home/acolombo/VAEs/checkpoint/imagenet_rqvae_a4gc/latest.pth"),
]
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))  # 0 = all val images
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
])


class ValImages(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return tfm(Image.open(self.paths[i]).convert("RGB"))


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    g = RQVAE(n_q=a["n_q"], bins=a["bins"], dim=a["dim"], c=a["c"]).to(DEVICE)
    missing, unexpected = g.load_state_dict(ckpt["g"], strict=False)
    leftover = [k for k in missing if not k.endswith(("_enc_scale", "_enc_calibrated"))]
    assert not leftover and not unexpected, f"key mismatch: missing={leftover} unexpected={unexpected}"
    g.eval()
    return g, a, ckpt.get("global_step")


@torch.no_grad()
def code_stats(g, n_q, bins, loader):
    freq = torch.zeros((n_q, bins), dtype=torch.float64, device=DEVICE)
    base = torch.tensor([bins ** i for i in range(n_q)], dtype=torch.long, device=DEVICE)
    tuple_ids = set()
    n_pos = 0
    for x in loader:
        x = x.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = g.encoder(x)                          # (B, dim, 8, 8)
            B, C, H, W = z.shape
            z_flat = z.reshape(B, C, H * W)           # (B, dim, 64)
            _, codes, _, _, _ = g.rvq(z_flat, 1, None, nq=n_q)   # codes: (n_q, B, N)
        codes = codes.long()
        nq, B, N = codes.shape
        for q in range(nq):
            freq[q] += torch.bincount(codes[q].reshape(-1), minlength=bins).to(torch.float64)
        flat = codes.permute(1, 2, 0).reshape(-1, nq)     # (B*N, n_q)
        ids = (flat * base).sum(dim=1)
        tuple_ids.update(ids.cpu().tolist())
        n_pos += flat.shape[0]

    probs = freq / freq.sum(dim=1, keepdim=True).clamp_min(1)
    ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)
    ppl = torch.exp2(ent).tolist()
    used = (freq > 0).sum(dim=1).tolist()
    return ppl, used, len(tuple_ids), n_pos


def main():
    paths = sorted(glob.glob(os.path.join(VAL_DIR, "*.JPEG")))
    if MAX_IMAGES:
        paths = paths[:MAX_IMAGES]
    print(f"val images: {len(paths):,}", flush=True)
    loader = DataLoader(ValImages(paths), batch_size=BATCH, shuffle=False,
                        num_workers=12, pin_memory=True)

    rows = []
    for name, path in CKPTS:
        g, a, gstep = load_model(path)
        ppl, used, n_tuples, n_pos = code_stats(g, a["n_q"], a["bins"], loader)
        rows.append((name, a, ppl, used, n_tuples, n_pos))
        ppl_s = ", ".join(f"{p:.1f}" for p in ppl)
        print(f"[{name:11s}] c={a['c']} n_q={a['n_q']} bins={a['bins']} dim={a['dim']} "
              f"gstep={gstep} ppl=[{ppl_s}] used={used} tuples={n_tuples:,} "
              f"(of {n_pos:,} positions)", flush=True)
        del g
        torch.cuda.empty_cache()

    nq = rows[0][1]["n_q"]
    bins = rows[0][1]["bins"]
    print(f"\n### ImageNet val  (n_q={nq}, bins={bins})\n")
    print("| experiment | " + " | ".join(f"ppl q{q}" for q in range(nq)) + " | mean ppl | distinct tuples |")
    print("|" + "---|" * (nq + 3))
    for name, a, ppl, used, n_tuples, n_pos in rows:
        cells = " | ".join(f"{p:.1f}" for p in ppl)
        mean_ppl = sum(ppl) / len(ppl)
        print(f"| {name} | {cells} | {mean_ppl:.1f} | {n_tuples:,} |")


if __name__ == "__main__":
    main()
