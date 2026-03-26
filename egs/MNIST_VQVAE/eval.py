"""Evaluation script for MNIST VQ-VAE: computes MSE, codebook perplexity, and saves reconstruction grid."""
import argparse
import os
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MNIST VQ-VAE")
    # model
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--n_q", type=int, default=4)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--kmeans_init", action="store_true")
    # eval
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, required=True, help="path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="/home/acolombo/VAEs/dataset/MNIST")
    parser.add_argument("--output_dir", type=str, default="/home/acolombo/VAEs/logs/mnist_eval",
                        help="directory to save reconstruction grid")
    parser.add_argument("--n_recon", type=int, default=16, help="number of images in reconstruction grid")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4)

    # ── Model ─────────────────────────────────────────────────────────
    from mnist_vqvae import MNISTVQVAE

    model = MNISTVQVAE(
        D=args.D, n_q=args.n_q, bins=args.bins,
        c=args.c, ema=args.ema, kmeans_init=args.kmeans_init,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────
    total_mse = 0.0
    n_batches = 0
    codes_hist = None
    total_tokens = 0
    recon_originals = None
    recon_generated = None

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            x_hat, commit_loss, codes = model(imgs)
            mse = F.mse_loss(x_hat, imgs)
            total_mse += mse.item()
            n_batches += 1

            # accumulate codebook usage histogram
            # codes shape: (n_q, B, N)
            if codes_hist is None:
                codes_hist = torch.zeros(args.n_q, args.bins, device=device, dtype=torch.float64)
            for q in range(codes.shape[0]):
                codes_q = codes[q].flatten()
                codes_hist[q].scatter_add_(0, codes_q, torch.ones_like(codes_q, dtype=torch.float64))
            total_tokens += codes.shape[1] * codes.shape[2]

            # save first batch for reconstruction grid
            if recon_originals is None:
                n = min(args.n_recon, imgs.size(0))
                recon_originals = imgs[:n].cpu()
                recon_generated = x_hat[:n].cpu()

    avg_mse = total_mse / n_batches

    # ── Perplexity ────────────────────────────────────────────────────
    perplexities = []
    usage_fracs = []
    if codes_hist is not None:
        probs = codes_hist / codes_hist.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
        perplexities = torch.exp2(entropy).tolist()
        used = (codes_hist > 0).sum(dim=-1).tolist()
        usage_fracs = [u / args.bins for u in used]

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test MSE:          {avg_mse:.6f}")
    print(f"  Test RMSE:         {math.sqrt(avg_mse):.6f}")
    for q, (ppl, usage) in enumerate(zip(perplexities, usage_fracs)):
        print(f"  Quantizer {q}: perplexity={ppl:.1f}  usage={usage:.1%}")
    print("=" * 60)

    # ── Reconstruction grid ───────────────────────────────────────────
    if recon_originals is not None:
        # interleave originals and reconstructions row by row
        grid = torch.stack([recon_originals, recon_generated], dim=1).flatten(0, 1)
        grid_path = os.path.join(args.output_dir, "reconstruction_grid.png")
        save_image(grid, grid_path, nrow=args.n_recon, padding=2)
        print(f"  Reconstruction grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
