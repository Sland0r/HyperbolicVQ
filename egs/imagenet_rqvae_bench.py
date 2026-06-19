"""Benchmark per-epoch training time of an RQ-VAE (RQ-VAE-gen paper style) on
ImageNet-1k across batch sizes, to pick the best fit for an A100.

Model (representative of Lee et al., CVPR 2022 "Autoregressive Image Generation
using Residual Quantization", stage-1 RQ-VAE):
  - 256x256 input, VQGAN-style conv encoder/decoder, downsampling factor 32
    -> 8x8 latent feature map.
  - repo ResidualVectorQuantizer: depth D=4, codebook 16384, dim 256, c=0
    (Euclidean, matching the paper; hyperbolic c>0 costs ~the same here).
  - adversarial training: PatchGAN discriminator + VGG perceptual loss + L1 recon,
    two optimizer steps per iter (G then D) -- the real per-step cost.

For each batch size we run a few warmup steps then time N steps of the full
G+D training step under bf16 autocast (A100 standard), recording images/sec and
peak GPU memory, and extrapolate to one ImageNet epoch (1,281,167 train images).
OOM batch sizes are reported and skipped.
"""
import argparse, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, "/home/acolombo/VAEs")
from academicodec.quantization import ResidualVectorQuantizer

IMAGENET_TRAIN_SIZE = 1_281_167


# ----------------------- VQGAN-ish encoder / decoder ------------------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    """256 -> 8 (factor 32). ch_mult length 6 => 5 downsamples."""
    def __init__(self, ch=128, ch_mult=(1, 1, 2, 2, 4, 4), n_res=2, z_ch=256):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        chs = [ch * m for m in ch_mult]
        blocks = []
        c_prev = ch
        for i, c_cur in enumerate(chs):
            for _ in range(n_res):
                blocks.append(ResBlock(c_prev, c_cur)); c_prev = c_cur
            if i != len(chs) - 1:
                blocks.append(nn.Conv2d(c_prev, c_prev, 3, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)
        self.mid = nn.Sequential(ResBlock(c_prev, c_prev), ResBlock(c_prev, c_prev))
        self.norm_out = nn.GroupNorm(32, c_prev)
        self.conv_out = nn.Conv2d(c_prev, z_ch, 3, padding=1)

    def forward(self, x):
        h = self.mid(self.blocks(self.conv_in(x)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 1, 2, 2, 4, 4), n_res=2, z_ch=256):
        super().__init__()
        chs = [ch * m for m in ch_mult][::-1]
        c_prev = chs[0]
        self.conv_in = nn.Conv2d(z_ch, c_prev, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(c_prev, c_prev), ResBlock(c_prev, c_prev))
        blocks = []
        for i, c_cur in enumerate(chs):
            for _ in range(n_res):
                blocks.append(ResBlock(c_prev, c_cur)); c_prev = c_cur
            if i != len(chs) - 1:
                blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                blocks.append(nn.Conv2d(c_prev, c_prev, 3, padding=1))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(32, c_prev)
        self.conv_out = nn.Conv2d(c_prev, 3, 3, padding=1)

    def forward(self, z):
        h = self.blocks(self.mid(self.conv_in(z)))
        return torch.tanh(self.conv_out(F.silu(self.norm_out(h))))


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        m = 1
        for n in range(1, n_layers):
            m_prev, m = m, min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * m_prev, ndf * m, 4, 2, 1),
                       nn.GroupNorm(32, ndf * m), nn.LeakyReLU(0.2, True)]
        m_prev, m = m, min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * m_prev, ndf * m, 4, 1, 1),
                   nn.GroupNorm(32, ndf * m), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ndf * m, 1, 4, 1, 1)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class RQVAE(nn.Module):
    def __init__(self, n_q=4, bins=16384, dim=256, c=0.0, **rvq_kwargs):
        super().__init__()
        self.encoder = Encoder(z_ch=dim)
        self.decoder = Decoder(z_ch=dim)
        # c=0 -> Euclidean RVQ; c>0 -> hyperbolic HRQ. Extra rvq_kwargs select the
        # quantization variant: vanilla (none) or e.g. the A4 config
        # (new_method + block_hste_pt + hste_riemannian + gradient_correction).
        self.rvq = ResidualVectorQuantizer(dimension=dim, n_q=n_q, bins=bins, c=c,
                                           **rvq_kwargs)
        self.n_q = n_q

    def forward(self, x):
        z = self.encoder(x)                       # (B, dim, 8, 8)
        B, C, H, W = z.shape
        z_flat = z.reshape(B, C, H * W)           # (B, dim, 64)
        q, codes, bw, commit_loss, dist = self.rvq(z_flat, 1, None, nq=self.n_q)
        q = q.reshape(B, C, H, W)
        x_hat = self.decoder(q)
        return x_hat, commit_loss


def perceptual_loss(vgg, x, y):
    # x,y in [-1,1] -> [0,1]; frozen VGG features L1 (LPIPS-style cost)
    feats_x = vgg((x + 1) / 2)
    feats_y = vgg((y + 1) / 2)
    return F.l1_loss(feats_x, feats_y)


def benchmark(bs, device, n_warmup, n_iter):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    g = RQVAE().to(device)
    d = PatchDiscriminator().to(device)
    vgg = torchvision.models.vgg16(weights=None).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    opt_g = torch.optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def step():
        x = torch.randn(bs, 3, 256, 256, device=device)
        # ---- G step ----
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_hat, commit = g(x)
            rec = F.l1_loss(x_hat, x)
            perc = perceptual_loss(vgg, x, x_hat)
            g_adv = -d(x_hat).mean()
            g_loss = rec + perc + 0.1 * g_adv + commit.mean()
        g_loss.backward(); opt_g.step()
        # ---- D step ----
        opt_d.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_real = d(x)
            logits_fake = d(x_hat.detach())
            d_loss = (F.relu(1 - logits_real).mean() + F.relu(1 + logits_fake).mean())
        d_loss.backward(); opt_d.step()

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    img_per_s = bs * n_iter / dt
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    epoch_s = IMAGENET_TRAIN_SIZE / img_per_s
    del g, d, vgg, opt_g, opt_d
    return img_per_s, peak_gb, epoch_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()
    device = "cuda"

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    g = RQVAE(); d = PatchDiscriminator()
    n_params = (sum(p.numel() for p in g.parameters()) + sum(p.numel() for p in d.parameters()))
    del g, d
    print(f"GPU: {name}  ({total_gb:.0f} GB)")
    print(f"Model: RQ-VAE 256x256 -> 8x8, D=4, codebook=16384  |  G+D params: {n_params/1e6:.1f}M")
    print(f"Precision: bf16 autocast | full adversarial step (G+D) | recon+VGG-perceptual+commit")
    print(f"ImageNet train images: {IMAGENET_TRAIN_SIZE:,}\n")
    print(f"{'batch':>6} {'img/s':>9} {'peak GB':>9} {'%mem':>6} {'epoch':>12}")
    print("-" * 48)

    results = []
    for bs in args.batch_sizes:
        try:
            ips, peak, epoch_s = benchmark(bs, device, args.warmup, args.iters)
            h = int(epoch_s // 3600); m = int((epoch_s % 3600) // 60)
            pct = 100 * peak / total_gb
            print(f"{bs:>6} {ips:>9.1f} {peak:>9.1f} {pct:>5.0f}% {h:>5}h{m:02d}m")
            results.append((bs, ips, peak, pct, epoch_s))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:>6} {'OOM':>9}")
                torch.cuda.empty_cache()
            else:
                raise

    if results:
        # best = highest throughput that stays under ~90% memory
        safe = [r for r in results if r[3] <= 90]
        pick = max(safe or results, key=lambda r: r[1])
        bs, ips, peak, pct, epoch_s = pick
        h = int(epoch_s // 3600); m = int((epoch_s % 3600) // 60)
        print("\n" + "=" * 48)
        print(f"RECOMMENDED batch size: {bs}")
        print(f"  throughput : {ips:.1f} img/s")
        print(f"  peak memory: {peak:.1f} GB ({pct:.0f}% of {total_gb:.0f} GB)")
        print(f"  ~1 epoch   : {h}h{m:02d}m")
        print("=" * 48)


if __name__ == "__main__":
    main()
