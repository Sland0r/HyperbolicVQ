"""Multi-GPU (DDP) training of the paper-style RQ-VAE on ImageNet-1k.

Reuses the exact model that was benchmarked in `imagenet_rqvae_bench.py`
(VQGAN-style enc/dec -> 8x8 latent, residual depth D=4, codebook 16384, c=0
Euclidean by default) plus the full adversarial step (PatchGAN D + VGG
perceptual + L1 recon + commit), bf16 autocast.

Launch with torchrun (one process per GPU), single node:

    torchrun --standalone --nproc_per_node=4 egs/imagenet_rqvae_ddp.py \
        --data /scratch-shared/acolombo/imagenet/ILSVRC/Data/CLS-LOC/train \
        --save_dir /home/acolombo/VAEs/checkpoint/imagenet_rqvae_euc \
        --epochs 1 --batch_size 16

Checkpointing / resume:
  - rank 0 writes `<save_dir>/latest.pth` every --ckpt_every optimizer steps and
    at the end of every epoch. It holds G, D, both optimizers, the epoch, the
    step-within-epoch and the global step.
  - `--resume` auto-loads `<save_dir>/latest.pth` if present and continues from
    exactly where it stopped (mid-epoch batches are skipped deterministically,
    since the DistributedSampler ordering is fixed by set_epoch + world_size).
    Resuming is a no-op if no checkpoint exists, so the same command works for a
    fresh start and a restart.
"""
import argparse, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms
import geoopt

sys.path.insert(0, "/home/acolombo/VAEs")
# Reuse the *exact* benchmarked model so training matches the timing study.
from egs.imagenet_rqvae_bench import RQVAE, PatchDiscriminator, perceptual_loss


# ------------------------------- DDP helpers -------------------------------
def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


# ------------------------------- checkpoint --------------------------------
def save_ckpt(path, g, d, opt_g, opt_d, epoch, step_in_epoch, global_step, args):
    if not is_main():
        return
    tmp = path + ".tmp"
    torch.save({
        "g": g.module.state_dict(),
        "d": d.module.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)  # atomic: never leave a half-written latest.pth


def load_ckpt(path, g, d, opt_g, opt_d, device):
    ckpt = torch.load(path, map_location=device)
    g.module.load_state_dict(ckpt["g"])
    d.module.load_state_dict(ckpt["d"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt["epoch"], ckpt["step_in_epoch"], ckpt["global_step"]


# --------------------------------- train -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default="/scratch-shared/acolombo/imagenet/ILSVRC/Data/CLS-LOC/train")
    ap.add_argument("--save_dir", type=str,
                    default="/home/acolombo/VAEs/checkpoint/imagenet_rqvae_euc")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16, help="per-GPU batch size")
    ap.add_argument("--num_workers", type=int, default=12, help="dataloader workers per GPU")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_manifold", type=float, default=1e-3,
                    help="lr for geoopt manifold (codebook) params when c>0")
    ap.add_argument("--geoopt_eps", type=float, default=1e-5)
    ap.add_argument("--c", type=float, default=0.0, help="Poincare curvature (0 = Euclidean)")
    ap.add_argument("--n_q", type=int, default=4)
    ap.add_argument("--bins", type=int, default=16384)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--adv_weight", type=float, default=0.1)
    # Quantization-variant flags (forwarded to ResidualVectorQuantizer).
    # A4 config = --new_method --block_hste_pt --hste_riemannian --gradient_correction
    ap.add_argument("--new_method", action="store_true", default=False)
    ap.add_argument("--block_hste_pt", action="store_true", default=False)
    ap.add_argument("--hste", action="store_true", default=False)
    ap.add_argument("--hste_riemannian", action="store_true", default=False)
    ap.add_argument("--gradient_correction", action="store_true", default=False)
    ap.add_argument("--ckpt_every", type=int, default=1000, help="save every N optimizer steps")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <save_dir>/latest.pth if it exists")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="stop after this many optimizer steps (0 = no cap); for smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, local_rank, world = ddp_setup()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "latest.pth")

    log(f"World size: {world} GPUs | per-GPU batch {args.batch_size} "
        f"-> effective batch {args.batch_size * world} | c={args.c}")

    # ---- data ----
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])
    dataset = torchvision.datasets.ImageFolder(args.data, transform=tfm)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.num_workers > 0)
    steps_per_epoch = len(loader)
    log(f"Dataset: {len(dataset):,} images | {steps_per_epoch:,} steps/epoch/GPU")

    # ---- models ----
    # c=0 -> Euclidean RVQ; c>0 -> hyperbolic HRQ. Only forward variant flags that
    # are explicitly on, so unset flags keep ResidualVectorQuantizer's own defaults
    # (notably new_method=True, which the vanilla-hyperbolic run relies on).
    rvq_kwargs = {k: True for k in
                  ("new_method", "block_hste_pt", "hste", "hste_riemannian",
                   "gradient_correction") if getattr(args, k)}
    if rvq_kwargs:
        log(f"RVQ variant flags: {sorted(rvq_kwargs)}")
    g = RQVAE(n_q=args.n_q, bins=args.bins, dim=args.dim, c=args.c, **rvq_kwargs).to(device)
    d = PatchDiscriminator().to(device)
    vgg = torchvision.models.vgg16(weights=None).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    g = DDP(g, device_ids=[local_rank], find_unused_parameters=False)
    d = DDP(d, device_ids=[local_rank], find_unused_parameters=False)

    if args.c > 0:
        # Codebook lives on the Poincare ball (geoopt.ManifoldParameter) -> needs a
        # Riemannian optimizer; everything else stays Euclidean. Mirrors MNIST train.py.
        manifold_params = [p for p in g.parameters() if hasattr(p, "manifold")]
        euclidean_params = [p for p in g.parameters() if not hasattr(p, "manifold")]
        param_groups = []
        if manifold_params:
            param_groups.append({"params": manifold_params, "lr": args.lr_manifold,
                                 "betas": (0.0, 0.95), "eps": args.geoopt_eps})
        param_groups.append({"params": euclidean_params, "lr": args.lr, "betas": (0.5, 0.9)})
        opt_g = geoopt.optim.RiemannianAdam(param_groups)
        log(f"RiemannianAdam: {len(manifold_params)} manifold + "
            f"{len(euclidean_params)} euclidean param tensors (lr_manifold={args.lr_manifold})")
    else:
        opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # ---- resume ----
    start_epoch, skip_steps, global_step = 0, 0, 0
    if args.resume and os.path.exists(ckpt_path):
        start_epoch, skip_steps, global_step = load_ckpt(ckpt_path, g, d, opt_g, opt_d, device)
        log(f"Resumed from {ckpt_path}: epoch {start_epoch}, "
            f"step_in_epoch {skip_steps}, global_step {global_step}")
    elif args.resume:
        log(f"--resume set but no checkpoint at {ckpt_path}; starting fresh")

    def train_step(x):
        x = x.to(device, non_blocking=True)
        # ---- G step (freeze D so DDP doesn't reduce D grads here) ----
        for p in d.parameters():
            p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_hat, commit = g(x)
            rec = F.l1_loss(x_hat, x)
            perc = perceptual_loss(vgg, x, x_hat)
            g_adv = -d(x_hat).mean()
            g_loss = rec + perc + args.adv_weight * g_adv + commit.mean()
        g_loss.backward()
        opt_g.step()
        for p in d.parameters():
            p.requires_grad_(True)
        # ---- D step ----
        opt_d.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_real = d(x)
            logits_fake = d(x_hat.detach())
            d_loss = F.relu(1 - logits_real).mean() + F.relu(1 + logits_fake).mean()
        d_loss.backward()
        opt_d.step()
        return rec.item(), perc.item(), commit.mean().item(), d_loss.item()

    g.train(); d.train()
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, (x, _) in enumerate(loader):
            # Deterministically skip already-completed batches on a resumed epoch.
            if epoch == start_epoch and i < skip_steps:
                continue
            rec, perc, commit, d_loss = train_step(x)
            global_step += 1

            if global_step % args.log_every == 0:
                ips = (global_step) * args.batch_size * world / (time.time() - t0)
                log(f"ep {epoch} step {i + 1}/{steps_per_epoch} gstep {global_step} "
                    f"| rec {rec:.4f} perc {perc:.4f} commit {commit:.4f} d {d_loss:.4f} "
                    f"| {ips:.1f} img/s")

            if global_step % args.ckpt_every == 0:
                save_ckpt(ckpt_path, g, d, opt_g, opt_d, epoch, i + 1, global_step, args)
                log(f"  checkpoint @ gstep {global_step} -> {ckpt_path}")

            if args.max_steps and global_step >= args.max_steps:
                save_ckpt(ckpt_path, g, d, opt_g, opt_d, epoch, i + 1, global_step, args)
                log(f"  reached max_steps={args.max_steps}; checkpoint saved, stopping")
                dist.barrier()
                dist.destroy_process_group()
                return
        # end of epoch: next resume should start the following epoch cleanly
        save_ckpt(ckpt_path, g, d, opt_g, opt_d, epoch + 1, 0, global_step, args)
        log(f"  end of epoch {epoch}: checkpoint saved (global_step {global_step})")

    log(f"Done. {args.epochs} epoch(s), {global_step} optimizer steps, "
        f"{(time.time() - t0) / 3600:.2f} h.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
