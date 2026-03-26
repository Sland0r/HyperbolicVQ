"""Training script for MNIST VQ-VAE with residual quantization."""
import argparse
import glob
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    # dataset
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar100"],
                        help="dataset to train on")
    # model
    parser.add_argument("--D", type=int, default=128, help="codebook / latent dimension")
    parser.add_argument("--n_q", type=int, default=4, help="number of residual quantizers")
    parser.add_argument("--bins", type=int, default=256, help="codebook size per quantizer")
    parser.add_argument("--c", type=float, default=0.0, help="curvature for hyperbolic quantization (0=Euclidean)")
    parser.add_argument("--ema", action="store_true", help="use EMA codebook updates")
    parser.add_argument("--kmeans_init", action="store_true", help="initialise codebooks with k-means")
    # training
    parser.add_argument("--N_EPOCHS", type=int, default=50, help="number of training epochs")
    parser.add_argument("--BATCH_SIZE", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--LAMBDA_COM", type=float, default=1.0, help="commitment loss weight")
    parser.add_argument("--print_freq", type=int, default=100, help="print every N batches")
    parser.add_argument("--codebook_number", type=int, default=0, help="which codebook to visualize")
    parser.add_argument("--number_of_steps", type=int, default=50, help="save codebook snapshot every N batches")
    # paths
    parser.add_argument("--PATH", type=str, default="/home/acolombo/VAEs/checkpoint/mnist_vqvae",
                        help="checkpoint save directory")
    parser.add_argument("--data_dir", type=str, default="",
                        help="dataset directory (auto-set if empty)")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default="", help="path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # create unique run dir
    if args.resume:
        args.PATH = args.resume_path
    else:
        if "SLURM_JOB_ID" in os.environ:
            time_str = os.environ["SLURM_JOB_ID"]
        else:
            time_str = time.strftime("%Y-%m-%d-%H-%M")
        args.PATH = os.path.join(args.PATH, time_str)
    os.makedirs(args.PATH, exist_ok=True)

    # Set default data_dir based on dataset
    if not args.data_dir:
        base = "/home/acolombo/VAEs/dataset"
        args.data_dir = os.path.join(base, "MNIST" if args.dataset == "mnist" else "CIFAR100")

    return args


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("All arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Save config to checkpoint directory
    config_path = os.path.join(args.PATH, "config.py")
    with open(config_path, "w") as f:
        f.write("# Training hyperparameters\n")
        for k, v in vars(args).items():
            f.write(f"{k} = {v!r}\n")
    print(f"Config saved to: {config_path}")

    # ── Data ──────────────────────────────────────────────────────────
    transform = transforms.ToTensor()  # scales to [0, 1]
    if args.dataset == "mnist":
        in_channels, img_size = 1, 28
        train_data = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == "cifar100":
        in_channels, img_size = 3, 32
        train_data = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
        val_data = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    from mnist_vqvae import VQVAE2D, _count_params

    model = VQVAE2D(
        D=args.D, n_q=args.n_q, bins=args.bins,
        c=args.c, ema=args.ema, kmeans_init=args.kmeans_init,
        in_channels=in_channels, img_size=img_size,
    ).to(device)
    print(model)
    total, trainable = _count_params(model)
    print(f"Model params — total: {total:,}  trainable: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    st_epoch = 1
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_path, "latest.pth"), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        st_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Codebook visualisation setup ────────────────────────────────
    from academicodec.visualization import fit_pca, plot_codes, create_gif
    pca_fitted = False
    pca_model = None
    codebook_plots_dir = os.path.join(args.PATH, "codebook_plots")
    global_step = 0

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = {"train_rec": [], "train_com": [], "val_rec": [], "val_com": [], "val_total": []}
    pbar = tqdm(range(st_epoch, args.N_EPOCHS + 1), desc="Training")
    for epoch in pbar:
        model.train()
        train_rec, train_com, n_batches = 0.0, 0.0, 0

        for batch_idx, (imgs, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device)

            x_hat, commit_loss, codes = model(imgs)
            rec_loss = F.mse_loss(x_hat, imgs)
            loss = rec_loss + args.LAMBDA_COM * commit_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            train_rec += rec_loss.item()
            train_com += commit_loss.item()
            n_batches += 1

            # ── Codebook snapshot ──────────────────────────────────
            if not pca_fitted:
                codebook_module = model.quantizer.vq.layers[args.codebook_number]._codebook
                if hasattr(codebook_module, "inited") and codebook_module.inited:
                    codes_cb = codebook_module.embed.detach().cpu()
                    pca_model = fit_pca(codes_cb, args.c)
                    os.makedirs(codebook_plots_dir, exist_ok=True)
                    pca_fitted = True

            if pca_fitted and (global_step % args.number_of_steps == 0):
                codebook_module = model.quantizer.vq.layers[args.codebook_number]._codebook
                codes_cb = codebook_module.embed.detach().cpu()
                plot_codes(codes_cb, pca_model, args.c, global_step, codebook_plots_dir)

            if batch_idx % args.print_freq == 0:
                print(f"  [epoch {epoch}, iter {batch_idx}] rec={rec_loss.item():.5f}  "
                      f"commit={commit_loss.item():.5f}  total={loss.item():.5f}")

        scheduler.step()
        train_rec /= n_batches
        train_com /= n_batches

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_rec, val_com, val_n = 0.0, 0.0, 0
        codes_hist = None
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                x_hat, commit_loss, codes = model(imgs)
                rec_loss = F.mse_loss(x_hat, imgs)
                val_rec += rec_loss.item()
                val_com += commit_loss.item()
                val_n += 1

                # Accumulate code usage: codes shape is (n_q, B, N)
                if codes_hist is None:
                    codes_hist = torch.zeros(codes.shape[0], args.bins, device=device)
                for q in range(codes.shape[0]):
                    codes_hist[q].scatter_add_(0, codes[q].flatten(), torch.ones(codes[q].numel(), device=device))

        val_rec /= val_n
        val_com /= val_n
        val_total = val_rec + args.LAMBDA_COM * val_com

        history["train_rec"].append(train_rec)
        history["train_com"].append(train_com)
        history["val_rec"].append(val_rec)
        history["val_com"].append(val_com)
        history["val_total"].append(val_total)

        pbar.set_postfix(train_rec=f"{train_rec:.4f}", val_rec=f"{val_rec:.4f}", val_total=f"{val_total:.4f}")

        print(f"Epoch {epoch}: train_rec={train_rec:.5f}  train_com={train_com:.5f}  |  "
              f"val_rec={val_rec:.5f}  val_com={val_com:.5f}  val_total={val_total:.5f}")

        # Print cluster sizes per quantizer
        if codes_hist is not None:
            for q in range(codes_hist.shape[0]):
                sizes = codes_hist[q].long().tolist()
                sizes = [s / sum(sizes) for s in sizes] 
                entropy = -sum(p * math.log2(p) for p in sizes if p > 0)
                perplexity = 2 ** entropy
                max_perplexity = args.bins
                print(f"  Q{q} perplexity: {perplexity:.2f} / {max_perplexity}", end=" -> ")
            print()

        # ── Checkpointing ─────────────────────────────────────────────
        # Only save checkpoints after reaching 75% of total epochs
        threshold_epoch = int(0.75 * args.N_EPOCHS)
        if epoch >= threshold_epoch:
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_rec": val_rec,
            }
            # Save latest checkpoint
            torch.save(save_dict, os.path.join(args.PATH, "latest.pth"))

            # Save best checkpoint if improved
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(save_dict, os.path.join(args.PATH, f"best_{epoch}.pth"))
                print(f"  ✓ New best model saved (val_total={val_total:.5f})")

    # ── End-of-training evaluation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Training complete — Final evaluation")
    print("=" * 60)

    # Last model validation loss (already computed in the last epoch)
    print(f"\n  Last model  (epoch {epoch}):  val_rec={val_rec:.5f}  "
          f"val_com={val_com:.5f}  val_total={val_total:.5f}")

    # Best model validation loss
    best_ckpts = sorted(glob.glob(os.path.join(args.PATH, "best_*.pth")))
    if best_ckpts:
        best_path = best_ckpts[-1]  # latest best checkpoint
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model"])
        model.eval()

        best_val_rec, best_val_com, best_val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                x_hat, commit_loss, codes = model(imgs)
                rec_loss = F.mse_loss(x_hat, imgs)
                best_val_rec += rec_loss.item()
                best_val_com += commit_loss.item()
                best_val_n += 1
        best_val_rec /= best_val_n
        best_val_com /= best_val_n
        best_val_total = best_val_rec + args.LAMBDA_COM * best_val_com
        best_epoch = best_ckpt["epoch"]

        print(f"  Best model  (epoch {best_epoch}):  val_rec={best_val_rec:.5f}  "
              f"val_com={best_val_com:.5f}  val_total={best_val_total:.5f}")
        print(f"  Loaded from: {best_path}")
    else:
        print("  No best checkpoint found (training may not have reached 75% of epochs).")

    # ── Plot examples: original vs reconstructed ─────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_examples = 8
    model.eval()
    # grab one batch from val set
    sample_imgs, _ = next(iter(val_loader))
    sample_imgs = sample_imgs[:n_examples].to(device)
    with torch.no_grad():
        recon_imgs, _, _ = model(sample_imgs)

    sample_imgs = sample_imgs.cpu()
    recon_imgs = recon_imgs.cpu()

    fig, axes = plt.subplots(2, n_examples, figsize=(2 * n_examples, 4))
    for i in range(n_examples):
        img_orig = sample_imgs[i].permute(1, 2, 0).squeeze()  # (H,W) or (H,W,3)
        img_recon = recon_imgs[i].permute(1, 2, 0).squeeze()
        cmap = "gray" if in_channels == 1 else None
        axes[0, i].imshow(img_orig, cmap=cmap, vmin=0, vmax=1)
        axes[0, i].set_title("Original", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(img_recon, cmap=cmap, vmin=0, vmax=1)
        axes[1, i].set_title("Recon", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("Original vs Reconstructed (best model)", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(args.PATH, "reconstruction_examples.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n  Reconstruction examples saved to: {fig_path}")

    # ── Loss curves ──────────────────────────────────────────────
    epochs_range = list(range(st_epoch, st_epoch + len(history["train_rec"])))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history["train_rec"], label="Train")
    axes[0].plot(epochs_range, history["val_rec"], label="Val")
    axes[0].set_title("Reconstruction Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_com"], label="Train")
    axes[1].plot(epochs_range, history["val_com"], label="Val")
    axes[1].set_title("Commitment Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    train_total = [r + args.LAMBDA_COM * c for r, c in zip(history["train_rec"], history["train_com"])]
    axes[2].plot(epochs_range, train_total, label="Train")
    axes[2].plot(epochs_range, history["val_total"], label="Val")
    axes[2].set_title("Total Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    loss_fig_path = os.path.join(args.PATH, "loss_curves.png")
    fig.savefig(loss_fig_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curves saved to: {loss_fig_path}")

    # ── Final codebook snapshot + GIF ────────────────────────────
    if pca_fitted:
        codebook_module = model.quantizer.vq.layers[args.codebook_number]._codebook
        codes_cb = codebook_module.embed.detach().cpu()
        plot_codes(codes_cb, pca_model, args.c, global_step, codebook_plots_dir)

    gif_path = os.path.join(args.PATH, "codebook_evolution.gif")
    create_gif(codebook_plots_dir, gif_path)
    print(f"  Codebook evolution GIF saved to: {gif_path}")


if __name__ == "__main__":
    main()
