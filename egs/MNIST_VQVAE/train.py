"""Training script for MNIST VQ-VAE with residual quantization."""
import argparse
import glob
import math
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geoopt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    # args for random
    parser.add_argument(
        '--seed', type=int, default=42,
        help='seed for initializing training')
    parser.add_argument(
        '--cudnn_deterministic', action='store_true',
        help='set cudnn.deterministic True')
    parser.add_argument(
        '--tensorboard', action='store_true',
        help='use tensorboard for logging')
    # args for dataset
    parser.add_argument(
        '--dataset', type=str, default='mnist',
        choices=['mnist', 'cifar100', 'emnist'],
        help='dataset to train on')
    parser.add_argument(
        '--emnist_split', type=str, default='byclass',
        choices=['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'],
        help='EMNIST split (only used when --dataset=emnist)')
    parser.add_argument(
        '--data_dir', type=str, default='',
        help='dataset directory (auto-set if empty)')
    # args for model
    parser.add_argument(
        '--D', type=int, default=128,
        help='codebook / latent dimension')
    parser.add_argument(
        '--n_q', type=int, default=4,
        help='number of residual quantizers')
    parser.add_argument(
        '--bins', type=int, default=256,
        help='codebook size per quantizer')
    parser.add_argument(
        '--c', type=float, default=0.0,
        help='curvature for hyperbolic quantization (0=Euclidean)')
    parser.add_argument(
        '--ema', action='store_true',
        help='use EMA for codebook (default: False)')
    parser.add_argument(
        '--kmeans_init', action='store_true',
        help='use kmeans_init for codebook (default: False)')
    parser.add_argument(
        '--pre_quant_batchnorm', action='store_true',
        help='apply BatchNorm1d on encoder output right before quantization')
    parser.add_argument(
        '--exponential_lambda', type=float, default=0.0,
        help='exponential_lambda of codebook dropout')
    parser.add_argument(
        '--remove', type=int, default=0,
        help='number of codebooks to remove (default: 0)')
    # args for training
    parser.add_argument(
        '--N_EPOCHS', type=int, default=50,
        help='Total training epochs')
    parser.add_argument(
        '--st_epoch', type=int, default=1,
        help='start training epoch')
    parser.add_argument(
        '--global_step', type=int, default=0,
        help='record the global step')
    parser.add_argument(
        '--BATCH_SIZE', type=int, default=128,
        help='batch size')
    parser.add_argument(
        '--LAMBDA_COM', type=float, default=1.0,
        help='hyper-parameter for commit loss')
    parser.add_argument(
        '--print_freq', type=int, default=100,
        help='print every N batches')
    parser.add_argument(
        '--codebook_number', type=int, default=0,
        help='which codebook to visualize (default: 0)')
    parser.add_argument(
        '--number_of_steps', type=int, default=50,
        help='save codebook every number_of_steps batches (default: 50)')
    # args for learning rate
    parser.add_argument(
        '--lr_g', type=float, default=3e-4,
        help='base learning rate for generator and euclidean parameters')
    parser.add_argument(
        '--lr_manifold', type=float, default=1e-3,
        help='learning rate for manifold parameters (geoopt)')
    parser.add_argument(
        '--geoopt_eps', type=float, default=1e-5,
        help='epsilon for geoopt RiemannianAdam denominator stability')
    parser.add_argument(
        '--geoopt_stabilize', type=int, default=1,
        help='apply manifold stabilization every N steps (1 = every step)')
    parser.add_argument(
        '--quantizer_grad_clip', type=float, default=0.05,
        help='max norm for quantizer/codebook gradients before global clipping')
    parser.add_argument(
        '--manifold_grad_clip', type=float, default=0.02,
        help='max norm for manifold quantizer gradients before optimizer step')
    parser.add_argument(
        '--warmup_epochs_g', type=int, default=0,
        help='number of linear warmup epochs for generator LR scheduler')
    # args for paths
    parser.add_argument(
        '--PATH', type=str,
        default='/home/acolombo/VAEs/checkpoint/mnist_vqvae',
        help='The path to save the model')
    parser.add_argument(
        '--resume', action='store_true',
        help='whether re-train model')
    parser.add_argument(
        '--resume_path', type=str, default='',
        help='resume_path')
    args = parser.parse_args()

    # create unique run dir
    if args.resume:
        args.PATH = args.resume_path
    else:
        if 'SLURM_JOB_ID' in os.environ:
            time_str = os.environ['SLURM_JOB_ID']
        else:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
        args.PATH = os.path.join(args.PATH, time_str)
    args.save_dir = args.PATH
    os.makedirs(args.PATH, exist_ok=True)

    # Set default data_dir based on dataset
    if not args.data_dir:
        base = '/home/acolombo/VAEs/dataset'
        ds_map = {'mnist': 'MNIST', 'cifar100': 'CIFAR100', 'emnist': 'EMNIST'}
        args.data_dir = os.path.join(base, ds_map[args.dataset])

    return args


from academicodec.utils import Logger

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(args)
    logger.log_info(f"Device: {device}")
    logger.log_info("All arguments:")
    for k, v in vars(args).items():
        logger.log_info(f"  {k}: {v}")

    # Save config to checkpoint directory
    config_path = os.path.join(args.PATH, "config.py")
    with open(config_path, "w") as f:
        f.write("# Training hyperparameters\n")
        for k, v in vars(args).items():
            f.write(f"{k} = {v!r}\n")
    logger.log_info(f"Config saved to: {config_path}")

    # ── Data ──────────────────────────────────────────────────────────
    transform = transforms.ToTensor()  # scales to [0, 1]
    if args.dataset == "mnist":
        in_channels, img_size = 1, 28
        train_data = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == "emnist":
        in_channels, img_size = 1, 28
        train_data = datasets.EMNIST(root=args.data_dir, split=args.emnist_split, train=True, download=True, transform=transform)
        val_data = datasets.EMNIST(root=args.data_dir, split=args.emnist_split, train=False, download=True, transform=transform)
    elif args.dataset == "cifar100":
        in_channels, img_size = 3, 32
        train_data = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
        val_data = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    logger.log_info(f"Train loader size: {len(train_loader)}")
    val_loader = DataLoader(val_data, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    from mnist_vqvae import VQVAE2D, _count_params

    model = VQVAE2D(
        D=args.D, n_q=args.n_q, bins=args.bins, c=args.c, 
        exponential_lambda=args.exponential_lambda, ema=args.ema, 
        kmeans_init=args.kmeans_init, in_channels=in_channels, img_size=img_size,
    ).to(device)
    logger.log_info(model)
    total, trainable = _count_params(model)
    logger.log_info(f"Model params — total: {total:,}  trainable: {trainable:,}")

    if args.c > 0:
        manifold_params = []
        euclidean_params = []
        for p in model.parameters():
            if hasattr(p, "manifold"):
                manifold_params.append(p)
            else:
                euclidean_params.append(p)
        logger.log_info(f"Manifold params: {len(manifold_params)}")
        
        param_groups = []
        if len(manifold_params) > 0:
            # Conservative betas for manifold states to reduce exp_avg instability.
            param_groups.append({"params": manifold_params, "lr": args.lr_manifold, "betas": (0.0, 0.95), "eps": args.geoopt_eps})
        if len(euclidean_params) > 0:
            param_groups.append({"params": euclidean_params, "lr": args.lr_g, 'betas': (0.5, 0.9)})

        logger.log_info(
            f"Geoopt groups: manifold={len(manifold_params)}, "
            f"euclidean={len(euclidean_params)}, lr_manifold={args.lr_manifold:.2e}, lr_g={args.lr_g:.2e}, "
            f"eps={args.geoopt_eps:.1e}, stabilize={args.geoopt_stabilize}, "
        )
        optimizer = geoopt.optim.RiemannianAdam(
            param_groups,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
        logger.log_info(f"AdamW initialised {args.lr_g}")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    st_epoch = 1
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_path, "latest.pth"), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        st_epoch = ckpt["epoch"] + 1
        logger.log_info(f"Resumed from epoch {ckpt['epoch']}")

    # ── Codebook visualisation setup ────────────────────────────────
    from academicodec.visualization import fit_pca, plot_codes, create_gif
    pca_fitted = False
    pca_model = None
    codebook_plots_dir = os.path.join(args.PATH, "codebook_plots")
    global_step = 0

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_epoch = -1
    history = {"train_rec": [], "train_com": [], "val_rec": [], "val_com": [], "val_total": []}
    all_val_ppls = []    # list of per-epoch val PPL lists  (one list per epoch)
    all_train_ppls = []  # list of per-epoch train PPL lists (whole epoch)
    pbar = tqdm(range(st_epoch, args.N_EPOCHS + 1), desc="Training")
    for epoch in pbar:
        model.train()
        train_rec, train_com, n_batches = 0.0, 0.0, 0
        total_fw_time, total_bw_time, total_iter_time = 0.0, 0.0, 0.0

        # ── Train PPL histograms ────────────────────────────────────
        train_codes_hist = None
        train_codes_hist_last10 = None
        total_train_codes = 0
        total_train_codes_last10 = 0
        total_train_iters = len(train_loader)
        last_10_percent_start = int(total_train_iters * 0.9)

        t0 = time.time()
        for batch_idx, (imgs, _) in enumerate(train_loader, 1):
            total_iter_time += time.time() - t0
            imgs = imgs.to(device)

            t_fw_start = time.time()
            x_hat, commit_loss, codes = model(imgs)
            rec_loss = F.mse_loss(x_hat, imgs)
            loss = rec_loss + args.LAMBDA_COM * commit_loss
            t_fw_end = time.time()
            total_fw_time += (t_fw_end - t_fw_start)

            t_bw_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_bw_end = time.time()
            total_bw_time += (t_bw_end - t_bw_start)
            global_step += 1

            train_rec += rec_loss.item()
            train_com += commit_loss.item()
            n_batches += 1

            # ── Accumulate train code usage for PPL ──────────────
            with torch.no_grad():
                if train_codes_hist is None:
                    num_q = codes.shape[0]
                    codebook_size = args.bins
                    train_codes_hist = torch.zeros(num_q, codebook_size, device=device)
                    train_codes_hist_last10 = torch.zeros(num_q, codebook_size, device=device)
                codes_count = codes.shape[1] * codes.shape[2]  # B * N
                total_train_codes += codes_count
                is_last10 = batch_idx > last_10_percent_start
                if is_last10:
                    total_train_codes_last10 += codes_count
                for q_idx in range(codes.shape[0]):
                    codes_q = codes[q_idx].flatten()
                    ones = torch.ones_like(codes_q, dtype=torch.float32)
                    train_codes_hist[q_idx].scatter_add_(0, codes_q, ones)
                    if is_last10:
                        train_codes_hist_last10[q_idx].scatter_add_(0, codes_q, ones)

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
                message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}>'.format(
                    epoch, batch_idx, loss.item(), rec_loss.item(), commit_loss.item())
                logger.log_info(message)

            t0 = time.time()

        scheduler.step()
        train_rec /= n_batches
        train_com /= n_batches

        # ── Compute and log train PPL ────────────────────────────────
        with torch.no_grad():
            train_ppl_whole = []
            if train_codes_hist is not None and total_train_codes > 0:
                probs = train_codes_hist / train_codes_hist.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
                train_ppl_whole = torch.exp2(entropy).tolist()

            train_ppl_last10 = []
            if train_codes_hist_last10 is not None and total_train_codes_last10 > 0:
                probs = train_codes_hist_last10 / train_codes_hist_last10.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
                train_ppl_last10 = torch.exp2(entropy).tolist()

        all_train_ppls.append(train_ppl_whole)
        train_ppl_str = ", ".join([f"{p:.1f}" for p in train_ppl_whole]) if train_ppl_whole else "N/A"
        train_ppl_last10_str = ", ".join([f"{p:.1f}" for p in train_ppl_last10]) if train_ppl_last10 else "N/A"
        logger.log_info(f"Train PPL (whole epoch): [{train_ppl_str}]")
        logger.log_info(f"Train PPL (last 10%): [{train_ppl_last10_str}]")

        avg_fw_time = total_fw_time / n_batches
        avg_bw_time = total_bw_time / n_batches
        avg_iter_time = total_iter_time / n_batches
        logger.log_info(f"Epoch {epoch} timing: avg fw pass={avg_fw_time:.4f}s, avg bw pass={avg_bw_time:.4f}s, avg new batch iter={avg_iter_time:.4f}s")

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

        message_train = '<epoch:{:d}, total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, commit_loss_train:{:.4f}>'.format(
            epoch, train_rec + args.LAMBDA_COM * train_com, train_rec, train_com)
        logger.log_info(message_train)

        # Compute & store validation PPL per codebook
        val_ppls = []
        if codes_hist is not None:
            for q in range(codes_hist.shape[0]):
                sizes = codes_hist[q].long().tolist()
                sizes = [s / sum(sizes) for s in sizes]
                entropy = -sum(p * math.log2(p) for p in sizes if p > 0)
                perplexity = 2 ** entropy
                val_ppls.append(perplexity)
        all_val_ppls.append(val_ppls)
        ppl_str = ", ".join([f"{p:.1f}" for p in val_ppls]) if val_ppls else "N/A"

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
                best_val_epoch = epoch
                torch.save(save_dict, os.path.join(args.PATH, f"best_{epoch}.pth"))
                logger.log_info(f"  ✓ New best model saved (val_total={val_total:.5f})")
        else:
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_val_epoch = epoch

        message_val = '<epoch:{:d}, total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, ppl:[{}], best_epoch:{:d}>'.format(
            epoch, val_total, val_rec, val_com, ppl_str, best_val_epoch)
        logger.log_info(message_val)

    # ── End-of-training evaluation ──────────────────────────────────
    logger.log_info("\n" + "=" * 60)
    logger.log_info("Training complete — Final evaluation")
    logger.log_info("=" * 60)

    # Last model validation loss (already computed in the last epoch)
    logger.log_info(f"\n  Last model  (epoch {epoch}):  val_rec={val_rec:.5f}  "
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

        logger.log_info(f"  Best model  (epoch {best_epoch}):  val_rec={best_val_rec:.5f}  "
              f"val_com={best_val_com:.5f}  val_total={best_val_total:.5f}")
        logger.log_info(f"  Loaded from: {best_path}")
    else:
        logger.log_info("  No best checkpoint found (training may not have reached 75% of epochs).")

    # ── Plot examples: original vs reconstructed ─────────────────
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
    logger.log_info(f"\n  Reconstruction examples saved to: {fig_path}")

    # ── 1) Loss curves ───────────────────────────────────────────
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
    logger.log_info(f"  Loss curves saved to: {loss_fig_path}")

    # ── 2) PPLs per codebook (val) ──────────────────────────────
    # Each codebook gets its own subplot showing PPL across epochs
    if all_val_ppls and len(all_val_ppls[0]) > 0:
        num_quantizers = len(all_val_ppls[0])
        lists_per_q = list(zip(*all_val_ppls))

        cols = min(5, num_quantizers)
        rows = math.ceil(num_quantizers / cols)

        fig, axes_ppl = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if num_quantizers == 1:
            axes_ppl = [axes_ppl]
        else:
            axes_ppl = axes_ppl.flatten() if hasattr(axes_ppl, 'flatten') else [axes_ppl]

        for i in range(num_quantizers):
            axes_ppl[i].plot(epochs_range, lists_per_q[i], color='tab:blue', linewidth=2)
            axes_ppl[i].set_title(f"Quantizer {i+1}", fontsize=10, fontweight='bold')
            axes_ppl[i].set_xlabel("Epoch", fontsize=8)
            axes_ppl[i].set_ylabel("PPL", fontsize=8)
            axes_ppl[i].grid(True, linestyle='--', alpha=0.7)

        for i in range(num_quantizers, len(axes_ppl)):
            fig.delaxes(axes_ppl[i])

        plt.tight_layout()
        ppl_cb_path = os.path.join(args.PATH, "ppls_per_codebook.png")
        plt.savefig(ppl_cb_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.log_info(f"  PPLs per codebook saved to: {ppl_cb_path}")

    # ── 3) PPLs per epoch (val) — codebook index vs PPL across epochs ─
    if all_val_ppls and len(all_val_ppls[0]) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        codebook_indices = list(range(1, num_quantizers + 1))

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / max(1, len(all_val_ppls) - 1)) for i in range(len(all_val_ppls))]

        for epoch_idx, ppls in enumerate(all_val_ppls):
            ax2.plot(codebook_indices, ppls, marker='.', markersize=4, alpha=0.6,
                     color=colors[epoch_idx], linewidth=1.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(all_val_ppls)-1))
        sm.set_array([])
        cbar = fig2.colorbar(sm, ax=ax2)
        cbar.set_label("Epoch", fontsize=10)

        ax2.set_xticks(codebook_indices)
        ax2.set_xlabel("Codebook Index", fontsize=10)
        ax2.set_ylabel("PPL", fontsize=10)
        ax2.set_title("Codebook vs Val PPL across Epochs", fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)

        ppl_epoch_path = os.path.join(args.PATH, "ppls_per_epoch.png")
        plt.savefig(ppl_epoch_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.log_info(f"  PPLs per epoch (val) saved to: {ppl_epoch_path}")

    # ── 4) Train PPLs per epoch — codebook index vs train PPL ───
    if all_train_ppls and len(all_train_ppls[0]) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        codebook_indices = list(range(1, len(all_train_ppls[0]) + 1))

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / max(1, len(all_train_ppls) - 1)) for i in range(len(all_train_ppls))]

        for epoch_idx, ppls in enumerate(all_train_ppls):
            ax3.plot(codebook_indices, ppls, marker='.', markersize=4, alpha=0.6,
                     color=colors[epoch_idx], linewidth=1.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(all_train_ppls)-1))
        sm.set_array([])
        cbar = fig3.colorbar(sm, ax=ax3)
        cbar.set_label("Epoch", fontsize=10)

        ax3.set_xticks(codebook_indices)
        ax3.set_xlabel("Codebook Index", fontsize=10)
        ax3.set_ylabel("PPL", fontsize=10)
        ax3.set_title("Codebook vs Train PPL across Epochs", fontsize=12, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.7)

        train_ppl_path = os.path.join(args.PATH, "train_ppls_per_epoch.png")
        plt.savefig(train_ppl_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.log_info(f"  Train PPLs per epoch saved to: {train_ppl_path}")

    # ── Final codebook snapshot + GIF ────────────────────────────
    if pca_fitted:
        codebook_module = model.quantizer.vq.layers[args.codebook_number]._codebook
        codes_cb = codebook_module.embed.detach().cpu()
        plot_codes(codes_cb, pca_model, args.c, global_step, codebook_plots_dir)

    gif_path = os.path.join(args.PATH, "codebook_evolution.gif")
    create_gif(codebook_plots_dir, gif_path)
    logger.log_info(f"  Codebook evolution GIF saved to: {gif_path}")


if __name__ == "__main__":
    main()
