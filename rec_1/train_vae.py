import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geoopt
from academicodec.quantization.vq import ResidualVectorQuantizer
from rec_1.amazon_dataset import prepare_data, DATASET_DIR

class HRQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, n_q=4, bins=256, c=1.0,
                 kmeans_init=False, ema=False, new_method=False, hste=False, approx=False,
                 hste_riemannian=False, gradient_correction=False, block_hste_pt=False,
                 hste_clip=False, a6_1=False, a7_1=False,
                 a6=False, a7=False, a8=False, A4_v2=False, A5=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, embed_dim),
        )
        self.quantizer = ResidualVectorQuantizer(
            dimension=embed_dim,
            n_q=n_q,
            bins=bins,
            c=c,
            kmeans_init=kmeans_init,
            ema=ema,
            threshold_ema_dead_code=2,
            new_method=new_method,
            hste=hste,
            approx=approx,
            hste_riemannian=hste_riemannian,
            gradient_correction=gradient_correction,
            block_hste_pt=block_hste_pt,
            hste_clip=hste_clip,
            a6_1=a6_1,
            a7_1=a7_1,
            a6=a6,
            a7=a7,
            a8=a8,
            A4_v2=A4_v2,
            A5=A5,
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.c = c

    def forward(self, x):
        emb = self.encoder(x)
        emb_unsqueezed = emb.unsqueeze(-1)
        quantized, codes, _, penalty, distance = self.quantizer(emb_unsqueezed, sample_rate=0)
        quantized = quantized.squeeze(-1)
        recon = self.decoder(quantized)
        return recon, quantized, codes, penalty, distance

def train(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'logs.txt')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Starting VAE training with c={args.c}")
    log(f"Args: {vars(args)}")

    _, item_catalog, _, _, embeddings_file = prepare_data(args.dataset)
    embeddings = torch.load(embeddings_file)

    log(f"Loaded {embeddings.shape[0]} item embeddings of dimension {embeddings.shape[1]}")

    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = HRQVAE(args.input_dim, args.hidden_dim, args.embed_dim, args.n_q, args.bins, args.c,
                   kmeans_init=args.kmeans_init, ema=args.ema,
                   new_method=args.new_method, hste=args.hste, approx=args.approx,
                   hste_riemannian=args.hste_riemannian,
                   gradient_correction=args.gradient_correction,
                   block_hste_pt=args.block_hste_pt,
                   hste_clip=args.hste_clip,
                   a6_1=args.a6_1, a7_1=args.a7_1,
                   a6=args.a6, a7=args.a7, a8=args.a8,
                   A4_v2=args.A4_v2, A5=args.A5).to(args.device)

    if args.c > 0:
        manifold_params = []
        euclidean_params = []
        for p in model.parameters():
            if hasattr(p, "manifold"):
                manifold_params.append(p)
            else:
                euclidean_params.append(p)

        param_groups = []
        if manifold_params:
            param_groups.append({"params": manifold_params, "lr": args.lr_manifold, "betas": (0.0, 0.95)})
        if euclidean_params:
            param_groups.append({"params": euclidean_params, "lr": args.lr, "betas": (0.5, 0.9)})

        optimizer = geoopt.optim.RiemannianAdam(param_groups)
        log(f"Using RiemannianAdam (manifold params: {len(manifold_params)}, euclidean: {len(euclidean_params)})")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    log(f"Dataset size: {len(dataset)}, Batches/epoch: {len(dataloader)}")
    log("")

    history = {'epoch': [], 'loss': [], 'recon_loss': [], 'commit_loss': [], 'approx_dist': []}
    for qi in range(args.n_q):
        history[f'ppl_q{qi}'] = []
    train_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_commit = 0
        total_dist = 0
        code_counts = [torch.zeros(args.bins, device=args.device) for _ in range(args.n_q)]

        for batch in dataloader:
            x = batch[0].to(args.device)
            optimizer.zero_grad()

            recon, quantized, codes, penalty, distance = model(x)
            recon_loss = nn.functional.mse_loss(recon, x)
            recon_term = recon_loss * args.recon
            commit = penalty.mean() * args.quant
            loss = recon_term + commit

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_term.item()
            total_commit += commit.item()
            total_dist += distance.item()

            # Accumulate code usage for perplexity: codes shape is (n_q, B, 1)
            with torch.no_grad():
                for qi in range(args.n_q):
                    counts = torch.bincount(codes[qi].reshape(-1), minlength=args.bins).float()
                    code_counts[qi] += counts

        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_commit = total_commit / n_batches
        avg_dist = total_dist / n_batches

        # Compute per-layer perplexity
        ppl_strs = []
        for qi in range(args.n_q):
            probs = code_counts[qi] / code_counts[qi].sum()
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            ppl = entropy.exp().item()
            history[f'ppl_q{qi}'].append(ppl)
            ppl_strs.append(f"Q{qi}={ppl:.1f}")

        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['commit_loss'].append(avg_commit)
        history['approx_dist'].append(avg_dist)

        elapsed = time.time() - train_start
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            log(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f}, ApproxDist: {avg_dist:.6f}), PPL: [{', '.join(ppl_strs)}], Time: {elapsed:.0f}s")

    log(f"\nTraining finished in {time.time() - train_start:.0f}s")

    # Validation pass (no_grad over entire dataset)
    model.eval()
    val_dist_sum = 0
    val_recon_sum = 0
    val_n = 0
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for batch in eval_loader:
            x = batch[0].to(args.device)
            recon, _, _, _, distance = model(x)
            val_recon_sum += nn.functional.mse_loss(recon, x).item()
            val_dist_sum += distance.item()
            val_n += 1
    val_approx_dist = val_dist_sum / val_n
    val_recon = val_recon_sum / val_n
    log(f"\nValidation: recon={val_recon:.6f}, approx_distance={val_approx_dist:.6f}")

    # Perplexity summary
    log("")
    log("Perplexity summary (across epochs):")
    for qi in range(args.n_q):
        ppls = np.array(history[f'ppl_q{qi}'])
        log(f"  Q{qi}: mean={ppls.mean():.2f}, var={ppls.var():.2f}, final={ppls[-1]:.2f}")
    all_final = [history[f'ppl_q{qi}'][-1] for qi in range(args.n_q)]
    log(f"  All layers final: mean={np.mean(all_final):.2f}, var={np.var(all_final):.2f}")

    # Save loss curves + perplexity + approx distance
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax, key, label in zip(axes[:3], ['loss', 'recon_loss', 'commit_loss'], ['Total Loss', 'Recon Loss', 'Commit Loss']):
        ax.plot(history['epoch'], history[key])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    ax_ppl = axes[3]
    for qi in range(args.n_q):
        ax_ppl.plot(history['epoch'], history[f'ppl_q{qi}'], label=f'Q{qi}')
    ax_ppl.axhline(y=args.bins, color='k', linestyle='--', alpha=0.3, label=f'max ({args.bins})')
    ax_ppl.set_xlabel('Epoch')
    ax_ppl.set_ylabel('Perplexity')
    ax_ppl.set_title('Codebook Perplexity')
    ax_ppl.legend(fontsize=8)
    ax_ppl.grid(True, alpha=0.3)
    ax_dist = axes[4]
    ax_dist.plot(history['epoch'], history['approx_dist'], label='Train')
    ax_dist.axhline(y=val_approx_dist, color='r', linestyle='--', alpha=0.7, label=f'Val ({val_approx_dist:.4f})')
    ax_dist.set_xlabel('Epoch')
    ax_dist.set_ylabel('Approx Distance')
    ax_dist.set_title('Hyperbolic Approx Distance')
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)
    plt.close(fig)
    log(f"Loss curves saved to {os.path.join(save_dir, 'loss_curves.png')}")

    # Save model and config
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    log(f"Model saved to {model_path}")

    config_path = os.path.join(save_dir, 'config.py')
    with open(config_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {repr(v)}\n")
    log(f"Config saved to {config_path}")

    # Extract codes
    log("\nExtracting discrete codes...")
    all_codes = []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for batch in eval_loader:
            x = batch[0].to(args.device)
            _, _, codes, _, _ = model(x)
            codes = codes.squeeze(-1) if len(codes.shape) > 2 else codes
            if codes.size(1) == x.size(0) and codes.size(0) != x.size(0):
                codes = codes.transpose(0, 1)
            all_codes.append(codes.cpu())

    all_codes = torch.cat(all_codes, dim=0) # shape: (num_items, n_q)

    # Code diversity check
    num_items = all_codes.shape[0]
    unique_codes = torch.unique(all_codes, dim=0).shape[0]
    log(f"\n--- Code Diversity ---")
    log(f"Total items: {num_items}")
    log(f"Unique multitoken sequences: {unique_codes}")
    log(f"Uniqueness ratio: {unique_codes / num_items:.4f}")
    for q in range(all_codes.shape[1]):
        codes_q = all_codes[:, q]
        unique_at_level = codes_q.unique().shape[0]
        counts = torch.bincount(codes_q.int(), minlength=args.bins).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        perplexity = entropy.exp().item()
        log(f"  Level {q}: {unique_at_level}/{args.bins} codebook entries used, perplexity: {perplexity:.2f}")

    codes_file = os.path.join(DATASET_DIR, f"{args.dataset.lower()}_item_codes_c{args.c}{args.codes_tag}.pt")
    torch.save(all_codes, codes_file)
    log(f"Saved {all_codes.shape[0]} discrete codes to {codes_file}")

    if args.dedup:
        # TIGER-style uniqueness token (paper §2.2): append a per-item counter
        # that disambiguates items colliding on the same n_q-token multitoken,
        # so every full (n_q+1)-token id is unique. The extra token carries no
        # semantic information — it only breaks ties.
        seen = {}
        dedup_col = torch.empty(num_items, 1, dtype=all_codes.dtype)
        for i in range(num_items):
            key = tuple(all_codes[i].tolist())
            cnt = seen.get(key, 0)
            dedup_col[i, 0] = cnt
            seen[key] = cnt + 1
        max_group = max(seen.values())
        log(f"\n--- Dedup (uniqueness) token ---")
        log(f"Max items sharing a multitoken: {max_group} "
            f"(dedup token range 0..{max_group - 1})")
        if max_group > args.bins:
            log(f"  WARNING: max collision group {max_group} > bins {args.bins}; "
                f"the dedup token will exceed the per-level vocab used by "
                f"train_recommender.py. Increase --bins.")
        all_codes_dedup = torch.cat([all_codes, dedup_col], dim=1)
        n_unique_full = torch.unique(all_codes_dedup, dim=0).shape[0]
        log(f"Unique full sequences after dedup: {n_unique_full}/{num_items} "
            f"(ratio {n_unique_full / num_items:.4f})")
        dedup_file = os.path.join(DATASET_DIR, f"{args.dataset.lower()}_item_codes_c{args.c}_dedup{args.codes_tag}.pt")
        torch.save(all_codes_dedup, dedup_file)
        log(f"Saved dedup codes {tuple(all_codes_dedup.shape)} to {dedup_file}")

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--codes_tag", type=str, default="",
                        help='Optional suffix appended to the saved codes filename '
                             'so concurrent runs (same c/dataset) do not clobber '
                             'each other; train_recommender must use the same tag.')
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--n_q", type=int, default=4)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_manifold", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--kmeans_init", action='store_true')
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--new_method", action='store_true',
                        help='Use left-subtraction encoding with right-associative decoding')
    parser.add_argument("--approx", action='store_true',
                        help='Track hyperbolic approximation distance (quantized+residual vs input)')
    parser.add_argument("--hste", action='store_true',
                        help='Use hyperbolic straight-through estimator instead of Euclidean STE')
    parser.add_argument("--hste_riemannian", action='store_true',
                        help='Use Riemannian (conformal-factor scaled) gradient in the hyperbolic STE')
    parser.add_argument("--gradient_correction", action='store_true',
                        help='Detach the residual after each quantization step (gradient correction)')
    parser.add_argument("--block_hste_pt", action='store_true',
                        help='Block-level PT-STE')
    parser.add_argument("--hste_clip", action='store_true',
                        help='Cap the HSTE outgoing gradient norm at the incoming norm (net conformal amplification <= 1)')
    parser.add_argument("--a6.1", dest="a6_1", action='store_true',
                        help='SUM routing: encoder recon grad = A4 block-hop grad + a6 '
                             '(keep-first) per-layer grad. Requires --new_method --hste '
                             '--block_hste_pt --hste_riemannian.')
    parser.add_argument("--a7.1", dest="a7_1", action='store_true',
                        help='SUM routing: encoder recon grad = A4 block-hop grad + a7 '
                             '(keep-last) per-layer grad. Requires --new_method --hste '
                             '--block_hste_pt --hste_riemannian.')
    parser.add_argument("--a6", action='store_true',
                        help='keep-first per-layer recon routing on the A3 per-layer-HSTE '
                             'base (no block hop). Requires --new_method --hste.')
    parser.add_argument("--a7", action='store_true',
                        help='keep-last per-layer recon routing on the A3 per-layer-HSTE '
                             'base (no block hop). Requires --new_method --hste.')
    parser.add_argument("--a8", action='store_true',
                        help='full-sum routing: A4 block hop + ALL per-layer recon '
                             '(un-truncated sibling of a6.1/a7.1). Requires --new_method.')
    parser.add_argument("--A4_v2", dest="A4_v2", action='store_true',
                        help='strict gradient correction: removes the layer-0 commit '
                             'residual leak that plain gc still passes through.')
    parser.add_argument("--A5", dest="A5", action='store_true',
                        help='A4 + last-commit chain (extra commitment term from the '
                             'final residual stage).')
    parser.add_argument("--quant", type=float, default=1.0,
                        help='Multiplier for the quantizer penalty term')
    parser.add_argument("--recon", type=float, default=1.0,
                        help='Multiplier for the reconstruction loss term')
    parser.add_argument("--dedup", action='store_true',
                        help='Also save a uniqueness-token variant of the codes '
                             '(paper §2.2): appends a per-item tie-break token so '
                             'colliding multitokens become unique. Written to '
                             'item_codes_c<c>_dedup.pt for train_recommender.py --dedup.')
    parser.add_argument("--save_dir", type=str, default="checkpoint/rec_1/default")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
