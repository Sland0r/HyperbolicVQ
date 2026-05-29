import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import geoopt
from torch.utils.data import DataLoader
import argparse
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent dir to path to import academicodec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from academicodec.quantization.vq import ResidualVectorQuantizer
from NLP.wordnet_dataset import WordNetHierarchyDataset

class HRQModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_q=4, bins=256, c=1.0, new_method=True, hste=False, approx=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.quantizer = ResidualVectorQuantizer(
            dimension=embed_dim,
            n_q=n_q,
            bins=bins,
            c=c,
            new_method=new_method,
            hste=hste,
            approx=approx,
        )
        self.c = c

    def forward(self, x):
        # x is shape [batch_size]
        emb = self.embedding(x)
        # RVQ expects [batch, channels, seq_len]
        emb_unsqueezed = emb.unsqueeze(-1)
        quantized, codes, _, penalty, distance = self.quantizer(emb_unsqueezed, sample_rate=0)
        return emb, quantized.squeeze(-1), codes, penalty, distance
         

def exp_map0(v, c):
    if c == 0.0:
        return v
    norm = v.norm(dim=-1, keepdim=True)
    sqrt_c = c ** 0.5
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm.clamp_min(1e-5))
    return v * scale

def poincare_distance(x, y, c):
    """Compute poincaré distance (NOT squared) between x and y.

    When c == 0 returns Euclidean distance.  When c > 0, maps tangent-space
    vectors onto the Poincaré ball first and returns the geodesic distance.
    """
    if c > 0.0:
        x = exp_map0(x, c)
        y = exp_map0(y, c)

    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)

    sq_dist = (x2 + y2 - 2 * xy).clamp_min(0.0)

    if c == 0.0:
        return (sq_dist + 1e-8).sqrt()  # eps prevents ∇sqrt(0) = ∞

    denom = ((1 - c * x2) * (1 - c * y2)).clamp_min(1e-6)
    arg = 1 + 2 * c * sq_dist / denom
    dist = (1 / (c**0.5)) * torch.acosh(arg.clamp_min(1.0 + 1e-5))
    return dist

def train(args):
    dataset = WordNetHierarchyDataset(num_negatives=50, split='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = HRQModel(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        n_q=args.n_q,
        bins=args.bins,
        c=args.c,
        new_method=args.new_method,
        hste=args.hste,
        approx=args.approx,
    ).to(args.device)
    
    # ── Constructive codebook initialisation ──────────────────────
    if args.constructive:
        import sys
        sys.path.insert(0, '/home/acolombo/music/hyperbolic_tree_embeddings')
        from tree_embeddings.trees.file_utils import load_hierarchy
        from tree_embeddings.embeddings.constructive_method import constructively_embed_tree

        # Balanced tree: bins children at depth 1 → bins+1 nodes total
        hierarchy = load_hierarchy(dataset="n_h_trees", hierarchy_name=f"{args.bins}_1")

        curvature = args.c if args.c > 0 else 1.0
        embeddings, _, _ = constructively_embed_tree(
            hierarchy=hierarchy,
            dataset="n_h_trees",
            hierarchy_name=f"{args.bins}_1",
            embedding_dim=args.embed_dim,
            tau=1.0,
            nc=1,
            curvature=curvature,
            root=0,
            gen_type="optim",
            dtype=torch.float64,
        )

        # Skip root (index 0, at origin); keep the bins children
        code_points = embeddings[1:].to(dtype=torch.float32, device=args.device) / args.n_q  # (bins, D)
        assert code_points.shape == (args.bins, args.embed_dim), \
            f"Expected ({args.bins}, {args.embed_dim}), got {code_points.shape}"

        # Copy same points into every codebook
        with torch.no_grad():
            for qi in range(args.n_q):
                cb = model.quantizer.vq.layers[qi]._codebook
                cb.embed.data.copy_(code_points)
                cb.embed_avg.data.copy_(code_points)
                cb.inited.data.copy_(torch.Tensor([True]))
                cb.cluster_size.data.fill_(3)  # default threshold_ema_dead_code is 2

    if args.c > 0:
        manifold_params = []
        euclidean_params = []
        for p in model.parameters():
            if hasattr(p, "manifold"):
                manifold_params.append(p)
            else:
                euclidean_params.append(p)
        print(f"Manifold params: {len(manifold_params)}, Euclidean params: {len(euclidean_params)}")

        # Paper (Appendix C.1): Riemannian SGD for HRQ
        param_groups = []
        if manifold_params:
            param_groups.append({"params": manifold_params, "lr": args.warmup_lr})
        if euclidean_params:
            param_groups.append({"params": euclidean_params, "lr": args.warmup_lr})
        optimizer = geoopt.optim.RiemannianSGD(param_groups, lr=args.warmup_lr)
    else:
        # Paper (Appendix C.1): SGD for RQ
        optimizer = optim.SGD(model.parameters(), lr=args.warmup_lr)
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'logs.txt')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Args: {vars(args)}")
    log(f"Dataset size: {len(dataset)}, Batches/epoch: {len(dataloader)}")
    log("")

    history = {'epoch': [], 'loss': [], 'ce_loss': [], 'commit_loss': [], 'approx_dist': []}
    for qi in range(args.n_q):
        history[f'ppl_q{qi}'] = []
    train_start = time.time()

    for epoch in range(args.epochs):
        # Paper (Appendix C.1): warmup lr=0.01 for first 20 epochs, then lr=1.0
        if epoch == args.warmup_epochs:
            new_lr = args.lr
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            log(f"Warmup done — switching lr to {new_lr}")

        model.train()
        total_loss = 0
        total_ce = 0
        total_commit = 0
        total_dist = 0
        code_counts = [torch.zeros(args.bins, device=args.device) for _ in range(args.n_q)]
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            u = batch['u'].to(args.device)
            v = batch['v'].to(args.device)
            negatives = batch['negatives'].to(args.device)

            u_emb, u_quant, u_codes, u_commit, u_dist = model(u)
            v_emb, v_quant, v_codes, v_commit, v_dist = model(v)

            # Accumulate code usage for perplexity: codes shape is (n_q, B, 1)
            for qi in range(args.n_q):
                counts = torch.bincount(u_codes[qi].reshape(-1), minlength=args.bins).float()
                counts += torch.bincount(v_codes[qi].reshape(-1), minlength=args.bins).float()
                code_counts[qi] += counts

            B, num_neg = negatives.shape
            neg_flat = negatives.view(-1)
            neg_emb, neg_quant, _, _, _ = model(neg_flat)
            neg_emb = neg_emb.view(B, num_neg, -1)

            # Paper: Contrastive loss is calculated on continuous embeddings E_theta
            pos_dist = poincare_distance(u_emb, v_emb, args.c).squeeze(-1)

            u_expanded = u_emb.unsqueeze(1).expand(-1, num_neg, -1).reshape(B*num_neg, -1)
            neg_dist = poincare_distance(u_expanded, neg_emb.reshape(B*num_neg, -1), args.c).view(B, num_neg)

            # Add self-distance d(u, u) = 0 to negatives to match the paper H'(u) definition
            self_dist = torch.zeros((B, 1), device=args.device)

            # Paper: log-softmax contrastive loss (no temperature scaling)
            logits = torch.cat([-pos_dist.unsqueeze(1), -neg_dist, -self_dist], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=args.device)

            ce_loss = nn.CrossEntropyLoss()(logits, labels)
            commit = u_commit.mean() + v_commit.mean()
            loss = ce_loss + commit
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_commit += commit.item()
            total_dist += (u_dist.item() + v_dist.item()) / 2
            if i % 100 == 0:
                log(f"  Step {i}, Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Commit: {commit.item():.4f})")

        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_ce = total_ce / n_batches
        avg_commit = total_commit / n_batches
        avg_dist = total_dist / n_batches

        # Compute per-layer perplexity: exp(entropy) of code usage distribution
        ppl_strs = []
        for qi in range(args.n_q):
            probs = code_counts[qi] / code_counts[qi].sum()
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            ppl = entropy.exp().item()
            history[f'ppl_q{qi}'].append(ppl)
            ppl_strs.append(f"Q{qi}={ppl:.1f}")

        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['ce_loss'].append(avg_ce)
        history['commit_loss'].append(avg_commit)
        history['approx_dist'].append(avg_dist)

        elapsed = time.time() - train_start
        log(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Commit: {avg_commit:.4f}, ApproxDist: {avg_dist:.6f}), PPL: [{', '.join(ppl_strs)}], Time: {elapsed:.0f}s")

    log(f"\nTraining finished in {time.time() - train_start:.0f}s")

    # Validation pass (no_grad over entire dataset)
    model.eval()
    val_dist_sum = 0
    val_n = 0
    with torch.no_grad():
        for batch in dataloader:
            u = batch['u'].to(args.device)
            v = batch['v'].to(args.device)
            _, _, _, _, u_dist = model(u)
            _, _, _, _, v_dist = model(v)
            val_dist_sum += (u_dist.item() + v_dist.item()) / 2
            val_n += 1
    val_approx_dist = val_dist_sum / val_n
    log(f"\nValidation approx_distance: {val_approx_dist:.6f}")

    log("")
    log("Perplexity summary (across epochs):")
    for qi in range(args.n_q):
        ppls = np.array(history[f'ppl_q{qi}'])
        log(f"  Q{qi}: mean={ppls.mean():.2f}, var={ppls.var():.2f}, final={ppls[-1]:.2f}")
    all_final = [history[f'ppl_q{qi}'][-1] for qi in range(args.n_q)]
    log(f"  All layers final: mean={np.mean(all_final):.2f}, var={np.var(all_final):.2f}")
    log_file.close()

    # Save loss curves + perplexity + approx distance
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax, key, label in zip(axes[:3], ['loss', 'ce_loss', 'commit_loss'], ['Total Loss', 'CE Loss', 'Commit Loss']):
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
    print(f"Loss curves saved to {os.path.join(save_dir, 'loss_curves.png')}")

    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    config_path = os.path.join(save_dir, 'config.py')
    with open(config_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {repr(v)}\n")
    print(f"Config saved to {config_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--n_q', type=int, default=4)
    parser.add_argument('--bins', type=int, default=256)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--warmup_lr', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoint/nlp/default')
    parser.add_argument('--constructive', action='store_true', help='initialize codebooks using constructive tree embeddings (depth=1)')
    parser.add_argument('--new_method', action='store_true',
                        help='Use left-subtraction encoding with right-associative decoding (default: True)')
    parser.add_argument('--approx', action='store_true',
                        help='Track hyperbolic approximation distance (quantized+residual vs input)')
    parser.add_argument('--hste', action='store_true',
                        help='Use hyperbolic straight-through estimator instead of Euclidean STE')

    args = parser.parse_args()
    train(args)
