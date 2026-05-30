"""Paper-faithful Hierarchy Modeling training (§4.1 / Appendix C.1).

Standalone alternative to ``NLP/train_hierarchy.py``. Identical model, loss and
optimiser schedule, but it pulls pairs from the paper-faithful closure-split
dataset (``NLP_2/dataset_paper.py``) by default.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from torch.utils.data import DataLoader
import argparse
import sys
import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent dir to path to import academicodec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from academicodec.quantization.vq import ResidualVectorQuantizer
from NLP_2.dataset_paper import WordNetHierarchyDataset


class HRQModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_q=4, bins=256, c=1.0,
                 new_method=True, hste=False, approx=False,
                 commitment_weight=0.25, ema=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.quantizer = ResidualVectorQuantizer(
            dimension=embed_dim, n_q=n_q, bins=bins, c=c,
            new_method=new_method, hste=hste, approx=approx,
            commitment_weight=commitment_weight, ema=ema,
        )
        self.c = c

    def forward(self, x):
        emb = self.embedding(x)
        emb_unsqueezed = emb.unsqueeze(-1)  # RVQ expects [batch, channels, seq_len]
        quantized, codes, _, penalty, distance = self.quantizer(emb_unsqueezed, sample_rate=0)
        return emb, quantized.squeeze(-1), codes, penalty, distance

    def emb_only_forward(self, x):
        return self.embedding(x)

def exp_map0(v, c):
    if c == 0.0:
        return v
    norm = v.norm(dim=-1, keepdim=True)
    sqrt_c = c ** 0.5
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm.clamp_min(1e-5))
    return v * scale


def poincare_distance(x, y, c):
    """Poincaré distance (NOT squared). c==0 -> Euclidean distance."""
    if c > 0.0:
        x = exp_map0(x, c)
        y = exp_map0(y, c)

    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    sq_dist = (x2 + y2 - 2 * xy).clamp_min(0.0)

    if c == 0.0:
        return (sq_dist + 1e-8).sqrt()

    denom = ((1 - c * x2) * (1 - c * y2)).clamp_min(1e-6)
    arg = 1 + 2 * c * sq_dist / denom
    return (1 / (c ** 0.5)) * torch.acosh(arg.clamp_min(1.0 + 1e-5))


class Diagnostics:
    """Training-health checks, all gated behind ``--test``.

    Two layers:
      * per-step sanity checks   -> collect (de-duplicated) warnings
      * per-epoch scalar history -> drives the diagnostics.png figure

    The goal is to answer "why is performance bad?": are gradients reaching
    the codebooks, are embeddings leaving the Poincaré ball, are codes dead,
    is the contrastive task actually being learned, is anything NaN/exploding?
    """

    EXPLODE = 1e3        # grad norm above this -> "exploding"
    VANISH = 1e-8        # grad norm below this -> "vanishing"

    def __init__(self, base_model, args):
        self.enabled = args.test
        self.args = args
        self.base_model = base_model          # un-compiled model (shared params)
        self.warnings = []
        self._seen = set()
        if not self.enabled:
            return

        # bucket parameters once: token embedding vs. codebooks, + manifold flag
        self.emb_params, self.cb_params = [], []
        n_manifold = 0
        for name, p in base_model.named_parameters():
            if name.startswith('embedding'):
                self.emb_params.append((name, p))
            else:
                self.cb_params.append((name, p))
            if getattr(p, 'manifold', None) is not None:
                n_manifold += 1
        print(f"[TEST] params: {len(self.emb_params)} embedding tensor(s), "
              f"{len(self.cb_params)} codebook tensor(s), "
              f"{n_manifold} on a manifold")
        if args.c > 0 and n_manifold == 0:
            self.warn('no_manifold',
                      "c>0 but NO parameter carries a .manifold -> "
                      "RiemannianSGD will treat codebooks as Euclidean")

        # snapshots for drift measurement
        self._emb0 = {n: p.detach().clone() for n, p in self.emb_params}
        self._cb0 = {n: p.detach().clone() for n, p in self.cb_params}

        self.ball_r = (1.0 / math.sqrt(args.c)) if args.c > 0 else float('inf')

        # per-epoch history
        self.h = {k: [] for k in
                  ('grad_emb', 'grad_cb', 'acc', 'pos_d', 'neg_d', 'margin',
                   'emb_maxnorm', 'emb_drift', 'cb_drift')}
        self.h['dead'] = [[] for _ in range(args.n_q)]
        self.start_epoch()

    # ---- warning helper --------------------------------------------------
    def warn(self, key, msg):
        if key in self._seen:
            return
        self._seen.add(key)
        self.warnings.append(msg)
        print(f"  [TEST][WARN] {msg}")

    # ---- per-epoch accumulators -----------------------------------------
    def start_epoch(self):
        if not self.enabled:
            return
        self._g_emb = self._g_cb = 0.0
        self._acc = self._posd = self._negd = 0.0
        self._emb_maxnorm = 0.0
        self._nsteps = 0

    # ---- per-step forward checks ----------------------------------------
    def check_forward(self, step, tensors, u_emb, logits, labels,
                      pos_dist, neg_dist):
        if not self.enabled:
            return
        # finite checks involve a host sync; do them at the logger's cadence
        if step % 100 == 0:
            for name, t in tensors.items():
                if not torch.isfinite(t).all():
                    self.warn(f'nan_{name}',
                              f"non-finite values in '{name}' (NaN/Inf) "
                              f"at step {step}")
        with torch.no_grad():
            # contrastive task accuracy: the positive should be the argmax
            self._acc += (logits.argmax(dim=1) == labels).float().mean().item()
            self._posd += pos_dist.mean().item()
            self._negd += neg_dist.mean().item()
            # Poincaré-ball containment of the *mapped* embeddings
            mapped = exp_map0(u_emb, self.args.c) if self.args.c > 0 else u_emb
            mn = mapped.norm(dim=-1).max().item()
            self._emb_maxnorm = max(self._emb_maxnorm, mn)
            if self.args.c > 0 and mn >= self.ball_r:
                self.warn('offball',
                          f"embedding left the ball: |x|={mn:.4f} >= "
                          f"radius {self.ball_r:.4f} (step {step})")
        self._nsteps += 1

    # ---- per-step grad checks (call after backward, before optim.step) ---
    def check_grads(self, step):
        if not self.enabled:
            return

        def group_norm(params, tag):
            sq, n_grad = 0.0, 0
            for name, p in params:
                if not p.requires_grad or p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    self.warn(f'gnan_{name}',
                              f"NaN/Inf gradient in '{name}' (step {step})")
                sq += g.detach().float().pow(2).sum().item()
                n_grad += 1
            if n_grad == 0 and step == 0:
                self.warn(f'nograd_{tag}',
                          f"{tag} received NO gradient (all grads are None) "
                          f"-> these parameters are frozen")
            return math.sqrt(sq)

        g_emb = group_norm(self.emb_params, 'embedding')
        g_cb = group_norm(self.cb_params, 'codebook')
        if step == 0:
            if g_cb == 0.0:
                self.warn('cb_zero',
                          "codebook gradient is exactly zero on the first step "
                          "-> codebooks are not being trained by the loss")
            for tag, g in (('embedding', g_emb), ('codebook', g_cb)):
                if g > self.EXPLODE:
                    self.warn(f'explode_{tag}',
                              f"{tag} grad norm {g:.2e} > {self.EXPLODE:.0e} "
                              f"-> exploding gradients (lr={self.args.lr}?)")
                elif 0.0 < g < self.VANISH:
                    self.warn(f'vanish_{tag}',
                              f"{tag} grad norm {g:.2e} < {self.VANISH:.0e} "
                              f"-> vanishing gradients")
        self._g_emb += g_emb
        self._g_cb += g_cb

    # ---- end of epoch ----------------------------------------------------
    def log_epoch(self, code_counts):
        if not self.enabled:
            return
        n = max(self._nsteps, 1)
        self.h['grad_emb'].append(self._g_emb / n)
        self.h['grad_cb'].append(self._g_cb / n)
        self.h['acc'].append(self._acc / n)
        self.h['pos_d'].append(self._posd / n)
        self.h['neg_d'].append(self._negd / n)
        self.h['margin'].append((self._negd - self._posd) / n)
        self.h['emb_maxnorm'].append(self._emb_maxnorm)
        for qi in range(self.args.n_q):
            self.h['dead'][qi].append(int((code_counts[qi] == 0).sum().item()))
        emb_d = sum((p.detach() - self._emb0[n]).norm().item()
                    for n, p in self.emb_params)
        cb_d = sum((p.detach() - self._cb0[n]).norm().item()
                   for n, p in self.cb_params)
        self.h['emb_drift'].append(emb_d)
        self.h['cb_drift'].append(cb_d)

    # ---- final text report ----------------------------------------------
    def report(self, log):
        if not self.enabled:
            return
        log("")
        log("=" * 60)
        log("DIAGNOSTIC SUMMARY (--test)")
        log("=" * 60)
        if self.h['grad_emb']:
            log(f"grad norm  (last epoch): embedding={self.h['grad_emb'][-1]:.3e}"
                f"  codebook={self.h['grad_cb'][-1]:.3e}")
            log(f"param drift from init  : embedding={self.h['emb_drift'][-1]:.3e}"
                f"  codebook={self.h['cb_drift'][-1]:.3e}")
            log(f"contrastive accuracy   : {self.h['acc'][-1]:.3f} "
                f"(margin neg-pos={self.h['margin'][-1]:.4f})")
            log(f"max |embedding|        : {self.h['emb_maxnorm'][-1]:.4f}"
                + (f" / ball radius {self.ball_r:.4f}"
                   if self.args.c > 0 else ""))
            log("dead codes (final)     : "
                + ", ".join(f"Q{qi}={self.h['dead'][qi][-1]}/{self.args.bins}"
                            for qi in range(self.args.n_q)))
        log("-" * 60)
        if self.warnings:
            log(f"{len(self.warnings)} ISSUE(S) DETECTED:")
            for w in self.warnings:
                log(f"  - {w}")
        else:
            log("No issues flagged by the sanity checks.")
        log("=" * 60)

    # ---- diagnostics figure ---------------------------------------------
    def visualize(self, save_dir, history):
        if not self.enabled or not self.h['grad_emb']:
            return
        ep = history['epoch']
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle(f"Training diagnostics (c={self.args.c}, "
                     f"n_q={self.args.n_q}, bins={self.args.bins}, "
                     f"dim={self.args.embed_dim}, lr={self.args.lr})")

        a = ax[0, 0]
        a.plot(ep, self.h['grad_emb'], label='embedding')
        a.plot(ep, self.h['grad_cb'], label='codebook')
        a.set_yscale('log'); a.set_title('Gradient norm'); a.set_xlabel('epoch')
        a.legend(); a.grid(alpha=0.3)

        a = ax[0, 1]
        a.plot(ep, self.h['emb_drift'], label='embedding')
        a.plot(ep, self.h['cb_drift'], label='codebook')
        a.set_title('Param drift from init (L2)'); a.set_xlabel('epoch')
        a.legend(); a.grid(alpha=0.3)

        a = ax[0, 2]
        a.plot(ep, self.h['acc'])
        a.set_ylim(-0.02, 1.02)
        a.set_title('Contrastive accuracy (pos=argmax)')
        a.set_xlabel('epoch'); a.grid(alpha=0.3)

        a = ax[1, 0]
        a.plot(ep, self.h['pos_d'], label='positive')
        a.plot(ep, self.h['neg_d'], label='negative (mean)')
        a.set_title('Positive vs negative distance'); a.set_xlabel('epoch')
        a.legend(); a.grid(alpha=0.3)

        a = ax[1, 1]
        for qi in range(self.args.n_q):
            a.plot(ep, self.h['dead'][qi], label=f'Q{qi}')
        a.set_title(f'Dead codes (of {self.args.bins})'); a.set_xlabel('epoch')
        a.legend(fontsize=8); a.grid(alpha=0.3)

        a = ax[1, 2]
        a.plot(ep, self.h['emb_maxnorm'], label='max |x|')
        if self.args.c > 0:
            a.axhline(self.ball_r, color='r', ls='--', label='ball radius')
        a.set_title('Embedding norm vs ball'); a.set_xlabel('epoch')
        a.legend(); a.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, 'diagnostics.png')
        fig.savefig(path, dpi=130); plt.close(fig)
        print(f"[TEST] diagnostics figure saved to {path}")


def train(args):
    dataset = WordNetHierarchyDataset(num_negatives=50, split='train',
                                      split_mode=args.split_mode)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'),
        persistent_workers=args.num_workers > 0,
    )

    model = HRQModel(
        vocab_size=dataset.vocab_size, embed_dim=args.embed_dim, n_q=args.n_q,
        bins=args.bins, c=args.c, new_method=args.new_method, hste=args.hste,
        approx=args.approx, commitment_weight=args.commitment_weight,
        ema=args.ema,
    ).to(args.device)

    if args.c > 0:
        manifold_params, euclidean_params = [], []
        for p in model.parameters():
            (manifold_params if hasattr(p, "manifold") else euclidean_params).append(p)
        print(f"Manifold params: {len(manifold_params)}, Euclidean params: {len(euclidean_params)}")
        # Paper (Appendix C.1): Riemannian SGD for HRQ
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=args.warmup_lr)
    else:
        # Paper (Appendix C.1): SGD for RQ
        optimizer = optim.SGD(model.parameters(), lr=args.warmup_lr)

    # Build diagnostics on the *un-compiled* model so .named_parameters() and
    # .grad introspection stay stable (params are shared with the compiled one).
    diag = Diagnostics(model, args)

    # torch.compile fuses the many small elementwise/transcendental kernels in
    # the hyperbolic path (mobius_add, exp/log maps, atanh/acosh, project) into
    # far fewer launches. Opt-in: the custom hyperbolic autograd Function and
    # data-dependent codebook ops may cause graph breaks on some setups.
    if args.compile:
        model = torch.compile(model)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'logs.txt')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Args: {vars(args)}")
    log(f"Split mode: {dataset.split_mode}")
    log(f"Dataset size: {len(dataset)}, Batches/epoch: {len(dataloader)}")
    log("")

    history = {'epoch': [], 'loss': [], 'ce_loss': [], 'commit_loss': [], 'approx_dist': []}
    for qi in range(args.n_q):
        history[f'ppl_q{qi}'] = []
    train_start = time.time()
    for epoch in range(args.epochs):
        # Paper (Appendix C.1): warmup lr=0.01 for first 20 epochs, then lr=1.0
        if epoch == args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            log(f"Warmup done — switching lr to {args.lr}")

        model.train()
        diag.start_epoch()
        # Accumulate on-device to avoid a host<->device sync (.item()) every step.
        total_loss = torch.zeros((), device=args.device)
        total_ce = torch.zeros((), device=args.device)
        total_commit = torch.zeros((), device=args.device)
        total_dist = torch.zeros((), device=args.device)
        code_counts = [torch.zeros(args.bins, device=args.device) for _ in range(args.n_q)]
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            u = batch['u'].to(args.device, non_blocking=True)
            v = batch['v'].to(args.device, non_blocking=True)
            negatives = batch['negatives'].to(args.device, non_blocking=True)

            u_emb, _, u_codes, u_commit, u_dist = model(u)
            v_emb, _, v_codes, v_commit, v_dist = model(v)

            # Accumulate code usage for perplexity: codes shape is (n_q, B, 1)
            for qi in range(args.n_q):
                counts = torch.bincount(u_codes[qi].reshape(-1), minlength=args.bins).float()
                counts += torch.bincount(v_codes[qi].reshape(-1), minlength=args.bins).float()
                code_counts[qi] += counts

            B, num_neg = negatives.shape
            neg_flat = negatives.view(-1)
            neg_emb = model.emb_only_forward(neg_flat)
            neg_emb = neg_emb.view(B, num_neg, -1)

            # Paper: contrastive loss on continuous embeddings E_theta
            pos_dist = poincare_distance(u_emb, v_emb, args.c).squeeze(-1)
            u_expanded = u_emb.unsqueeze(1).expand(-1, num_neg, -1).reshape(B * num_neg, -1)
            neg_dist = poincare_distance(u_expanded, neg_emb.reshape(B * num_neg, -1), args.c).view(B, num_neg)

            # Add self-distance d(u,u)=0 to match the paper's H'(u) ∪ {u}
            self_dist = torch.zeros((B, 1), device=args.device)

            # Paper: log-softmax contrastive loss (no temperature scaling)
            logits = torch.cat([-pos_dist.unsqueeze(1), -neg_dist, -self_dist], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=args.device)

            ce_loss = nn.CrossEntropyLoss()(logits, labels)
            quant_penalty = u_commit.mean() + v_commit.mean()
            commit = quant_penalty * args.quant
            loss = ce_loss + commit
            diag.check_forward(
                i,
                {'u_emb': u_emb, 'v_emb': v_emb, 'logits': logits,
                 'loss': loss, 'commit': commit},
                u_emb, logits, labels, pos_dist, neg_dist)
            loss.backward()
            diag.check_grads(i)
            optimizer.step()

            total_loss += loss.detach()
            total_ce += ce_loss.detach()
            total_commit += commit.detach()
            total_dist += (u_dist.detach() + v_dist.detach()) / 2
            if i % 100 == 0:
                log(f"  Step {i}, Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Commit: {commit.item():.4f})")

        diag.log_epoch(code_counts)
        n_batches = len(dataloader)
        avg_loss = (total_loss / n_batches).item()
        avg_ce = (total_ce / n_batches).item()
        avg_commit = (total_commit / n_batches).item()
        avg_dist = (total_dist / n_batches).item()

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
        log(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} "
            f"(CE: {avg_ce:.4f}, Commit: {avg_commit:.4f}, ApproxDist: {avg_dist:.6f}), "
            f"PPL: [{', '.join(ppl_strs)}], Time: {elapsed:.0f}s")

    log(f"\nTraining finished in {time.time() - train_start:.0f}s")

    # Validation pass (no_grad over entire dataset)
    model.eval()
    val_dist_sum = torch.zeros((), device=args.device)
    val_n = 0
    with torch.no_grad():
        for batch in dataloader:
            u = batch['u'].to(args.device, non_blocking=True)
            v = batch['v'].to(args.device, non_blocking=True)
            _, _, _, _, u_dist = model(u)
            _, _, _, _, v_dist = model(v)
            val_dist_sum += (u_dist.detach() + v_dist.detach()) / 2
            val_n += 1
    val_approx_dist = (val_dist_sum / val_n).item()
    log(f"\nValidation approx_distance: {val_approx_dist:.6f}")

    log("")
    log("Perplexity summary (across epochs):")
    for qi in range(args.n_q):
        ppls = np.array(history[f'ppl_q{qi}'])
        log(f"  Q{qi}: mean={ppls.mean():.2f}, var={ppls.var():.2f}, final={ppls[-1]:.2f}")
    all_final = [history[f'ppl_q{qi}'][-1] for qi in range(args.n_q)]
    log(f"  All layers final: mean={np.mean(all_final):.2f}, var={np.var(all_final):.2f}")
    diag.report(log)
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

    # Extra diagnostics figure + persisted warning list (only when --test).
    diag.visualize(save_dir, history)
    if args.test:
        with open(os.path.join(save_dir, 'diagnostics.txt'), 'w') as f:
            f.write("DIAGNOSTIC WARNINGS (--test)\n" + "=" * 40 + "\n")
            if diag.warnings:
                for w in diag.warnings:
                    f.write(f"- {w}\n")
            else:
                f.write("No issues flagged.\n")

    model_path = os.path.join(save_dir, 'model.pt')
    # Unwrap torch.compile so checkpoint keys match the plain model (eval loads this).
    save_model = getattr(model, "_orig_mod", model)
    torch.save(save_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    config_path = os.path.join(save_dir, 'config.py')
    with open(config_path, 'w') as f:
        for k, val in vars(args).items():
            f.write(f"{k} = {repr(val)}\n")
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
    parser.add_argument('--split_mode', type=str, default='closure',
                        choices=['closure', 'direct'],
                        help="'closure' = paper-faithful transitive-closure split; "
                             "'direct' = direct-edge split (leak-free)")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes (parallelize negative '
                             'sampling so the GPU is not starved). Keep <= '
                             '--cpus-per-task in the SLURM script.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoint/nlp_2/default')
    parser.add_argument('--commitment_weight', type=float, default=0.25,
                        help='Weight for the VQ commitment loss term')
    parser.add_argument('--quant', type=float, default=1.0,
                        help='Multiplier applied to the quantizer penalty in the loss')
    parser.add_argument('--ema', action='store_true',
                        help='Enable EMA updates in the VQ codebooks')
    parser.add_argument('--compile', action='store_true',
                        help='Wrap model in torch.compile to fuse hyperbolic kernels')
    parser.add_argument('--new_method', action='store_true')
    parser.add_argument('--approx', action='store_true')
    parser.add_argument('--hste', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='Enable training diagnostics: per-component '
                             'gradient / NaN / Poincaré-ball / dead-code '
                             'checks, a per-epoch health history, and an extra '
                             'diagnostics.png + diagnostics.txt at the end.')
    args = parser.parse_args()
    train(args)
