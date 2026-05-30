#!/usr/bin/env python3
"""Backfill loss curves for NLP_2 checkpoints from logs.txt.

Creates loss_curves.png in each checkpoint folder. If logs include PPL and
approx-distance metrics, those plots are populated; otherwise, placeholders are
shown to keep layout consistent with NLP_1.
"""

import argparse
import ast
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_config(checkpoint_dir):
    config_path = os.path.join(checkpoint_dir, 'config.py')
    if not os.path.exists(config_path):
        return {}
    cfg = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        exec(f.read(), cfg)
    return {k: v for k, v in cfg.items() if not k.startswith('_')}


def parse_args_line(line):
    if not line.startswith('Args: '):
        return None
    payload = line[len('Args: '):].strip()
    try:
        return ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return None


def parse_logs(log_path):
    epochs = []
    losses = []
    ce_losses = []
    commit_losses = []
    approx_dists = []
    ppl_by_q = {}
    val_approx_dist = None
    args_from_log = None

    epoch_re = re.compile(
        r"Epoch\s+(\d+)/(\d+),\s+Loss:\s+([0-9.]+)\s+\(CE:\s+([0-9.]+),\s+Commit:\s+([0-9.]+)"
    )
    approx_re = re.compile(r"ApproxDist:\s+([0-9.]+)")
    ppl_re = re.compile(r"Q(\d+)=([0-9.]+)")
    val_re = re.compile(r"Validation approx_distance:\s+([0-9.]+)")

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if args_from_log is None:
                maybe_args = parse_args_line(line)
                if isinstance(maybe_args, dict):
                    args_from_log = maybe_args

            val_match = val_re.search(line)
            if val_match:
                val_approx_dist = float(val_match.group(1))

            match = epoch_re.search(line)
            if not match:
                continue

            epoch = int(match.group(1))
            loss = float(match.group(3))
            ce = float(match.group(4))
            commit = float(match.group(5))

            epochs.append(epoch)
            losses.append(loss)
            ce_losses.append(ce)
            commit_losses.append(commit)

            approx_match = approx_re.search(line)
            if approx_match:
                approx_dists.append(float(approx_match.group(1)))

            for q, val in ppl_re.findall(line):
                q_idx = int(q)
                ppl_by_q.setdefault(q_idx, []).append(float(val))

    return {
        'epochs': epochs,
        'losses': losses,
        'ce_losses': ce_losses,
        'commit_losses': commit_losses,
        'approx_dists': approx_dists,
        'ppl_by_q': ppl_by_q,
        'val_approx_dist': val_approx_dist,
        'args_from_log': args_from_log or {},
    }


def plot_checkpoint(checkpoint_dir):
    log_path = os.path.join(checkpoint_dir, 'logs.txt')
    if not os.path.exists(log_path):
        print(f"[skip] no logs.txt in {checkpoint_dir}")
        return False

    data = parse_logs(log_path)
    if not data['epochs']:
        print(f"[skip] no epoch data in {log_path}")
        return False

    cfg = load_config(checkpoint_dir)
    args_from_log = data['args_from_log']
    n_q = cfg.get('n_q') or args_from_log.get('n_q')
    bins = cfg.get('bins') or args_from_log.get('bins')

    epochs = data['epochs']
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    # Loss curves
    for ax, series, label in [
        (axes[0], data['losses'], 'Total Loss'),
        (axes[1], data['ce_losses'], 'CE Loss'),
        (axes[2], data['commit_losses'], 'Commit Loss'),
    ]:
        ax.plot(epochs, series)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    # Perplexity
    ax_ppl = axes[3]
    if data['ppl_by_q']:
        for q_idx in sorted(data['ppl_by_q'].keys()):
            ax_ppl.plot(epochs[:len(data['ppl_by_q'][q_idx])], data['ppl_by_q'][q_idx], label=f"Q{q_idx}")
        if bins is not None:
            ax_ppl.axhline(y=bins, color='k', linestyle='--', alpha=0.3, label=f"max ({bins})")
        ax_ppl.legend(fontsize=8)
        ax_ppl.set_xlabel('Epoch')
        ax_ppl.set_ylabel('Perplexity')
        ax_ppl.set_title('Codebook Perplexity')
        ax_ppl.grid(True, alpha=0.3)
    else:
        ax_ppl.set_title('Codebook Perplexity')
        ax_ppl.text(0.5, 0.5, 'Perplexity not logged', ha='center', va='center')
        ax_ppl.set_xticks([])
        ax_ppl.set_yticks([])

    # Approx distance
    ax_dist = axes[4]
    if data['approx_dists']:
        ax_dist.plot(epochs[:len(data['approx_dists'])], data['approx_dists'], label='Train')
        if data['val_approx_dist'] is not None:
            ax_dist.axhline(y=data['val_approx_dist'], color='r', linestyle='--', alpha=0.7,
                            label=f"Val ({data['val_approx_dist']:.4f})")
        ax_dist.legend(fontsize=8)
        ax_dist.set_xlabel('Epoch')
        ax_dist.set_ylabel('Approx Distance')
        ax_dist.set_title('Hyperbolic Approx Distance')
        ax_dist.grid(True, alpha=0.3)
    else:
        ax_dist.set_title('Hyperbolic Approx Distance')
        ax_dist.text(0.5, 0.5, 'Approx distance not logged', ha='center', va='center')
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])

    fig.tight_layout()
    out_path = os.path.join(checkpoint_dir, 'loss_curves.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root', default='/home/acolombo/VAEs/checkpoint/nlp_2')
    parser.add_argument('--only', nargs='*', help='Optional list of run folder names to process')
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_root):
        print(f"checkpoint_root not found: {args.checkpoint_root}")
        return 1

    run_dirs = []
    if args.only:
        run_dirs = [os.path.join(args.checkpoint_root, name) for name in args.only]
    else:
        for name in sorted(os.listdir(args.checkpoint_root)):
            path = os.path.join(args.checkpoint_root, name)
            if os.path.isdir(path):
                run_dirs.append(path)

    processed = 0
    for run_dir in run_dirs:
        if plot_checkpoint(run_dir):
            processed += 1

    print(f"Done. Processed {processed} checkpoints.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
