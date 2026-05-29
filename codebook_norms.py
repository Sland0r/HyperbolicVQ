#!/usr/bin/env python3
"""Compute mean L2 norm of all codebook entries in a checkpoint."""

import argparse
import glob
import os
import sys

import torch


def find_checkpoint(path):
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
    if os.path.isdir(path):
        latest = os.path.join(path, "latest.pth")
        if os.path.isfile(latest):
            return latest
        pths = sorted(glob.glob(os.path.join(path, "*.pth")))
        if pths:
            return pths[-1]
    print(f"No .pth checkpoint found in {path}", file=sys.stderr)
    sys.exit(1)


def extract_codebooks(state_dict):
    codebooks = {}
    for key, tensor in state_dict.items():
        if "_codebook.embed" in key and "embed_avg" not in key:
            codebooks[key] = tensor
    return codebooks


def load_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    for model_key in ("soundstream", "model"):
        if model_key in ckpt:
            codebooks = extract_codebooks(ckpt[model_key])
            if codebooks:
                return codebooks

    codebooks = extract_codebooks(ckpt)
    if codebooks:
        return codebooks

    print("No codebook embeddings found in checkpoint.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Checkpoint folder or .pth file")
    args = parser.parse_args()

    ckpt_path = find_checkpoint(args.path)
    print(f"Loading: {ckpt_path}")

    codebooks = load_codebooks(ckpt_path)
    n_layers = len(codebooks)

    all_norms = []
    for key in sorted(codebooks.keys()):
        embed = codebooks[key].float()
        norms = torch.norm(embed, dim=-1)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        all_norms.append(norms)
        layer_idx = key.split("layers.")[1].split(".")[0]
        print(f"  Layer {layer_idx:>2}: mean_norm={mean_norm:.4f}  std={std_norm:.4f}  "
              f"min={norms.min().item():.4f}  max={norms.max().item():.4f}  "
              f"shape={list(embed.shape)}")

    all_norms = torch.cat(all_norms)
    print(f"\nOverall ({n_layers} layers, {all_norms.numel()} codes): "
          f"mean_norm={all_norms.mean().item():.4f}  std={all_norms.std().item():.4f}")


if __name__ == "__main__":
    main()
