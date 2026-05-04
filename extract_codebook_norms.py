#!/usr/bin/env python3
"""Extract and compare codebook norms at initialization vs end of training."""

import os
import sys
import torch

# Add paths for tree embeddings
sys.path.insert(0, '/home/acolombo/music/hyperbolic_tree_embeddings')

from tree_embeddings.trees.file_utils import load_hierarchy
from tree_embeddings.embeddings.constructive_method import constructively_embed_tree


def extract_norms(job_id):
    """Extract init and final codebook norms for a job."""
    base = f'/home/acolombo/VAEs/checkpoint/soundstream/{job_id}'
    ckpt_path = os.path.join(base, 'latest.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"JOB ID: {job_id}")
    print('='*70)
    
    # Load checkpoint
    state = torch.load(ckpt_path, map_location='cpu')
    ss = state.get('soundstream', state)
    
    # Extract final codebook norms
    final_stats = []
    for k, v in ss.items():
        if torch.is_tensor(v) and 'quantizer.vq.layers' in k and k.endswith('._codebook.embed'):
            n = v.norm(dim=-1)
            layer = k.split('quantizer.vq.layers.')[1].split('._codebook.embed')[0]
            final_stats.append((
                int(layer),
                n.min().item(),
                n.max().item(),
                n.mean().item(),
                n.std().item()
            ))
    
    final_stats.sort(key=lambda x: x[0])
    
    # Reconstruct constructive init (fixed hyperparams from job)
    bins = 1024  # From config logs; main3_ddp.py uses this
    D = 512      # Hardcoded in main3_ddp.py for constructive init
    curvature = 1.0  # From config logs for job 22425043
    
    try:
        hierarchy = load_hierarchy(dataset='n_h_trees', hierarchy_name=f'{bins}_1')
        embeddings, _, _ = constructively_embed_tree(
            hierarchy=hierarchy,
            dataset='n_h_trees',
            hierarchy_name=f'{bins}_1',
            embedding_dim=D,
            tau=1.0,
            nc=1,
            curvature=curvature,
            root=0,
            gen_type='optim',
            dtype=torch.float32,
        )
        init_codes = embeddings[1:].to(torch.float32)  # Skip root
        init_norm = init_codes.norm(dim=-1)
        
        print(f"\n--- INITIALIZATION (Constructive) ---")
        print(f"shape: {tuple(init_codes.shape)}")
        print(f"norm min:  {init_norm.min().item():.6f}")
        print(f"norm max:  {init_norm.max().item():.6f}")
        print(f"norm mean: {init_norm.mean().item():.6f}")
        print(f"norm std:  {init_norm.std().item():.6f}")
        
    except Exception as e:
        print(f"Failed to compute init norms: {e}")
        init_norm = None
    
    print(f"\n--- FINAL CHECKPOINT ---")
    if final_stats:
        print(f"{'Layer':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 56)
        for layer, mn, mx, mean, std in final_stats:
            print(f"{layer:<8} {mn:<12.6f} {mx:<12.6f} {mean:<12.6f} {std:<12.6f}")
        
        means = torch.tensor([x[3] for x in final_stats])
        print("\n--- Summary across all layers ---")
        print(f"mean min:  {means.min().item():.6f}")
        print(f"mean max:  {means.max().item():.6f}")
        print(f"mean avg:  {means.mean().item():.6f}")
        
        if init_norm is not None:
            print(f"\n--- CHANGE (Init → Final) ---")
            init_mean = init_norm.mean().item()
            final_mean = means.mean().item()
            pct_change = ((final_mean - init_mean) / init_mean * 100) if init_mean != 0 else 0
            print(f"init mean:  {init_mean:.6f}")
            print(f"final mean: {final_mean:.6f}")
            print(f"change:     {final_mean - init_mean:+.6f} ({pct_change:+.1f}%)")
    else:
        print("No codebook embeddings found in checkpoint")


if __name__ == '__main__':
    # Check multiple recent jobs
    jobs = ['22425043', '22425048', '22425042', '22424124', '22424054']
    
    for job_id in jobs:
        extract_norms(job_id)
