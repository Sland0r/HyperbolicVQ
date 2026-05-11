import time
import torch
import sys
import os

from academicodec.quantization.vq import ResidualVectorQuantizer

def test_setting(c, constructive, B=128, D=128, N=16, n_q=4, bins=256, device="cuda"):
    print(f"\n{'='*80}")
    print(f"Testing c={c}, constructive={constructive}")
    print(f"{'='*80}")
    
    vq = ResidualVectorQuantizer(
        dimension=D,
        n_q=n_q,
        bins=bins,
        c=c,
        ema=True,
        threshold_ema_dead_code=2,
    ).to(device)

    # Initialize constructive logic
    if constructive:
        sys.path.insert(0, '/home/acolombo/music/hyperbolic_tree_embeddings')
        from tree_embeddings.trees.file_utils import load_hierarchy
        from tree_embeddings.embeddings.constructive_method import constructively_embed_tree

        hierarchy = load_hierarchy(dataset="n_h_trees", hierarchy_name=f"{bins}_1")

        curvature = c if c > 0 else 1.0
        embeddings, _, _ = constructively_embed_tree(
            hierarchy=hierarchy,
            dataset="n_h_trees",
            hierarchy_name=f"{bins}_1",
            embedding_dim=D,
            tau=1.0,
            nc=1,
            curvature=curvature,
            root=0,
            gen_type="optim",
            dtype=torch.float64,
        )

        code_points = embeddings[1:].to(dtype=torch.float32, device=device)
        
        with torch.no_grad():
            for qi in range(n_q):
                cb = vq.vq.layers[qi]._codebook
                cb.embed.data.copy_(code_points)
                cb.embed_avg.data.copy_(code_points)
                cb.inited.data.copy_(torch.Tensor([True]))
                cb.cluster_size.data.fill_(2 + 1)
                
    vq.train()
    
    x = torch.randn(B, D, N, device=device)
    x.requires_grad_(True)
    
    print("Running warmup...")
    # Warmup
    for _ in range(3):
        quantized, codes, bw, commit_loss = vq(x, sample_rate=16000, nq=n_q)
        loss = commit_loss + quantized.sum()
        loss.backward()
        if x.grad is not None:
            x.grad.zero_()
        
    print("Profiling with torch.profiler...")
    
    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(5):
            quantized, codes, bw, commit_loss = vq(x, sample_rate=16000, nq=n_q)
            loss = commit_loss + quantized.sum()
            loss.backward()
            if x.grad is not None:
                x.grad.zero_()
            
    print(f"\n--- Profiling Results (Sorted by CUDA time, Top 30) for [c={c}, constructive={constructive}] ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    test_setting(c=0.0, constructive=False, device=device)
    test_setting(c=1.0, constructive=False, device=device)
    test_setting(c=1.0, constructive=True, device=device)
