import torch
import time
import argparse
import sys
sys.path.insert(0, '/home/acolombo/VAEs')
from egs.MNIST_VQVAE.mnist_vqvae import VQVAE2D
import geoopt

def test(constructive):
    device = "cpu"
    model = VQVAE2D(
        D=128, n_q=4, bins=256, c=1.0, 
        ema=False, kmeans_init=False, 
        threshold_ema_dead_code=2, 
        in_channels=1, img_size=28
    ).to(device)

    if constructive:
        import sys
        sys.path.insert(0, '/home/acolombo/music/hyperbolic_tree_embeddings')
        from tree_embeddings.trees.file_utils import load_hierarchy
        from tree_embeddings.embeddings.constructive_method import constructively_embed_tree
        hierarchy = load_hierarchy(dataset="n_h_trees", hierarchy_name="256_1")
        embeddings, _, _ = constructively_embed_tree(
            hierarchy=hierarchy, dataset="n_h_trees", hierarchy_name="256_1",
            embedding_dim=128, tau=1.0, nc=1, curvature=1.0, root=0, gen_type="optim", dtype=torch.float64
        )
        code_points = embeddings[1:].to(dtype=torch.float32, device=device)
        with torch.no_grad():
            for qi in range(4):
                cb = model.quantizer.vq.layers[qi]._codebook
                cb.embed.data.copy_(code_points)
                cb.embed_avg.data.copy_(code_points)
                cb.inited.data.copy_(torch.Tensor([True]))
                cb.cluster_size.data.fill_(1)

    manifold_params = []
    euclidean_params = []
    for p in model.parameters():
        if hasattr(p, "manifold"):
            manifold_params.append(p)
        else:
            euclidean_params.append(p)
            
    param_groups = []
    if len(manifold_params) > 0:
        param_groups.append({"params": manifold_params, "lr": 1e-3, "betas": (0.0, 0.95), "eps": 1e-5})
    if len(euclidean_params) > 0:
        param_groups.append({"params": euclidean_params, "lr": 3e-4, 'betas': (0.5, 0.9)})
    
    optimizer = geoopt.optim.RiemannianAdam(param_groups)
    
    x = torch.randn(128, 1, 28, 28, device=device)
    
    # Warmup
    for _ in range(5):
        x_hat, latent_loss, codes = model(x)
        loss = torch.nn.functional.mse_loss(x_hat, x) + latent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    t0 = time.time()
    total_fw, total_bw = 0, 0
    for _ in range(50):
        t_fw0 = time.time()
        x_hat, latent_loss, codes = model(x)
        loss = torch.nn.functional.mse_loss(x_hat, x) + latent_loss
        
        t_fw1 = time.time()
        total_fw += t_fw1 - t_fw0
        
        t_bw0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t_bw1 = time.time()
        total_bw += t_bw1 - t_bw0

    print(f"Constructive={constructive} | Avg FW: {total_fw/50:.4f}s | Avg BW: {total_bw/50:.4f}s")
    
test(False)
test(True)
