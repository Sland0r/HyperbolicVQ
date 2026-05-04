import argparse
import os
import sys

# Ensure the root project directory and MNIST_VQVAE are in sys.path
sys.path.insert(0, "/home/acolombo/VAEs")
sys.path.insert(0, "/home/acolombo/VAEs/egs/MNIST_VQVAE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate VQ-VAE model: robustness and generation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--data_dir", type=str, default="/home/acolombo/VAEs/dataset/MNIST")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "emnist", "cifar100"])
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of generated samples for FID")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()

def load_model(args, device):
    checkpoint_dir = os.path.dirname(args.checkpoint)
    sys.path.insert(0, checkpoint_dir)
    try:
        import config
    except ImportError:
        raise RuntimeError(f"Could not find config.py in {checkpoint_dir}")
    sys.path.pop(0)

    from mnist_vqvae import VQVAE2D
    in_channels = 3 if args.dataset == "cifar100" else 1
    img_size = 32 if args.dataset == "cifar100" else 28

    model = VQVAE2D(
        D=config.D,
        n_q=config.n_q,
        bins=config.bins,
        c=config.c,
        exponential_lambda=getattr(config, "exponential_lambda", 0.0),
        uniform=getattr(config, "uniform", False),
        ema=getattr(config, "ema", False),
        kmeans_init=getattr(config, "kmeans_init", False),
        threshold_ema_dead_code=getattr(config, "threshold_ema_dead_code", 2),
        codebook_weight=getattr(config, "codebook_weight", 1.0),
        commitment_weight=getattr(config, "commitment_weight", 0.25),
        dot_product_weight=getattr(config, "dot_product_weight", 0.0),
        entailment_cone_weight=getattr(config, "entailment_cone_weight", 0.0),
        in_channels=in_channels,
        img_size=img_size,
    ).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

def evaluate_robustness(model, val_loader, device, output_dir):
    print("Evaluating robustness to noise...")
    sigmas = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    mse_input = []
    mse_latent_pre = []
    mse_latent_post = []

    for sigma in sigmas:
        total_mse_in = 0
        total_mse_pre = 0
        total_mse_post = 0
        count = 0

        for imgs, _ in tqdm(val_loader, desc=f"Sigma {sigma}", leave=False):
            imgs = imgs.to(device)
            B = imgs.size(0)

            with torch.no_grad():
                # 1. Input noise
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                recon_in, _, _ = model(noisy_imgs)
                total_mse_in += F.mse_loss(recon_in, imgs, reduction="sum").item()

                # 2. Latent noise (pre-quantization)
                z = model.encoder(imgs) # (B, D, 1)
                z_noisy = z + torch.randn_like(z) * sigma
                bw = model.target_bandwidths[-1]
                quantized_pre, _, _, _ = model.quantizer(z_noisy, model.frame_rate, bw, nq=model.n_q, validation=False)
                recon_pre = model.decoder(quantized_pre)
                total_mse_pre += F.mse_loss(recon_pre, imgs, reduction="sum").item()

                # 3. Latent noise (post-quantization)
                quantized_post, _, _, _ = model.quantizer(z, model.frame_rate, bw, nq=model.n_q, validation=False)
                quantized_post_noisy = quantized_post + torch.randn_like(quantized_post) * sigma
                recon_post = model.decoder(quantized_post_noisy)
                total_mse_post += F.mse_loss(recon_post, imgs, reduction="sum").item()

            count += B * imgs.size(1) * imgs.size(2) * imgs.size(3) # MSE per pixel

        mse_input.append(total_mse_in / count)
        mse_latent_pre.append(total_mse_pre / count)
        mse_latent_post.append(total_mse_post / count)

    print("MSE Input:", mse_input)
    print("MSE Latent Pre:", mse_latent_pre)
    print("MSE Latent Post:", mse_latent_post)
    plt.figure(figsize=(8, 6))
    plt.plot(sigmas, mse_input, marker='o', label='Input Noise')
    plt.plot(sigmas, mse_latent_pre, marker='s', label='Latent Noise (Pre-Quant)')
    plt.plot(sigmas, mse_latent_post, marker='^', label='Latent Noise (Post-Quant)')
    plt.xlabel('Gaussian Noise Std Dev (\u03c3)')
    plt.ylabel('Reconstruction MSE')
    plt.title('Robustness to Noise')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(output_dir, "robustness_mse.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved robustness plot to {out_path}")

def generate_and_evaluate(model, config, val_loader, args, device, output_dir):
    print("Calculating empirical codebook frequencies...")
    bw = model.target_bandwidths[-1]
    
    # frequencies shape: (n_q, bins)
    frequencies = torch.zeros((config.n_q, config.bins), device=device)
    
    real_images_for_fid = []
    
    with torch.no_grad():
        for imgs, _ in tqdm(val_loader, desc="Extracting codes", leave=False):
            imgs = imgs.to(device)
            _, _, codes, _ = model(imgs, validation=True) 
            # codes: (n_q, B, N) where N=1 for MNIST
            for q in range(config.n_q):
                q_codes = codes[q].flatten()
                freq = torch.bincount(q_codes, minlength=config.bins)
                frequencies[q] += freq
                
            real_images_for_fid.append(imgs.cpu())
            
    # Normalize to get probabilities
    probs = frequencies / frequencies.sum(dim=1, keepdim=True)
    
    print(f"Generating {args.num_samples} samples...")
    # Sample codes
    sampled_codes = []
    for q in range(config.n_q):
        # Sample B indices for this quantizer level
        samples = torch.multinomial(probs[q], args.num_samples, replacement=True)
        sampled_codes.append(samples.unsqueeze(1)) # (num_samples, 1) -> since N=1
        
    sampled_codes = torch.stack(sampled_codes, dim=0).to(device) # (n_q, num_samples, N)
    
    # Decode
    generated_images = []
    with torch.no_grad():
        # Decode in batches to avoid OOM
        for i in range(0, args.num_samples, args.batch_size):
            batch_codes = sampled_codes[:, i:i+args.batch_size, :]
            batch_gen = model.decode(batch_codes)
            generated_images.append(batch_gen.cpu())
            
    generated_images = torch.cat(generated_images, dim=0)
    
    # Save a grid of 64 images
    grid_images = generated_images[:64]
    grid_path = os.path.join(output_dir, "generated_samples.png")
    save_image(grid_images, grid_path, nrow=8, normalize=True)
    print(f"Saved generated samples to {grid_path}")
    
    # FID & IS
    print("Calculating FID and IS...")
    real_images = torch.cat(real_images_for_fid, dim=0)
    
    # Shuffle and subset real images to match num_samples
    idx = torch.randperm(len(real_images))[:args.num_samples]
    real_images = real_images[idx]
    
    # Convert to 3 channel uint8 [0, 255]
    def prepare_for_inception(imgs):
        if imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # Ensure range [0, 1] then scale to [0, 255]
        imgs = torch.clamp(imgs, 0, 1)
        imgs = (imgs * 255).to(torch.uint8)
        return imgs
        
    real_prepared = prepare_for_inception(real_images)
    gen_prepared = prepare_for_inception(generated_images)
    
    if HAS_TORCHMETRICS:
        try:
            fid = FrechetInceptionDistance(feature=2048, normalize=False)
            fid.update(real_prepared, real=True)
            fid.update(gen_prepared, real=False)
            fid_score = fid.compute().item()
            print(f"FID Score: {fid_score:.4f}")
            
            # IS
            # IS expects uint8 [0, 255] if normalize=False
            inception = InceptionScore(normalize=False)
            inception.update(gen_prepared)
            is_mean, is_std = inception.compute()
            print(f"Inception Score: {is_mean.item():.4f} \u00b1 {is_std.item():.4f}")
            
        except Exception as e:
            print(f"Failed to compute FID/IS: {e}")
    else:
        print("Skipping FID and IS calculation: 'torchmetrics' is not installed.")
        print("To compute metrics, please run: pip install torchmetrics")

def evaluate_latent_interpretability(model, config, train_loader, val_loader, device, output_dir):
    print("Evaluating latent interpretability (training a 2-layer perceptron)...")
    
    in_dim = config.D
    hidden_dim = 128
    
    # Determine out_dim based on dataset
    if "cifar" in train_loader.dataset.__class__.__name__.lower():
        out_dim = 100
    elif "emnist" in train_loader.dataset.__class__.__name__.lower():
        out_dim = 62 # byclass split has 62 classes
    else:
        out_dim = 10

    if config.c > 0:
        from hypll.manifolds.poincare_ball import PoincareBall as HypllPoincareBall
        from hypll.manifolds.poincare_ball import Curvature
        from hypll.nn.modules.linear import HLinear
        from hypll.tensors.manifold_tensor import ManifoldTensor
        from academicodec.quantization.core_vq import exp_map0, project, log_map0
        
        class HyperbolicMLP(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim, c):
                super().__init__()
                self.c = c
                self.manifold = HypllPoincareBall(c=Curvature(c))
                self.fc = HLinear(in_dim, out_dim, manifold=self.manifold)

            def forward(self, x):
                # x is in Euclidean space (tangent space at origin)
                x_ball = project(exp_map0(x, self.c), self.c)
                x_man = ManifoldTensor(x_ball, manifold=self.manifold)
                
                out_man = self.fc(x_man)
                logits = log_map0(out_man.tensor, self.c)
                return logits

        mlp = HyperbolicMLP(in_dim, hidden_dim, out_dim, config.c).to(device)
        from hypll.optim import RiemannianAdam
        optimizer = RiemannianAdam(mlp.parameters(), lr=1e-3)
    else:
        class EuclideanMLP(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.fc = nn.Linear(in_dim, out_dim)

            def forward(self, x):
                return self.fc(x)

        mlp = EuclideanMLP(in_dim, hidden_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = 10
    mlp.train()
    for epoch in range(epochs):
        for imgs, labels in tqdm(train_loader, desc=f"MLP Epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                z = model.encoder(imgs).squeeze(-1) # (B, D)
            
            optimizer.zero_grad()
            logits = mlp(z)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    mlp.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating MLP", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            z = model.encoder(imgs).squeeze(-1)
            logits = mlp(z)
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    accuracy = 100.0 * val_correct / val_total
    print(f"Latent Interpretability Accuracy (Linear Probing): {accuracy:.2f}%")
    
    with open(os.path.join(output_dir, "latent_interpretability.txt"), "w") as f:
        f.write(f"Dataset: {train_loader.dataset.__class__.__name__}\n")
        f.write(f"Manifold: {'Hyperbolic (c=' + str(config.c) + ')' if config.c > 0 else 'Euclidean'}\n")
        f.write(f"Validation Accuracy: {accuracy:.2f}%\n")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Resolve the absolute path of the checkpoint to correctly determine the output directory
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join("/home/acolombo/VAEs/checkpoint/mnist_vqvae", args.checkpoint)
        
    output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset
    transform = transforms.ToTensor()
    if args.dataset == "mnist":
        train_data = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == "emnist":
        train_data = datasets.EMNIST(root=args.data_dir, split="byclass", train=True, download=True, transform=transform)
        val_data = datasets.EMNIST(root=args.data_dir, split="byclass", train=False, download=True, transform=transform)
    elif args.dataset == "cifar100":
        train_data = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
        val_data = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model, config = load_model(args, device)
    
    evaluate_robustness(model, val_loader, device, output_dir)
    generate_and_evaluate(model, config, val_loader, args, device, output_dir)
    evaluate_latent_interpretability(model, config, train_loader, val_loader, device, output_dir)

if __name__ == "__main__":
    main()
