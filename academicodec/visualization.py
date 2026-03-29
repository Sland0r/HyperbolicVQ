import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

def fit_pca(codes, c, n_components=2):
    """
    Fit a PCA on the codebook embeddings.
    If c > 0, first map the hyperbolic embeddings back to Euclidean space using log_map0.
    """
    if c > 0:
        from academicodec.quantization.core_vq import log_map0
        codes = log_map0(codes, c)
    pca = PCA(n_components=n_components)
    pca.fit(codes.numpy())
    return pca

def plot_codes(codes, pca, c, step, output_dir):
    """
    Plot the codebook embeddings in 2D using the fitted PCA.
    If c > 0, first map the embeddings to Euclidean space using log_map0 before PCA,
    and then map the 2D projected coordinates back to the Poincare disk using exp_map0.
    Saves the plot as a PNG.
    """
    if c > 0:
        from academicodec.quantization.core_vq import log_map0, exp_map0
        codes = log_map0(codes, c)
        
    codes_2d_euclidean = pca.transform(codes.numpy())
    codes_2d_euclidean = torch.tensor(codes_2d_euclidean, dtype=torch.float32)
    
    if c > 0:
        from academicodec.quantization.core_vq import exp_map0
        codes_2d = exp_map0(codes_2d_euclidean, c).numpy()
    else:
        codes_2d = codes_2d_euclidean.numpy()
        
    plt.figure(figsize=(8, 8))
    plt.scatter(codes_2d[:, 0], codes_2d[:, 1], alpha=0.6, s=15, c='blue')
    plt.title(f"Codebook Evolution - Step {step}")
    
    if c > 0:
        # Draw the boundary of the Poincare disk
        radius = 1.0 / np.sqrt(c) if c > 0 else 1.0
        circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        plt.xlim(-radius - 0.1, radius + 0.1)
        plt.ylim(-radius - 0.1, radius + 0.1)
    else:
        # Per-frame adaptive bounds with a minimum floor to avoid near-zero axes
        current_radius = float(np.max(np.abs(codes_2d))) if codes_2d.size > 0 else 0.5
        bound = max(current_radius, 0.5) * 1.3
        plt.xlim(-bound, bound)
        plt.ylim(-bound, bound)
    
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"step_{step:06d}.png"))
    plt.close()

def create_gif(image_folder, output_path, duration=200):
    """
    Create a GIF from the saved PNG frames in the image folder.
    """
    if not os.path.exists(image_folder):
        return
    filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    if not filenames:
        print(f"No PNG frames found in {image_folder} to create GIF.")
        return
        
    images = [Image.open(os.path.join(image_folder, fn)) for fn in filenames]
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"Saved GIF to {output_path}")
