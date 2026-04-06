import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Plot VQ-VAE embeddings from checkpoint iteratively")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint")
    #parser.add_argument("--c", type=float, default=None, help="Curvature (c > 0 for Poincare ball). If not provided, inferred from config.")
    parser.add_argument("--output", type=str, default="", help="Output image file path")
    parser.add_argument("--plot_images", action="store_true", help="Plot decoded images instead of scatter points")
    parser.add_argument("--image_zoom", type=float, default=0.5, help="Zoom factor for plotted images")
    return parser.parse_args()


def plot_image_at_point(ax, point, image, zoom, zorder=10):
    """Plot a single image at a specific (x, y) coordinate."""
    # Ensure image is in 0-1 range and proper shape (H, W) or (H, W, 3)
    if image.shape[0] == 1:
        image = image.squeeze(0)  # Grayscale
    elif image.shape[0] == 3:
        image = image.transpose(1, 2, 0)  # RGB
        
    imagebox = OffsetImage(image, zoom=zoom, cmap="gray")
    ab = AnnotationBbox(imagebox, point, frameon=False, pad=0.0)
    ab.zorder = zorder
    ax.add_artist(ab)


def main():
    args = get_args()
    
    checkpoint_dir = os.path.dirname(args.checkpoint)
    
    # Try to infer config values
    #c = args.c
    dataset = "mnist"
    #if c is None:
    try:
        sys.path.insert(0, checkpoint_dir)
        import config
        c = hasattr(config, "c") and config.c or 0.0
        dataset = hasattr(config, "dataset") and config.dataset or "mnist"
        sys.path.pop(0)
        print(f"Inferred c={c}, dataset={dataset} from {checkpoint_dir}/config.py")
    except Exception as e:
        print(f"Could not read config: {e}. Defaulting to c=0.0")
        c = 0.0

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    
    codebooks = []
    # Collect all codebooks
    for i in range(100):
        found_key = None
        for k in state_dict.keys():
            if f"layers.{i}._codebook.embed" in k:
                found_key = k
                break
        if found_key:
            cb_embed = state_dict[found_key].detach().cpu()
            codebooks.append(cb_embed)
        else:
            break
            
    if not codebooks:
        raise ValueError(f"No codebooks found in checkpoint.")

    if c > 0:
        from academicodec.quantization.core_vq import mobius_add

    dim = codebooks[0].shape[-1]
    
    # Setup decoder if plotting images
    decoder = None
    if args.plot_images:
        print("Initializing decoder for image plotting...")
        sys.path.insert(0, "/home/acolombo/VAEs/egs/MNIST_VQVAE")
        from mnist_vqvae import Decoder
        sys.path.pop(0)
        
        in_channels = 3 if dataset == "cifar100" else 1
        img_size = 32 if dataset == "cifar100" else 28
        
        decoder = Decoder(D=dim, out_channels=in_channels, img_size=img_size)
        
        # Extract decoder state dict
        decoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("decoder."):
                decoder_state[k.replace("decoder.", "")] = v
        decoder.load_state_dict(decoder_state)
        decoder.eval()
        print("Decoder loaded successfully.")

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot origin
    if not args.plot_images:
        ax.scatter([0], [0], c="black", marker="x", s=100, label="Origin", zorder=100)
    
    current_points = [torch.zeros(dim)]
    all_points_to_bound = [np.zeros(2)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(codebooks)))

    for layer_idx, cb in enumerate(codebooks):
        new_points = []
        lines_x = []
        lines_y = []
        
        for curr_pt in current_points:
            for cb_pt in cb:
                if c > 0:
                    new_pt = mobius_add(curr_pt.unsqueeze(0), cb_pt.unsqueeze(0), c).squeeze(0)
                else:
                    new_pt = curr_pt + cb_pt
                    
                new_points.append(new_pt)
                
                pt1 = curr_pt.numpy()[:2]
                pt2 = new_pt.numpy()[:2]
                
                lines_x.extend([pt1[0], pt2[0], None])
                lines_y.extend([pt1[1], pt2[1], None])
                
        # Draw all lines for this layer
        ax.plot(lines_x, lines_y, color=colors[layer_idx], alpha=0.3, linestyle="-", linewidth=1, zorder=layer_idx)
                
        # Handle the new points
        new_points_np = np.stack([pt.numpy()[:2] for pt in new_points])
        
        if args.plot_images:
            # Decode each point and plot image
            # The Decoder expects input (1, D, N) where N=16
            with torch.no_grad():
                for pt, pt_np in zip(new_points, new_points_np):
                    # Replicate point to spatial grid
                    z_map = pt.unsqueeze(0).unsqueeze(2).repeat(1, 1, 16)  # (1, D, 16)
                    decoded_img = decoder(z_map).squeeze(0)  # (C, H, W)
                    decoded_img_np = decoded_img.numpy()
                    
                    # Plot the image at pt_np
                    plot_image_at_point(ax, pt_np, decoded_img_np, args.image_zoom, zorder=layer_idx + 10)
        else:
            sz = 50 if layer_idx == 0 else max(5, 50 - layer_idx * 15)
            rgb_color = colors[layer_idx].reshape(1, -1)
            ax.scatter(
                new_points_np[:, 0], new_points_np[:, 1], 
                c=rgb_color, s=sz, alpha=0.8, 
                label=f"Codebook layer {layer_idx}", 
                zorder=layer_idx + 10
            )
        
        all_points_to_bound.extend(new_points_np)
        current_points = new_points
        
    all_points_to_bound = np.array(all_points_to_bound)
    
    if c > 0:
        radius = 1.0 / np.sqrt(c)
        circle = plt.Circle((0, 0), radius, color="green", fill=False, linestyle="-", linewidth=2, label="Poincaré boundary")
        ax.add_patch(circle)
        bound = radius * 1.1
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        plt.title(f"Hierarchy of Codebooks ({'Images' if args.plot_images else 'Points'}, Poincaré, c={c})")
    else:
        current_radius = float(np.max(np.abs(all_points_to_bound))) if all_points_to_bound.size > 0 else 0.5
        bound = max(current_radius, 0.5) * 1.1
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        plt.title(f"Hierarchy of Codebooks ({'Images' if args.plot_images else 'Points'}, Euclidean)")

    ax.grid(True, alpha=0.3)
    if not args.plot_images:
        ax.legend()
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    
    output_path = args.output
    if not output_path:
        checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]
        suffix = "_images" if args.plot_images else "_points"
        output_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_hierarchy{suffix}.png")
        
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
