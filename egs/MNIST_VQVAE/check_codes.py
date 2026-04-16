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
    parser.add_argument("--val_scatter", action="store_true", help="Plot validation points and first codebook images")
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
    dataset = "mnist"
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
        from academicodec.quantization.core_vq import mobius_add, project, exp_map0

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
                    new_pt = project(new_pt.unsqueeze(0), c).squeeze(0)
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
            # The Decoder expects input (1, D, 1)
            with torch.no_grad():
                for pt, pt_np in zip(new_points, new_points_np):
                    # No need to repeat spatially since N=1
                    z_map = pt.unsqueeze(0).unsqueeze(2)  # (1, D, 1)
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
    print(f"Plot saved to: {output_path}")

    if args.val_scatter:
        print("Generating validation scatter plot...")
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        transform = transforms.ToTensor()
        
        # Determine dataset dir
        base = '/home/acolombo/VAEs/dataset'
        ds_map = {'mnist': 'MNIST', 'cifar100': 'CIFAR100', 'emnist': 'EMNIST'}
        data_dir = os.path.join(base, ds_map.get(dataset, 'MNIST'))
        
        if dataset == "mnist":
            val_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        elif dataset == "emnist":
            val_data = datasets.EMNIST(root=data_dir, split="byclass", train=False, download=True, transform=transform)
        elif dataset == "cifar100":
            val_data = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        else:
            val_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
            
        val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Init encoder
        sys.path.insert(0, "/home/acolombo/VAEs/egs/MNIST_VQVAE")
        from mnist_vqvae import Encoder, Decoder
        in_channels = 3 if dataset == "cifar100" else 1
        img_size = 32 if dataset == "cifar100" else 28
        
        encoder = Encoder(D=dim, in_channels=in_channels, img_size=img_size)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k.replace("encoder.", "")] = v
        encoder.load_state_dict(encoder_state)
        encoder.to(device)
        encoder.eval()
        
        # also need decoder if not init
        dec = decoder
        if dec is None:
            dec = Decoder(D=dim, out_channels=in_channels, img_size=img_size)
            decoder_state = {}
            for k, v in state_dict.items():
                if k.startswith("decoder."):
                    decoder_state[k.replace("decoder.", "")] = v
            dec.load_state_dict(decoder_state)
        
        dec.to(device)
        dec.eval()
        
        print("Extracting encoded validation points...")
        all_z = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                z = encoder(imgs) # (B, D, 1)
                z = z.squeeze(-1) # (B, D)
                if c > 0:
                    # Encoder output is in tangent space; map to the ball before plotting.
                    z = project(exp_map0(z, c), c)
                all_z.append(z.cpu())
                all_labels.append(labels.cpu())
                
        all_z = torch.cat(all_z, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        
        # Scatter val points
        cmap_name = 'tab20' if dataset == 'emnist' else 'tab10'
        scatter = ax2.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap=cmap_name, s=10, alpha=0.5)
        legend1 = ax2.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
        ax2.add_artist(legend1)
        
        # Plot origin
        ax2.scatter([0], [0], c="black", marker="x", s=100, label="Origin", zorder=100)
        
        # Plot codebook 0 as images
        cb0 = codebooks[0].numpy()
        dec.cpu()
        with torch.no_grad():
            for pt in cb0:
                pt_tensor = torch.from_numpy(pt).unsqueeze(0).unsqueeze(2) # (1, D, 1)
                decoded_img = dec(pt_tensor).squeeze(0)
                decoded_img_np = decoded_img.numpy()
                plot_image_at_point(ax2, pt[:2], decoded_img_np, args.image_zoom, zorder=20)
                
        if c > 0:
            radius = 1.0 / np.sqrt(c)
            circle = plt.Circle((0, 0), radius, color="green", fill=False, linestyle="-", linewidth=2, label="Poincaré boundary")
            ax2.add_patch(circle)
            bound = radius * 1.1
            ax2.set_xlim(-bound, bound)
            ax2.set_ylim(-bound, bound)
        else:
            bound = float(np.max(np.abs(all_z[:, :2]))) * 1.1 if all_z.size > 0 else 0.5
            ax2.set_xlim(-bound, bound)
            ax2.set_ylim(-bound, bound)
            
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Dim 1")
        ax2.set_ylabel("Dim 2")
        plt.title("Validation Set Encodings and First Codebook Images")
        
        val_output_path = os.path.join(checkpoint_dir, f"{os.path.basename(args.checkpoint).split('.')[0]}_val_scatter.png")
        plt.savefig(val_output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Validation scatter plot saved to: {val_output_path}")


if __name__ == "__main__":
    main()
