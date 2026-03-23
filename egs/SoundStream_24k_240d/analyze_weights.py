import argparse
import sys
import torch

def analyze_weights(model_path):
    print(f"Loading checkpoint from: {model_path}")
    try:
        # Load the checkpoint to CPU to avoid strict GPU requirements during analysis
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)
        
    if 'soundstream' in checkpoint:
        print("Detected 'latest.pth' format containing optimizers, epochs, etc.")
        print("\n=== SoundStream Codebook Weights ===")
        analyze_state_dict(checkpoint['soundstream'])
    else:
        print("Detected 'best_*.pth' format containing only model state_dict.")
        print("\n=== Model Codebook Weights ===")
        analyze_state_dict(checkpoint)


def analyze_state_dict(state_dict):
    print(f"{'Layer Name':<65} | {'Shape':<20} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'NaNs':>8} | {'Infs':>8}")
    print("-" * 165)
    
    for name, tensor in state_dict.items():
        if "codebook" not in name.lower() and "embed" not in name.lower():
            continue
            
        if tensor.is_floating_point():
            nans = torch.isnan(tensor).sum().item()
            infs = torch.isinf(tensor).sum().item()
            
            valid_mask = ~(torch.isnan(tensor) | torch.isinf(tensor))
            valid_tensor = tensor[valid_mask]
            
            if valid_tensor.numel() > 0:
                mean = valid_tensor.mean().item()
                std = valid_tensor.std().item() if valid_tensor.numel() > 1 else 0.0
                min_val = valid_tensor.min().item()
                max_val = valid_tensor.max().item()
            else:
                mean, std, min_val, max_val = float('nan'), float('nan'), float('nan'), float('nan')
                
            shape_str = str(list(tensor.shape))
            
            print(f"{name:<65} | {shape_str:<20} | {mean:10.4f} | {std:10.4f} | {min_val:10.4f} | {max_val:10.4f} | {nans:8d} | {infs:8d}")
            
            print(f"  -> DETAILED CODEBOOK INFO for {name}:")
            if tensor.dim() >= 2:
                snippet = tensor.flatten(0, -2)[:3] # Grab first 3 "entries"
                print(f"     First 3 entries snippet:\n{snippet}")
                
                norms = tensor.norm(dim=-1)
                print(f"     Norm stats: Mean={norms.mean().item():.6f}, Std={norms.std().item() if norms.numel() > 1 else 0.0:.6f}, Min={norms.min().item():.6f}, Max={norms.max().item():.6f}")

    print("-" * 165)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze weights of a saved PyTorch model.")
    parser.add_argument("--path", type=str, help="Path to the .pth model file")
    args = parser.parse_args()
    
    analyze_weights(args.path)
