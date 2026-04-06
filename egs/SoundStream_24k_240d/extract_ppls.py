import argparse
import sys
import re
import ast
import os
import math
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Extract and plot ppl arrays from a log file.")
    parser.add_argument("log_file", type=str, help="Path to the input log file")
    args = parser.parse_args()

    log_file = args.log_file
    
    # Regexes
    pattern = re.compile(r'ppl:\s*(\[[0-9.,\s+-]+\])')
    train_pattern = re.compile(r'Train PPL \(whole epoch\):\s*(\[[0-9.,\s+-]+\])')
    save_dir_pattern = re.compile(r'save_dir:\s*(.+)')
    
    all_ppls = []
    train_ppls = []
    save_dir = None
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Look for save_dir to know where to save the plot
                if save_dir is None:
                    sd_match = save_dir_pattern.search(line)
                    if sd_match:
                        save_dir = sd_match.group(1).strip()
                
                # Look for ppl values
                match = pattern.search(line)
                train_match = train_pattern.search(line)
                if match:
                    list_str = match.group(1)
                    try:
                        ppls = ast.literal_eval(list_str)
                        all_ppls.append(ppls)
                    except Exception as e:
                        print(f"Warning: could not parse list '{list_str}': {e}", file=sys.stderr)
                if train_match:
                    list_str = train_match.group(1)
                    try:
                        ppls = ast.literal_eval(list_str)
                        train_ppls.append(ppls)
                    except Exception as e:
                        print(f"Warning: could not parse list '{list_str}': {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Could not find log file '{log_file}'", file=sys.stderr)
        sys.exit(1)

    if not all_ppls:
        print("No ppl values found in the log file.")
        sys.exit(0)
                        
    # Optionally save the text output
    # if args.output:
    #     with open(args.output, 'w', encoding='utf-8') as out_f:
    #         out_f.write(f"Extracted {len(all_ppls)} sets of ppls:\n\n")
    #         for i, ppls in enumerate(all_ppls):
    #             out_f.write(f"Match {i+1}: {ppls}\n")
    #     print(f"Successfully saved {len(all_ppls)} sets of ppls to {args.output}")

    # Process and Plot Data
    if not save_dir:
        # Fallback if save_dir was not found in the log
        # Try to extract Job ID from the log file name (e.g. log_21308852.out)
        job_id_match = re.search(r'\_(\d+)\.out$', log_file)
        if job_id_match:
            job_id = job_id_match.group(1)
            save_dir = f"/home/acolombo/VAEs/checkpoint/soundstream/{job_id}"
        else:
            save_dir = "."
            
    os.makedirs(save_dir, exist_ok=True)
    
    # all_ppls is a list of lists. We transpose it to get a list of values for each quantizer index.
    # Assuming all lists have the same length
    num_quantizers = len(all_ppls[0])
    lists_per_q = list(zip(*all_ppls))
    
    cols = min(5, num_quantizers)
    rows = math.ceil(num_quantizers / cols)
    
    # Create the subplots table
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    
    # Ensure axes is always a correctly shaped 1D array even if there's only 1 row or column
    if num_quantizers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i in range(num_quantizers):
        axes[i].plot(lists_per_q[i], color='tab:blue', linewidth=2)
        axes[i].set_title(f"Quantizer {i+1}", fontsize=10, fontweight='bold')
        axes[i].set_xlabel("Logging Steps", fontsize=8)
        axes[i].set_ylabel("PPL", fontsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
    # Remove any completely empty subplots at the end of the grid
    for i in range(num_quantizers, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    
    plot_path1 = os.path.join(save_dir, "ppls_per_codebook.png")
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ----------------------------------------------------
    # Second plot: Codebook vs PPL for every epoch/step
    # ----------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    codebook_indices = list(range(1, num_quantizers + 1))
    
    # Use a colormap to show progression over time (epochs/steps)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(all_ppls) - 1)) for i in range(len(all_ppls))]
    
    for epoch_idx, ppls in enumerate(all_ppls):
        ax2.plot(codebook_indices, ppls, marker='.', markersize=4, alpha=0.6, color=colors[epoch_idx], linewidth=1.5)
        
    # Create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(all_ppls)-1))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2)
    cbar.set_label("Logging Step (Time)", fontsize=10)
    
    ax2.set_xticks(codebook_indices)
    ax2.set_xlabel("Codebook Index", fontsize=10)
    ax2.set_ylabel("PPL", fontsize=10)
    ax2.set_title("Codebook vs PPL across all Logging Steps", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plot_path2 = os.path.join(save_dir, "ppls_per_epoch.png")
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved successfully to:\n - {plot_path1}\n - {plot_path2}")

    # ----------------------------------------------------
    # Third plot: Train PPL vs every epoch/step
    # ----------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    codebook_indices = list(range(1, num_quantizers + 1))
    
    # Use a colormap to show progression over time (epochs/steps)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(train_ppls) - 1)) for i in range(len(train_ppls))]
    
    for epoch_idx, ppls in enumerate(train_ppls):
        ax3.plot(codebook_indices, ppls, marker='.', markersize=4, alpha=0.6, color=colors[epoch_idx], linewidth=1.5)
        
    # Create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(train_ppls)-1))
    sm.set_array([])
    cbar = fig3.colorbar(sm, ax=ax3)
    cbar.set_label("Logging Step (Time)", fontsize=10)
    
    ax3.set_xticks(codebook_indices)
    ax3.set_xlabel("Codebook Index", fontsize=10)
    ax3.set_ylabel("PPL", fontsize=10)
    ax3.set_title("Codebook vs Train PPL across all Logging Steps", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plot_path3 = os.path.join(save_dir, "train_ppls_per_epoch.png")
    plt.savefig(plot_path3, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved successfully to:\n - {plot_path1}\n - {plot_path2}\n - {plot_path3}")
    

if __name__ == "__main__":
    main()
