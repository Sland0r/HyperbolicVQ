#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Extract and rank rec_loss from checkpoint logs")
parser.add_argument("--folder", type=str, default="new",
                    help="Path to the checkpoint directory (default: checkpoint/new)")
args = parser.parse_args()

folder = "/home/acolombo/VAEs/checkpoint/" + args.folder
checkpoints_dir = Path(folder)

# Get list of checkpoint directories
checkpoints = sorted([d.name for d in checkpoints_dir.iterdir() if d.is_dir()])

# Dictionary to store results
results = {}

train_pattern = re.compile(r"<epoch:(\d+),[^\n]*?recon_loss_train:\s*([+-]?\d+(?:\.\d+)?)")
valid_pattern = re.compile(r"<epoch:(\d+),[^\n]*?recon_loss_valid:\s*([+-]?\d+(?:\.\d+)?)")

for checkpoint in checkpoints:
    # Log file is inside the checkpoint directory
    log_file = checkpoints_dir / checkpoint / "logs" / "log.txt"
    
    if not log_file.exists():
        print(f"WARNING: No log file found for checkpoint: {checkpoint}")
        continue
    
    # Read the log file and extract last epoch rec_loss values
    with open(log_file, 'r') as f:
        content = f.read()
    
    train_matches = train_pattern.findall(content)
    valid_matches = valid_pattern.findall(content)
    
    if train_matches and valid_matches:
        # Get the last epoch values
        last_epoch_train = int(train_matches[-1][0])
        last_rec_loss_train = float(train_matches[-1][1])
        
        last_epoch_valid = int(valid_matches[-1][0])
        last_rec_loss_valid = float(valid_matches[-1][1])
        
        dp_gt_pattern = re.compile(r"\[Epoch (\d+)\] Validation Dot Products > 0: \[([^\]]+)\]")
        dp_lt_pattern = re.compile(r"\[Epoch (\d+)\] Validation Dot Products < 0: \[([^\]]+)\]")
        
        dp_vectors = {}
        for g in dp_gt_pattern.findall(content):
            epoch = int(g[0])
            vec = [float(x.strip().replace('%', '')) for x in g[1].split(',')]
            if epoch not in dp_vectors:
                dp_vectors[epoch] = {}
            dp_vectors[epoch]['gt'] = vec

        for g in dp_lt_pattern.findall(content):
            epoch = int(g[0])
            vec = [float(x.strip().replace('%', '')) for x in g[1].split(',')]
            if epoch not in dp_vectors:
                dp_vectors[epoch] = {}
            dp_vectors[epoch]['lt'] = vec
            
        best_epoch_pattern = re.compile(r"<epoch:(\d+),[^\n]*?best_epoch:(\d+)>")
        best_epoch_matches = best_epoch_pattern.findall(content)
        best_epoch = int(best_epoch_matches[-1][1]) if best_epoch_matches else last_epoch_valid
        
        avg_diff_last = None
        last_gt = None
        last_lt = None
        avg_diff_best = None
        best_gt = None
        best_lt = None
        
        if last_epoch_valid in dp_vectors and 'gt' in dp_vectors[last_epoch_valid] and 'lt' in dp_vectors[last_epoch_valid]:
            last_gt = dp_vectors[last_epoch_valid]['gt']
            last_lt = dp_vectors[last_epoch_valid]['lt']
            diffs = [abs(g - l) for g, l in zip(last_gt, last_lt)]
            avg_diff_last = sum(diffs) / len(diffs) if diffs else 0.0

        if best_epoch in dp_vectors and 'gt' in dp_vectors[best_epoch] and 'lt' in dp_vectors[best_epoch]:
            best_gt = dp_vectors[best_epoch]['gt']
            best_lt = dp_vectors[best_epoch]['lt']
            diffs = [abs(g - l) for g, l in zip(best_gt, best_lt)]
            avg_diff_best = sum(diffs) / len(diffs) if diffs else 0.0
        
        results[checkpoint] = {
            'train_rec_loss': last_rec_loss_train,
            'valid_rec_loss': last_rec_loss_valid,
            'epoch_train': last_epoch_train,
            'epoch_valid': last_epoch_valid,
            'avg_diff_last': avg_diff_last,
            'last_gt': last_gt,
            'last_lt': last_lt,
            'avg_diff_best': avg_diff_best,
            'best_gt': best_gt,
            'best_lt': best_lt,
            'best_epoch': best_epoch
        }
    else:
        print(f"WARNING: Could not extract rec_loss from {checkpoint}")

# Create rankings
if results:
    print("\n" + "="*70)
    print("TRAINING REC_LOSS RANKING (Best to Worst)")
    print("="*70)
    
    train_sorted = sorted(results.items(), key=lambda x: x[1]['train_rec_loss'])
    for rank, (checkpoint, data) in enumerate(train_sorted, 1):
        print(f"{rank:2d}. {checkpoint:40s} -> {data['train_rec_loss']:8.4f} (Epoch {data['epoch_train']})")
    
    print("\n" + "="*70)
    print("VALIDATION REC_LOSS RANKING (Best to Worst)")
    print("="*70)
    
    valid_sorted = sorted(results.items(), key=lambda x: x[1]['valid_rec_loss'])
    for rank, (checkpoint, data) in enumerate(valid_sorted, 1):
        print(f"{rank:2d}. {checkpoint:40s} -> {data['valid_rec_loss']:8.4f} (Epoch {data['epoch_valid']})")
    
    last_diff_results = {k: v for k, v in results.items() if v['avg_diff_last'] is not None}
    if last_diff_results:
        print("\n" + "="*70)
        print("VALIDATION DP DIFFERENCE (LAST EPOCH) RANKING (Smallest to Largest)")
        print("="*70)
        
        last_diff_sorted = sorted(last_diff_results.items(), key=lambda x: x[1]['avg_diff_last'])
        for rank, (checkpoint, data) in enumerate(last_diff_sorted, 1):
            gt_str = "[" + ", ".join(f"{x:.3f}%" for x in data['last_gt']) + "]"
            lt_str = "[" + ", ".join(f"{x:.3f}%" for x in data['last_lt']) + "]"
            print(f"{rank:2d}. {checkpoint:40s} -> Diff: {data['avg_diff_last']:8.4f}% (Epoch {data['epoch_valid']})")
            print(f"    > 0: {gt_str}")
            print(f"    < 0: {lt_str}")
            
    best_diff_results = {k: v for k, v in results.items() if v['avg_diff_best'] is not None}
    if best_diff_results:
        print("\n" + "="*70)
        print("VALIDATION DP DIFFERENCE (BEST EPOCH) RANKING (Smallest to Largest)")
        print("="*70)
        
        best_diff_sorted = sorted(best_diff_results.items(), key=lambda x: x[1]['avg_diff_best'])
        for rank, (checkpoint, data) in enumerate(best_diff_sorted, 1):
            gt_str = "[" + ", ".join(f"{x:.3f}%" for x in data['best_gt']) + "]"
            lt_str = "[" + ", ".join(f"{x:.3f}%" for x in data['best_lt']) + "]"
            print(f"{rank:2d}. {checkpoint:40s} -> Diff: {data['avg_diff_best']:8.4f}% (Epoch {data['best_epoch']})")
            print(f"    > 0: {gt_str}")
            print(f"    < 0: {lt_str}")
    
    print("\n" + "="*70)
    print(f"Total checkpoints processed: {len(results)}")
    print("="*70)
else:
    print("No results found!")
