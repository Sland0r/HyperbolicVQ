import re
from pathlib import Path

content = Path("/home/acolombo/VAEs/checkpoint/mnist_vqvae/21855395/logs/log.txt").read_text()

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
    dp_vectors[epoch]['lt'] = vec

best_epoch_pattern = re.compile(r"<epoch:(\d+),[^\n]*?best_epoch:(\d+)>")
best_epoch_matches = best_epoch_pattern.findall(content)
best_epoch = int(best_epoch_matches[-1][1]) if best_epoch_matches else None
last_epoch = int(best_epoch_matches[-1][0]) if best_epoch_matches else None

if best_epoch and last_epoch:
    print(f"Best epoch: {best_epoch}")
    print(f"Last epoch: {last_epoch}")
    
    if last_epoch in dp_vectors:
        gt = dp_vectors[last_epoch]['gt']
        lt = dp_vectors[last_epoch]['lt']
        diffs = [abs(g - l) for g, l in zip(gt, lt)]
        avg_diff_last = sum(diffs) / len(diffs)
        print(f"Last epoch gt: {gt}")
        print(f"Last epoch lt: {lt}")
        print(f"Avg diff last: {avg_diff_last}")

    if best_epoch in dp_vectors:
        gt = dp_vectors[best_epoch]['gt']
        lt = dp_vectors[best_epoch]['lt']
        diffs = [abs(g - l) for g, l in zip(gt, lt)]
        avg_diff_best = sum(diffs) / len(diffs)
        print(f"Best epoch gt: {gt}")
        print(f"Best epoch lt: {lt}")
        print(f"Avg diff best: {avg_diff_best}")

