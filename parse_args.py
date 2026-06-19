import os
import glob
import re

directories = [
    "/home/acolombo/VAEs/checkpoint/mnist_new",
    "/home/acolombo/VAEs/checkpoint/mnist",
    "/home/acolombo/VAEs/checkpoint/mnist_vqvae",
    "/home/acolombo/VAEs/checkpoint/cifar_new",
    "/home/acolombo/VAEs/checkpoint/cifar100",
    "/home/acolombo/VAEs/checkpoint/cifar100_100",
    "/home/acolombo/VAEs/checkpoint/cifar100_vqvae",
]

def parse_args(filepath):
    args = {}
    if not os.path.exists(filepath): return args
    with open(filepath, 'r') as f:
        lines = f.readlines()
        in_args = False
        for line in lines:
            if line.startswith("==> args:"):
                in_args = True
                continue
            if in_args:
                m = re.match(r"\s+([a-zA-Z0-9_]+):\s+(.*)", line)
                if m:
                    key, val = m.groups()
                    # ignore structural/logging/dataset arguments
                    if key not in ['PATH', 'save_dir', 'data_dir', 'dataset', 'global_step', 'st_epoch', 'resume_path', 'emnist_split', 'print_freq', 'cudnn_deterministic']:
                        args[key] = val
    return args

def parse_log(filepath):
    res = {}
    if not os.path.exists(filepath): return res
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            m = re.search(r"Best model\s+\(epoch (\d+)\):\s+val_rec=([0-9.]+)", line)
            if m:
                res['epoch'] = m.group(1)
                res['val_rec'] = float(m.group(2))
                return res
    return res

results = []

for d in directories:
    if not os.path.exists(d): continue
    for root, dirs, files in os.walk(d):
        if 'configs' in dirs and 'logs' in dirs:
            args_file = os.path.join(root, 'configs', 'args.txt')
            log_file = os.path.join(root, 'logs', 'log.txt')
            if not os.path.exists(args_file):
                continue
            
            a = parse_args(args_file)
            l = parse_log(log_file)
            if a and l:
                # determine dataset from root
                ds = 'cifar' if 'cifar' in root else 'mnist'
                
                results.append({
                    'path': root.replace('/home/acolombo/VAEs/checkpoint/', ''),
                    'dataset': ds,
                    'args': a,
                    'val_rec': l['val_rec'],
                    'epoch': l['epoch']
                })

# The user wants to compare MNIST OR CIFAR. 
# So we have two target groups:
# 1. mnist_new/c1_blockpt
# 2. cifar_new/c1_blockpt

target_mnist_args = None
target_cifar_args = None

for r in results:
    if 'mnist_new/c1_blockpt' in r['path']:
        target_mnist_args = r['args']
    if 'cifar_new/c1_blockpt' in r['path']:
        target_cifar_args = r['args']

def compare_args(a1, a2):
    # Only compare keys that exist in both or are relevant to the model logic
    # Important hyperparameters:
    keys = ['D', 'n_q', 'bins', 'codebook_weight', 'commitment_weight', 'lr_g', 'lr_manifold', 'N_EPOCHS', 'BATCH_SIZE', 'c', 'approx']
    for k in keys:
        if a1.get(k) != a2.get(k):
            return False
    return True

print(f"Total experiments found: {len(results)}")

print("\n--- MNIST Comparable Experiments ---")
if target_mnist_args:
    for r in sorted(results, key=lambda x: x['val_rec']):
        if r['dataset'] == 'mnist' and compare_args(r['args'], target_mnist_args):
            print(f"[{r['dataset'].upper()}] {r['path']:<35} | val_rec: {r['val_rec']:.5f} | epoch: {r['epoch']}")
else:
    print("Could not find mnist_new/c1_blockpt target")

print("\n--- CIFAR Comparable Experiments ---")
if target_cifar_args:
    for r in sorted(results, key=lambda x: x['val_rec']):
        if r['dataset'] == 'cifar' and compare_args(r['args'], target_cifar_args):
            print(f"[{r['dataset'].upper()}] {r['path']:<35} | val_rec: {r['val_rec']:.5f} | epoch: {r['epoch']}")
else:
    print("Could not find cifar_new/c1_blockpt target")
