import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import combinations
from collections import Counter
import nltk
from nltk.corpus import wordnet as wn
import random
import gensim.downloader as api
import numpy as np

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NLP.train_hierarchy import HRQModel, WordNetHierarchyDataset

def get_embedding(name, embed_model):
    word = name.split('.')[0]
    parts = word.split('_')
    vecs = [embed_model[w] for w in parts if w in embed_model]
    if not vecs:
        return None
    vecs = np.array(vecs)
    return np.mean(vecs, axis=0)

def compute_cluster_similarities(model, dataset, device, embed_model, max_clusters=5000):
    model.eval()
    
    # Extract all codes
    all_codes = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, dataset.vocab_size, batch_size):
            end = min(i + batch_size, dataset.vocab_size)
            idx_tensor = torch.arange(i, end, dtype=torch.long).to(device)
            _, _, codes, _, _ = model(idx_tensor)
            # RVQ returns codes of shape [n_q, batch_size], transpose to [batch_size, n_q]
            all_codes.append(codes.transpose(0, 1).cpu())
            
    all_codes = torch.cat(all_codes, dim=0) # [vocab_size, n_q]
    
    # Group by code tuple
    code_to_synsets = {}
    for idx, c in enumerate(all_codes):
        ctup = tuple(c.view(-1).tolist())
        if ctup not in code_to_synsets:
            code_to_synsets[ctup] = []
        code_to_synsets[ctup].append(idx)
        
    print(f"  Total unique codes: {len(code_to_synsets)}")
    
    # Filter for clusters with more than 1 item
    valid_clusters = {k: v for k, v in code_to_synsets.items() if len(v) > 1}
    print(f"  Clusters with > 1 items: {len(valid_clusters)}")
    
    # Evaluate similarities
    clusters_to_eval = list(valid_clusters.values())

    total_path_sim = 0.0
    total_wup_sim = 0.0
    total_emb_sim = 0.0
    pair_count = 0
    emb_pair_count = 0
    
    size_emb_sim_total = {}
    size_emb_sim_count = {}
    
    rng = random.Random(42)
    for cluster_idxs in tqdm(clusters_to_eval, desc="  Computing similarities"):
        orig_size = len(cluster_idxs)
        if orig_size not in size_emb_sim_total:
            size_emb_sim_total[orig_size] = 0.0
            size_emb_sim_count[orig_size] = 0

        synsets = [dataset.idx2synset[idx] for idx in cluster_idxs]
        
        # Limit to avoid O(N^2) on gigantic clusters
        if len(synsets) > 50:
            synsets = rng.sample(synsets, 50)
            
        for s1_name, s2_name in combinations(synsets, 2):
            s1 = wn.synset(s1_name)
            s2 = wn.synset(s2_name)
            sim_path = s1.path_similarity(s2)
            sim_wup = s1.wup_similarity(s2)
            
            if sim_path is not None and sim_wup is not None:
                total_path_sim += sim_path
                total_wup_sim += sim_wup
                pair_count += 1
                
            v1 = get_embedding(s1_name, embed_model)
            v2 = get_embedding(s2_name, embed_model)
            if v1 is not None and v2 is not None:
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                total_emb_sim += cos_sim
                emb_pair_count += 1
                size_emb_sim_total[orig_size] += cos_sim
                size_emb_sim_count[orig_size] += 1
                
    avg_path_sim = total_path_sim / max(1, pair_count)
    avg_wup_sim = total_wup_sim / max(1, pair_count)
    avg_emb_sim = total_emb_sim / max(1, emb_pair_count)
    avg_emb_by_size = {sz: (size_emb_sim_total[sz] / size_emb_sim_count[sz] if size_emb_sim_count[sz] > 0 else 0) 
                       for sz in size_emb_sim_total}
    
    # Calculate size distribution statistics
    cluster_sizes = [len(v) for v in code_to_synsets.values()]
    mean_size = np.mean(cluster_sizes)
    var_size = np.var(cluster_sizes)
    size_counts = dict(Counter(cluster_sizes))
    
    stats = {
        'num_valid_clusters': len(valid_clusters),
        'mean_size': mean_size,
        'var_size': var_size,
        'size_counts': size_counts,
        'avg_emb_by_size': avg_emb_by_size
    }
    
    return avg_path_sim, avg_wup_sim, avg_emb_sim, stats

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading GloVe embeddings (glove-wiki-gigaword-50)...")
    embed_model = api.load('glove-wiki-gigaword-50')
    
    # Initialize dataset
    dataset = WordNetHierarchyDataset(num_negatives=1)
    
    print("\nEvaluating...")
    model_euc = HRQModel(vocab_size=dataset.vocab_size, embed_dim=16, n_q=4, bins=128, c=1.0).to(device)
    model_euc.load_state_dict(torch.load("/home/acolombo/VAEs/checkpoint/nlp/new/model.pt", map_location=device))
    avg_path_euc, avg_wup_euc, avg_emb_euc, stats_euc = compute_cluster_similarities(model_euc, dataset, device, embed_model)
    
    def print_results(name, avg_path, avg_wup, avg_emb, stats):
        print(f"\n{name}:")
        print(f"  Valid clusters (>1 item) : {stats['num_valid_clusters']}")
        print(f"  Mean cluster size        : {stats['mean_size']:.4f}")
        print(f"  Variance of cluster size : {stats['var_size']:.4f}")
        print(f"  Average Path Similarity  : {avg_path:.4f}")
        print(f"  Average WUP Similarity   : {avg_wup:.4f}")
        print(f"  Average Embed Similarity : {avg_emb:.4f}")
        
        # Sort counts by cluster size
        sorted_counts = sorted(stats['size_counts'].items())
        print(f"  Size Distribution & Embed Sim:")
        for k, count in sorted_counts:
            sim_str = f", Avg Emb Sim: {stats['avg_emb_by_size'][k]:.4f}" if k in stats.get('avg_emb_by_size', {}) else ""
            print(f"    Size {k}: {count} clusters{sim_str}")

    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print_results("Results:", avg_path_euc, avg_wup_euc, avg_emb_euc, stats_euc)
    print("="*80)

if __name__ == "__main__":
    main()
