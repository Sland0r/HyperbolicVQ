"""Representation-quality eval for NLP_2 HRQ checkpoints.

For each model: extract the per-concept code tuples, then measure
  (1) code uniqueness  — how many distinct code sequences are used, and
  (2) intra-cluster word similarity — for concepts that SHARE a code tuple,
      how semantically similar are they (WordNet path/Wu-Palmer + GloVe cosine).

A good representation has LOW uniqueness collapse (codes reused, real clusters)
AND HIGH intra-cluster similarity (a shared code groups related words).
"""
import os, sys, ast, random
from itertools import combinations
from collections import Counter

import numpy as np
import torch
import nltk
from nltk.corpus import wordnet as wn
import gensim.downloader as api

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from NLP_2.train_paper import HRQModel
from NLP_2.dataset_paper import WordNetHierarchyDataset

CKPT_DIR = "/home/acolombo/VAEs/checkpoint/nlp_2"
MODELS = [
    ("Euclidean",        "23711830_euccw1_s-1p0"),
    ("Vanilla hyp",      "23711047_hypcw1_s-1p0"),
    ("A4 + gc",          "23710509_a4cw1_s-1p0"),
]
FLAG_KEYS = ['embed_dim', 'n_q', 'bins', 'c', 'new_method', 'hste',
             'hste_riemannian', 'approx', 'commitment_weight',
             'gradient_correction', 'embed_init_scale', 'block_hste_pt',
             'hste_clip', 'A5', 'A4_v2', 'a6', 'a7', 'a6_1', 'a7_1']


def read_cfg(run_dir):
    cfg = {}
    with open(os.path.join(CKPT_DIR, run_dir, "config.py")) as f:
        for line in f:
            if '=' not in line or line.strip().startswith('#'):
                continue
            k, _, v = line.partition('=')
            k = k.strip()
            if k in FLAG_KEYS:
                try:
                    cfg[k] = ast.literal_eval(v.strip())
                except Exception:
                    pass
    return cfg


def get_embedding(name, emb):
    word = name.split('.')[0]
    vecs = [emb[w] for w in word.split('_') if w in emb]
    return np.mean(np.array(vecs), axis=0) if vecs else None


@torch.no_grad()
def extract_codes(cfg, run_dir, dataset, device):
    mk = {k: cfg[k] for k in FLAG_KEYS if k in cfg and k not in ('embed_dim', 'n_q', 'bins', 'c')}
    model = HRQModel(vocab_size=dataset.vocab_size, embed_dim=cfg['embed_dim'],
                     n_q=cfg['n_q'], bins=cfg['bins'], c=cfg['c'], **mk).to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, run_dir, "model.pt"),
                                     map_location=device))
    model.eval()
    out = []
    idx = torch.arange(dataset.vocab_size, device=device)
    for i in range(0, dataset.vocab_size, 1024):
        b = idx[i:i + 1024]
        _, _, codes, _, _ = model(b)
        if codes.size(0) != b.size(0):
            codes = codes.transpose(0, 1)
        out.append(codes.cpu())
    return torch.cat(out, 0)  # (vocab, n_q)


def analyse(codes, dataset, emb, rng):
    vocab = codes.shape[0]
    code_to_idx = {}
    for i, c in enumerate(codes):
        code_to_idx.setdefault(tuple(c.view(-1).tolist()), []).append(i)

    n_unique = len(code_to_idx)
    clusters = {k: v for k, v in code_to_idx.items() if len(v) > 1}
    sizes = [len(v) for v in code_to_idx.values()]

    tp = tw = te = 0.0
    np_ = ne = 0
    for members in clusters.values():
        names = [dataset.idx2synset[i] for i in members]
        if len(names) > 50:
            names = rng.sample(names, 50)
        for a, b in combinations(names, 2):
            s1, s2 = wn.synset(a), wn.synset(b)
            sp, sw = s1.path_similarity(s2), s1.wup_similarity(s2)
            if sp is not None and sw is not None:
                tp += sp; tw += sw; np_ += 1
            v1, v2 = get_embedding(a, emb), get_embedding(b, emb)
            if v1 is not None and v2 is not None:
                te += float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
                ne += 1
    return {
        'vocab': vocab,
        'n_unique': n_unique,
        'uniq_ratio': n_unique / vocab,
        'n_clusters': len(clusters),
        'mean_size': float(np.mean(sizes)),
        'max_size': int(np.max(sizes)),
        'path_sim': tp / max(1, np_),
        'wup_sim': tw / max(1, np_),
        'emb_sim': te / max(1, ne),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading GloVe (glove-wiki-gigaword-50)...")
    emb = api.load('glove-wiki-gigaword-50')
    dataset = WordNetHierarchyDataset(num_negatives=1, split_mode='closure')
    print(f"vocab_size={dataset.vocab_size}\n")
    rng = random.Random(42)

    rows = []
    for name, run in MODELS:
        cfg = read_cfg(run)
        print(f"== {name}  ({run})  c={cfg.get('c')} dim={cfg.get('embed_dim')} "
              f"bins={cfg.get('bins')} ==")
        codes = extract_codes(cfg, run, dataset, device)
        r = analyse(codes, dataset, emb, rng)
        r['name'] = name
        rows.append(r)
        print(f"   unique={r['n_unique']}/{r['vocab']} ({r['uniq_ratio']:.3f})  "
              f"clusters>1={r['n_clusters']}  mean_sz={r['mean_size']:.2f} "
              f"max_sz={r['max_size']}  path={r['path_sim']:.4f} "
              f"wup={r['wup_sim']:.4f} emb={r['emb_sim']:.4f}\n")

    print("\n" + "=" * 92)
    print(f"{'Method':<14}{'uniq ratio':>12}{'clusters>1':>12}{'mean sz':>9}"
          f"{'path sim':>10}{'wup sim':>10}{'emb sim':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:<14}{r['uniq_ratio']:>12.3f}{r['n_clusters']:>12}"
              f"{r['mean_size']:>9.2f}{r['path_sim']:>10.4f}{r['wup_sim']:>10.4f}"
              f"{r['emb_sim']:>10.4f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
