"""Paper-faithful WordNet hypernymy dataset for the Hierarchy Modeling experiment.

This is a standalone alternative to ``NLP/wordnet_dataset.py``. The key
difference is the train/test split:

- ``split_mode='closure'`` (default, paper-faithful): train/test pairs are the
  *transitive closure* of the noun hypernymy relation (every ancestor, not just
  immediate parents). We randomly hold out 15% of the closure pairs as the test
  set. This matches the paper (§4.1 / Appendix A.1): "we use the transitive
  closure of the WordNet noun taxonomy ... 743,241 hypernymy relations" and
  "a randomly selected 15% of all hypernymy relations" (following Nickel &
  Kiela [42]). Use this mode to compare directly against the paper's Table 1.

  Caveat: a randomly split closure leaks. A held-out pair (u->v) is usually
  re-derivable by composing train edges (u->w, w->v), so a no-model
  graph-composition baseline already scores high recall. The paper's reported
  numbers are on this leaky split.

- ``split_mode='direct'``: train/test only on *direct* (immediate) hypernym
  edges (the behaviour of the original NLP/ dataset). The transitive closure is
  still built but used ONLY to filter negatives. Removes the composition leak,
  but the numbers are not directly comparable to the paper's Table 1.

In both modes the negative sampler never draws a true ancestor (full closure)
of u, matching the paper's H'(u) = {v : (u,v) not in H} definition.
"""

import nltk
from nltk.corpus import wordnet as wn
import torch
from torch.utils.data import Dataset

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import random


class WordNetHierarchyDataset(Dataset):
    def __init__(self, num_negatives=50, split=None, split_mode='closure', seed=42):
        assert split_mode in ('closure', 'direct'), split_mode
        self.split_mode = split_mode

        # Get all noun synsets
        self.synsets = list(wn.all_synsets('n'))
        self.synset2idx = {s.name(): i for i, s in enumerate(self.synsets)}
        self.idx2synset = {i: s for s, i in self.synset2idx.items()}
        self.vocab_size = len(self.synsets)

        # Build the full transitive closure once. It is the set of train/test
        # pairs in 'closure' mode, and always the negative-sampling filter.
        closure_pairs = []
        direct_pairs = []
        self.u_to_hypernyms = {i: set() for i in range(self.vocab_size)}
        for s in self.synsets:
            u_idx = self.synset2idx[s.name()]
            # Full transitive closure (all ancestors)
            for h in s.closure(lambda x: x.hypernyms()):
                h_idx = self.synset2idx[h.name()]
                self.u_to_hypernyms[u_idx].add(h_idx)
                closure_pairs.append((u_idx, h_idx))
            # Direct (immediate) hypernym edges
            for p in s.hypernyms():
                direct_pairs.append((u_idx, self.synset2idx[p.name()]))

        base_pairs = closure_pairs if split_mode == 'closure' else direct_pairs

        rng = random.Random(seed)
        pairs_shuffled = list(base_pairs)
        rng.shuffle(pairs_shuffled)
        # Paper (Appendix A.1): randomly choose 85% for train, 15% for test.
        split_idx = int(len(pairs_shuffled) * 0.85)
        self.train_pairs = pairs_shuffled[:split_idx]
        self.test_pairs = pairs_shuffled[split_idx:]

        if split == 'train':
            self.hypernymy_pairs = self.train_pairs
        elif split == 'test':
            self.hypernymy_pairs = self.test_pairs
        else:
            self.hypernymy_pairs = base_pairs

        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.hypernymy_pairs)

    def __getitem__(self, idx):
        u, v = self.hypernymy_pairs[idx]

        # Vectorized rejection sampling: draw candidates in bulk with torch
        # (C-level, ~free) and filter out true ancestors (and u itself) against
        # the small forbidden set. Ancestors are tiny relative to vocab_size, so
        # a single oversized draw almost always yields enough negatives in one
        # pass. Much cheaper than the previous per-sample Python random.randint
        # loop, which was the data-loading bottleneck.
        forbidden = self.u_to_hypernyms[u]
        need = self.num_negatives
        negatives = []
        while len(negatives) < need:
            for neg in torch.randint(0, self.vocab_size, (2 * need,)).tolist():
                if neg != u and neg not in forbidden:
                    negatives.append(neg)
                    if len(negatives) == need:
                        break

        return {
            'u': torch.tensor(u, dtype=torch.long),
            'v': torch.tensor(v, dtype=torch.long),
            'negatives': torch.tensor(negatives, dtype=torch.long),
        }


if __name__ == '__main__':
    for mode in ('closure', 'direct'):
        ds = WordNetHierarchyDataset(split_mode=mode)
        print(f"[{mode}] vocab={ds.vocab_size}, pairs={len(ds)}, "
              f"train={len(ds.train_pairs)}, test={len(ds.test_pairs)}")
