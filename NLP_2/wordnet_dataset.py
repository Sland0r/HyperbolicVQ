import nltk
from nltk.corpus import wordnet as wn
import torch
from torch.utils.data import Dataset, DataLoader
import random

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class WordNetHierarchyDataset(Dataset):
    def __init__(self, num_negatives=50, split=None):
        # Get all noun synsets
        self.synsets = list(wn.all_synsets('n'))
        self.synset2idx = {s.name(): i for i, s in enumerate(self.synsets)}
        self.idx2synset = {i: s for s, i in self.synset2idx.items()}
        self.vocab_size = len(self.synsets)

        # We split on DIRECT hypernym edges (immediate parents), not on the
        # transitive closure. Randomly splitting the closure leaks: a held-out
        # pair (u->v) is usually re-derivable by composing train edges
        # (u->w, w->v), so ~84% of the test set becomes trivially inferable and
        # a no-model graph-composition baseline already scores ~80% recall@10.
        # Splitting direct edges removes that composition leak.
        #
        # The full transitive closure is still built, but used ONLY to filter
        # negative samples (never sample a true ancestor as a negative), which
        # matches the paper's H'(u) = {v : (u,v) not in H} definition.
        direct_pairs = []
        self.u_to_hypernyms = {i: set() for i in range(self.vocab_size)}
        for s in self.synsets:
            u_idx = self.synset2idx[s.name()]
            # Full transitive closure -> negative-sample filter only
            for h in s.closure(lambda x: x.hypernyms()):
                self.u_to_hypernyms[u_idx].add(self.synset2idx[h.name()])
            # Direct (immediate) hypernym edges -> the pairs we train/test on
            for p in s.hypernyms():
                direct_pairs.append((u_idx, self.synset2idx[p.name()]))

        rng = random.Random(42)
        direct_pairs_shuffled = list(direct_pairs)
        rng.shuffle(direct_pairs_shuffled)
        split_idx = int(len(direct_pairs_shuffled) * 0.85)
        self.train_pairs = direct_pairs_shuffled[:split_idx]
        self.test_pairs = direct_pairs_shuffled[split_idx:]

        if split == 'train':
            self.hypernymy_pairs = self.train_pairs
        elif split == 'test':
            self.hypernymy_pairs = self.test_pairs
        else:
            self.hypernymy_pairs = direct_pairs

        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.hypernymy_pairs)

    def __getitem__(self, idx):
        u, v = self.hypernymy_pairs[idx]
        
        # Sample negative hypernyms
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.vocab_size - 1)
            # Ensure neg is not a hypernym of u, and not u itself
            if neg not in self.u_to_hypernyms[u] and neg != u:
                negatives.append(neg)
                
        return {
            'u': torch.tensor(u, dtype=torch.long),
            'v': torch.tensor(v, dtype=torch.long),
            'negatives': torch.tensor(negatives, dtype=torch.long)
        }

if __name__ == '__main__':
    dataset = WordNetHierarchyDataset()
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Pairs: {len(dataset)}")
