import nltk
from nltk.corpus import wordnet as wn
import torch
from torch.utils.data import Dataset, DataLoader
import random

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class WordNetHierarchyDataset(Dataset):
    def __init__(self, num_negatives=50):
        # Get all noun synsets
        self.synsets = list(wn.all_synsets('n'))
        self.synset2idx = {s.name(): i for i, s in enumerate(self.synsets)}
        self.idx2synset = {i: s for s, i in self.synset2idx.items()}
        self.vocab_size = len(self.synsets)
        
        self.hypernymy_pairs = []
        self.u_to_hypernyms = {i: set() for i in range(self.vocab_size)}
        for s in self.synsets:
            # Transitive closure of hypernyms
            hypernyms = list(s.closure(lambda x: x.hypernyms()))
            u_idx = self.synset2idx[s.name()]
            for h in hypernyms:
                v_idx = self.synset2idx[h.name()]
                self.hypernymy_pairs.append((u_idx, v_idx))
                self.u_to_hypernyms[u_idx].add(v_idx)
                
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
