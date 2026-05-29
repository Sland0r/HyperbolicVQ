import os
import gzip
import json
import torch
import urllib.request
from collections import defaultdict
from torch.utils.data import Dataset

DATASET_DIR = '/home/acolombo/VAEs/dataset/Amazon'
os.makedirs(DATASET_DIR, exist_ok=True)

REVIEWS_URL = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'
META_URL = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz'

REVIEWS_FILE = os.path.join(DATASET_DIR, 'reviews_Beauty_5.json.gz')
META_FILE = os.path.join(DATASET_DIR, 'meta_Beauty.json.gz')

EMBEDDINGS_FILE = os.path.join(DATASET_DIR, 'beauty_embeddings_mpnet.pt')
PROCESSED_FILE = os.path.join(DATASET_DIR, 'beauty_processed.pt')

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print(f"File {filepath} already exists.")

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        try:
            yield eval(l)
        except:
            yield json.loads(l)

def prepare_data(min_history_len=5, seq_len=20):
    if os.path.exists(PROCESSED_FILE) and os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading processed data from {PROCESSED_FILE}")
        data = torch.load(PROCESSED_FILE)
        return data['user_histories'], data['item_catalog'], data['item_to_id'], data['id_to_item'], EMBEDDINGS_FILE

    download_file(REVIEWS_URL, REVIEWS_FILE)
    download_file(META_URL, META_FILE)

    print("Parsing reviews...")
    user_reviews = defaultdict(list)
    item_set = set()
    for review in parse(REVIEWS_FILE):
        user = review['reviewerID']
        item = review['asin']
        timestamp = review['unixReviewTime']
        user_reviews[user].append((timestamp, item))
        item_set.add(item)
    
    print(f"Found {len(user_reviews)} users and {len(item_set)} items in reviews.")

    print("Parsing metadata...")
    item_texts = {}
    for meta in parse(META_FILE):
        asin = meta.get('asin', '')
        if asin in item_set:
            title = meta.get('title', '')
            desc = meta.get('description', '')
            if isinstance(desc, list):
                desc = ' '.join(desc)
            text = f"{title}. {desc}".strip()
            item_texts[asin] = text
            
    # Remove items not in metadata to be safe
    valid_items = set(item_texts.keys())
    
    user_histories = {}
    final_valid_items = set()
    for user, reviews in user_reviews.items():
        # filter to valid items only
        valid_reviews = [r for r in reviews if r[1] in valid_items]
        if len(valid_reviews) >= min_history_len:
            valid_reviews.sort()
            history = [item for _, item in valid_reviews]
            user_histories[user] = history
            final_valid_items.update(history)
            
    item_catalog = sorted(list(final_valid_items))
    item_to_id = {item: idx for idx, item in enumerate(item_catalog)}
    id_to_item = {idx: item for item, idx in item_to_id.items()}
    
    print("Converting histories to IDs (leave-one-out split)...")
    train_sequences = []
    val_sequences = []
    test_sequences = []
    for user, history in user_histories.items():
        history_idx = [item_to_id[item] for item in history]
        # Leave-one-out: last=test, second-to-last=val, rest=train
        # Test: predict last item from all previous
        test_src = history_idx[max(0, len(history_idx) - 1 - seq_len):-1]
        test_tgt = history_idx[-1]
        test_sequences.append((test_src, test_tgt))
        # Validation: predict second-to-last from all before it
        if len(history_idx) >= 3:
            val_src = history_idx[max(0, len(history_idx) - 2 - seq_len):-2]
            val_tgt = history_idx[-2]
            val_sequences.append((val_src, val_tgt))
        # Training: all other transitions (up to second-to-last excluded)
        train_history = history_idx[:-2] if len(history_idx) >= 3 else history_idx[:-1]
        for i in range(1, len(train_history)):
            src = train_history[max(0, i - seq_len):i]
            tgt = train_history[i]
            train_sequences.append((src, tgt))

    user_histories_idx = {
        'train': train_sequences,
        'val': val_sequences,
        'test': test_sequences,
    }
    print(f"Split: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test sequences over {len(item_catalog)} items.")
    
    print("Computing item embeddings using all-mpnet-base-v2...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    texts = [item_texts[item] for item in item_catalog]
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    
    # Save the continuous embeddings
    torch.save(embeddings.cpu(), EMBEDDINGS_FILE)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")
    
    data = {
        'user_histories': user_histories_idx,
        'item_catalog': item_catalog,
        'item_to_id': item_to_id,
        'id_to_item': id_to_item,
    }
    torch.save(data, PROCESSED_FILE)
    print(f"Saved processed data to {PROCESSED_FILE}")

    return user_histories_idx, item_catalog, item_to_id, id_to_item, EMBEDDINGS_FILE

class AmazonSequenceDataset(Dataset):
    def __init__(self, sequences, num_items, seq_len=20):
        self.sequences = sequences
        self.seq_len = seq_len
        self.pad_item = num_items
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        src, target = self.sequences[idx]
        
        if len(src) < self.seq_len:
            src = [self.pad_item] * (self.seq_len - len(src)) + src
        else:
            src = src[-self.seq_len:]
            
        return torch.tensor(src, dtype=torch.long), torch.tensor(target, dtype=torch.long)

if __name__ == '__main__':
    prepare_data()
