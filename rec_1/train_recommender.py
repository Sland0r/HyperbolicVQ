import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import os
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rec_1.amazon_dataset import prepare_data, AmazonSequenceDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class RecommenderT5(nn.Module):
    def __init__(self, bins=256, d_model=384, nhead=6, num_layers=6, n_q=4,
                 dropout=0.1, dim_feedforward=1024):
        super().__init__()
        self.n_q = n_q
        self.bins = bins
        self.d_model = d_model
        self.pad_token = bins * n_q
        self.bos_token = bins * n_q + 1
        vocab_size = bins * n_q + 2

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_token)
        self.enc_pos = PositionalEncoding(d_model, dropout=dropout)
        self.dec_pos = PositionalEncoding(d_model, dropout=dropout, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.ModuleList([nn.Linear(d_model, bins) for _ in range(n_q)])

    def encode(self, src_tokens, src_padding_mask=None):
        emb = self.enc_pos(self.token_embedding(src_tokens))
        return self.encoder(emb, src_key_padding_mask=src_padding_mask)

    def _decode(self, tgt_offset_codes, memory, memory_padding_mask=None):
        bos = torch.full((tgt_offset_codes.size(0), 1), self.bos_token,
                         device=tgt_offset_codes.device, dtype=torch.long)
        dec_input = torch.cat([bos, tgt_offset_codes[:, :-1]], dim=1)

        dec_emb = self.dec_pos(self.token_embedding(dec_input))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.n_q, device=dec_input.device)

        out = self.decoder(dec_emb, memory, tgt_mask=tgt_mask,
                           memory_key_padding_mask=memory_padding_mask)
        return [self.fc_out[q](out[:, q, :]) for q in range(self.n_q)]

    def forward(self, src_tokens, src_padding_mask, tgt_offset_codes):
        memory = self.encode(src_tokens, src_padding_mask)
        return self._decode(tgt_offset_codes, memory, src_padding_mask)


def offset_codes(raw_codes, bins):
    n_q = raw_codes.size(-1)
    offsets = torch.arange(n_q, device=raw_codes.device) * bins
    return raw_codes + offsets


def flatten_history(src_idx, item_offset_codes_padded, pad_token):
    codes = item_offset_codes_padded[src_idx]
    flat = codes.reshape(codes.size(0), -1)
    padding_mask = (flat == pad_token)
    return flat, padding_mask


def calculate_metrics(predictions, targets, k):
    batch_size = targets.size(0)
    _, top_k_indices = torch.topk(predictions, k, dim=-1)

    recall = 0.0
    ndcg = 0.0
    for i in range(batch_size):
        if targets[i] in top_k_indices[i]:
            recall += 1.0
            rank = (top_k_indices[i] == targets[i]).nonzero(as_tuple=True)[0].item()
            ndcg += 1.0 / math.log2(rank + 2)

    return recall / batch_size, ndcg / batch_size


def score_all_items(model, memory, enc_mask, item_offset_codes, item_raw_codes,
                    n_q, device, chunk_size=2048):
    B = memory.size(0)
    num_items = item_raw_codes.size(0)
    scores = torch.zeros(B, num_items, device=device)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(n_q, device=device)

    for b in range(B):
        mem_b = memory[b:b+1]
        mask_b = enc_mask[b:b+1] if enc_mask is not None else None

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            n_c = end - start

            chunk_offset = item_offset_codes[start:end]
            bos_col = torch.full((n_c, 1), model.bos_token, device=device, dtype=torch.long)
            dec_input = torch.cat([bos_col, chunk_offset[:, :-1]], dim=1)

            dec_emb = model.dec_pos(model.token_embedding(dec_input))
            mem_exp = mem_b.expand(n_c, -1, -1)
            mask_exp = mask_b.expand(n_c, -1) if mask_b is not None else None

            out = model.decoder(dec_emb, mem_exp, tgt_mask=tgt_mask,
                                memory_key_padding_mask=mask_exp)

            chunk_scores = torch.zeros(n_c, device=device)
            for q in range(n_q):
                log_probs = torch.log_softmax(model.fc_out[q](out[:, q, :]), dim=-1)
                chunk_scores = chunk_scores + log_probs[
                    torch.arange(n_c, device=device), item_raw_codes[start:end, q]]

            scores[b, start:end] = chunk_scores

    return scores


def evaluate(model, src_flat, src_padding_mask, tgt_item_indices,
             item_offset_codes, item_raw_codes, args):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src_flat, src_padding_mask)
        scores = score_all_items(model, memory, src_padding_mask,
                                 item_offset_codes, item_raw_codes,
                                 args.n_q, args.device)
        recall_5, ndcg_5 = calculate_metrics(scores, tgt_item_indices, 5)
        recall_10, ndcg_10 = calculate_metrics(scores, tgt_item_indices, 10)
    return recall_5, ndcg_5, recall_10, ndcg_10


def train(args):
    print("Preparing Amazon dataset...")
    user_histories_idx, item_catalog, item_to_id, id_to_item, _ = prepare_data()
    num_items = len(item_catalog)

    codes_file = f"/home/acolombo/VAEs/dataset/Amazon/item_codes_c{args.c}.pt"
    if not os.path.exists(codes_file):
        raise FileNotFoundError(f"Item codes not found at {codes_file}. Run train_vae.py first.")

    raw_codes = torch.load(codes_file).to(args.device).long()
    item_offset_codes = offset_codes(raw_codes, args.bins)

    pad_token = args.bins * args.n_q
    pad_entry = torch.full((1, args.n_q), pad_token, device=args.device, dtype=torch.long)
    item_offset_padded = torch.cat([item_offset_codes, pad_entry], dim=0)

    model = RecommenderT5(
        bins=args.bins, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, n_q=args.n_q, dropout=args.dropout,
        dim_feedforward=args.dim_feedforward
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    warmup_steps = args.warmup_epochs * (len(user_histories_idx['train']) // args.batch_size + 1)
    total_steps = args.epochs * (len(user_histories_idx['train']) // args.batch_size + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    train_seqs = user_histories_idx['train']
    val_seqs = user_histories_idx['val']
    test_seqs = user_histories_idx['test']

    train_dataset = AmazonSequenceDataset(train_seqs, num_items, args.seq_len)
    val_dataset = AmazonSequenceDataset(val_seqs, num_items, args.seq_len)
    test_dataset = AmazonSequenceDataset(test_seqs, num_items, args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Training on {len(train_seqs)} sequences, Val: {len(val_seqs)}, Test: {len(test_seqs)}.")
    pad_item = num_items

    best_val_recall = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for src_idx, tgt_idx in train_loader:
            src_idx = src_idx.to(args.device)
            tgt_idx = tgt_idx.to(args.device)

            src_flat, src_padding_mask = flatten_history(
                src_idx, item_offset_padded, model.pad_token)
            tgt_offset = item_offset_codes[tgt_idx]
            tgt_raw = raw_codes[tgt_idx]

            optimizer.zero_grad()
            logits = model(src_flat, src_padding_mask, tgt_offset)

            loss = sum(loss_fn(logits[q], tgt_raw[:, q]) for q in range(args.n_q))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            avg_loss = total_loss / len(train_loader)
            val_metrics = [[], [], [], []]
            with torch.no_grad():
                for src_idx, tgt_idx in val_loader:
                    src_idx = src_idx.to(args.device)
                    tgt_idx = tgt_idx.to(args.device)
                    src_flat, src_padding_mask = flatten_history(
                        src_idx, item_offset_padded, model.pad_token)
                    r5, n5, r10, n10 = evaluate(
                        model, src_flat, src_padding_mask, tgt_idx,
                        item_offset_codes[:num_items], raw_codes[:num_items], args)
                    val_metrics[0].append(r5)
                    val_metrics[1].append(n5)
                    val_metrics[2].append(r10)
                    val_metrics[3].append(n10)

            val_recall_10 = sum(val_metrics[2]) / len(val_metrics[2])
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Recall@10: {val_recall_10:.4f}")
            if val_recall_10 > best_val_recall:
                best_val_recall = val_recall_10
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest Val Recall@10: {best_val_recall:.4f}")

    print("\n--- Evaluation on Test Set ---")
    test_metrics = [[], [], [], []]
    with torch.no_grad():
        for src_idx, tgt_idx in test_loader:
            src_idx = src_idx.to(args.device)
            tgt_idx = tgt_idx.to(args.device)
            src_flat, src_padding_mask = flatten_history(
                src_idx, item_offset_padded, model.pad_token)
            r5, n5, r10, n10 = evaluate(
                model, src_flat, src_padding_mask, tgt_idx,
                item_offset_codes[:num_items], raw_codes[:num_items], args)
            test_metrics[0].append(r5)
            test_metrics[1].append(n5)
            test_metrics[2].append(r10)
            test_metrics[3].append(n10)

    print(f"Recall@5: {sum(test_metrics[0]) / len(test_metrics[0]):.4f}")
    print(f"NDCG@5: {sum(test_metrics[1]) / len(test_metrics[1]):.4f}")
    print(f"Recall@10: {sum(test_metrics[2]) / len(test_metrics[2]):.4f}")
    print(f"NDCG@10: {sum(test_metrics[3]) / len(test_metrics[3]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--n_q", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
