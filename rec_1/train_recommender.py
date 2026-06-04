import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import os
import sys
from collections import defaultdict
from torch.utils.data import DataLoader

torch.backends.cuda.enable_cudnn_sdp(False)

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


def build_item_trie(raw_codes, n_q, bins, device):
    """Build a prefix trie over item multitokens for constrained beam search.

    Generative retrieval (Rajput et al., TIGER) decodes the next item's
    multitoken autoregressively and constrains each step to tokens that lead to
    a real item. This builds the data structures for that constraint.

    Returns:
        allowed: list (len n_q); allowed[q] is a (num_nodes_q, bins) bool tensor,
                 True where a token is a valid continuation of that prefix node.
        child:   list (len n_q); child[q] is a (num_nodes_q, bins) long tensor
                 giving the level-(q+1) node id reached by each token (-1 if none).
        code_to_items: dict mapping a full multitoken tuple -> list of item ids
                 (a list because items can collide on the same multitoken).
    """
    codes = raw_codes.tolist()
    nodes_by_level = [dict() for _ in range(n_q)]   # prefix tuple -> node id
    nodes_by_level[0][()] = 0
    allowed_sets = [defaultdict(set) for _ in range(n_q)]  # node id -> {tokens}
    child_maps = [dict() for _ in range(n_q)]              # (node id, token) -> child id
    next_node_id = [1] + [0] * (n_q - 1)            # level 0 already has the root

    for code in codes:
        prefix = ()
        for q in range(n_q):
            node_id = nodes_by_level[q][prefix]
            tok = code[q]
            allowed_sets[q][node_id].add(tok)
            child_prefix = prefix + (tok,)
            if q + 1 < n_q:
                if child_prefix not in nodes_by_level[q + 1]:
                    nodes_by_level[q + 1][child_prefix] = next_node_id[q + 1]
                    next_node_id[q + 1] += 1
                child_maps[q][(node_id, tok)] = nodes_by_level[q + 1][child_prefix]
            prefix = child_prefix

    allowed, child = [], []
    for q in range(n_q):
        n_nodes = len(nodes_by_level[q])
        a = torch.zeros(n_nodes, bins, dtype=torch.bool, device=device)
        c = torch.full((n_nodes, bins), -1, dtype=torch.long, device=device)
        for node_id, toks in allowed_sets[q].items():
            a[node_id, torch.tensor(sorted(toks), device=device)] = True
        for (node_id, tok), child_id in child_maps[q].items():
            c[node_id, tok] = child_id
        allowed.append(a)
        child.append(c)

    code_to_items = defaultdict(list)
    for item_id, code in enumerate(codes):
        code_to_items[tuple(code)].append(item_id)

    return allowed, child, code_to_items


@torch.no_grad()
def beam_search(model, memory, memory_padding_mask, trie, num_beams, n_q, bins, device):
    """Constrained beam search over item multitokens (TIGER-style generation).

    Returns (final_seqs, final_scores):
        final_seqs:   (B, num_beams, n_q) raw token ids, sorted by score desc.
        final_scores: (B, num_beams) cumulative log-probabilities.
    """
    allowed, child, _ = trie
    B = memory.size(0)
    level_offsets = torch.arange(n_q, device=device) * bins

    seqs = torch.empty(B, 1, 0, dtype=torch.long, device=device)
    scores = torch.zeros(B, 1, device=device)
    node_ids = torch.zeros(B, 1, dtype=torch.long, device=device)  # root
    cur_beams = 1

    for q in range(n_q):
        N = B * cur_beams
        bos = torch.full((N, 1), model.bos_token, dtype=torch.long, device=device)
        if q > 0:
            offset_seq = seqs.reshape(N, q) + level_offsets[:q]
            dec_input = torch.cat([bos, offset_seq], dim=1)         # [BOS, t_0..t_{q-1}]
        else:
            dec_input = bos
        dec_emb = model.dec_pos(model.token_embedding(dec_input))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(q + 1, device=device)

        mem_exp = memory.repeat_interleave(cur_beams, dim=0)
        mask_exp = (memory_padding_mask.repeat_interleave(cur_beams, dim=0)
                    if memory_padding_mask is not None else None)

        out = model.decoder(dec_emb, mem_exp, tgt_mask=tgt_mask,
                            memory_key_padding_mask=mask_exp)
        logp = torch.log_softmax(model.fc_out[q](out[:, -1, :]), dim=-1)   # (N, bins)

        # Constrain to tokens that continue a real item's prefix.
        valid = allowed[q][node_ids.reshape(-1)]                   # (N, bins) bool
        logp = logp.masked_fill(~valid, float('-inf'))

        cand = (scores.reshape(N, 1) + logp).reshape(B, cur_beams * bins)
        k = min(num_beams, cand.size(1))
        top_scores, top_idx = cand.topk(k, dim=1)
        beam_idx = top_idx // bins
        tok_idx = top_idx % bins

        prev_seqs = torch.gather(seqs, 1, beam_idx.unsqueeze(-1).expand(B, k, q))
        seqs = torch.cat([prev_seqs, tok_idx.unsqueeze(-1)], dim=2)
        scores = top_scores

        if q + 1 < n_q:
            prev_nodes = torch.gather(node_ids, 1, beam_idx).reshape(-1).clamp_min(0)
            next_nodes = child[q][prev_nodes, tok_idx.reshape(-1)].clamp_min(0)
            node_ids = next_nodes.reshape(B, k)
        cur_beams = k

    return seqs, scores


def beam_metrics(final_seqs, targets, code_to_items, ks):
    """Map generated multitokens back to items and compute Recall@k / NDCG@k."""
    B, _, _ = final_seqs.shape
    seqs_list = final_seqs.tolist()
    tgt_list = targets.tolist()
    max_k = max(ks)
    recalls = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}

    for b in range(B):
        ranked, seen = [], set()
        for beam in seqs_list[b]:
            for it in code_to_items.get(tuple(beam), ()):
                if it not in seen:
                    seen.add(it)
                    ranked.append(it)
            if len(ranked) >= max_k:
                break
        tgt = tgt_list[b]
        for k in ks:
            top = ranked[:k]
            if tgt in top:
                recalls[k] += 1.0
                ndcgs[k] += 1.0 / math.log2(top.index(tgt) + 2)
    return recalls, ndcgs


@torch.no_grad()
def run_eval(model, loader, item_offset_padded, trie, args, ks=(5, 10)):
    model.eval()
    use_amp = args.device == 'cuda'
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_amp else torch.amp.autocast('cuda', enabled=False)
    code_to_items = trie[2]
    sums = {k: [0.0, 0.0] for k in ks}   # [recall_sum, ndcg_sum]
    n = 0
    for src_idx, tgt_idx in loader:
        src_idx = src_idx.to(args.device)
        tgt_idx = tgt_idx.to(args.device)
        src_flat, src_mask = flatten_history(src_idx, item_offset_padded, model.pad_token)
        with autocast_ctx:
            memory = model.encode(src_flat, src_mask)
            final_seqs, _ = beam_search(model, memory, src_mask, trie,
                                        args.num_beams, args.n_q, args.bins, args.device)
        recalls, ndcgs = beam_metrics(final_seqs, tgt_idx, code_to_items, ks)
        for k in ks:
            sums[k][0] += recalls[k]
            sums[k][1] += ndcgs[k]
        n += tgt_idx.size(0)
    return {k: (sums[k][0] / n, sums[k][1] / n) for k in ks}


def train(args):
    print("Preparing Amazon dataset...")
    user_histories_idx, item_catalog, item_to_id, id_to_item, _ = prepare_data()
    num_items = len(item_catalog)

    suffix = '_dedup' if args.dedup else ''
    codes_file = f"/home/acolombo/VAEs/dataset/Amazon/item_codes_c{args.c}{suffix}.pt"
    if not os.path.exists(codes_file):
        raise FileNotFoundError(
            f"Item codes not found at {codes_file}. Run train_vae.py first"
            + (" with --dedup." if args.dedup else "."))

    raw_codes = torch.load(codes_file).to(args.device).long()
    # The saved tensor already has the right number of levels: n_q semantic
    # tokens, plus a uniqueness/tie-break token when --dedup (paper §2.2). Drive
    # the whole token pipeline off the actual width so every full id is unique.
    n_levels = raw_codes.size(1)
    if args.dedup:
        max_tok = int(raw_codes.max().item())
        if max_tok >= args.bins:
            raise ValueError(
                f"Dedup token value {max_tok} >= bins {args.bins}: a collision "
                f"group is larger than the per-level vocab. Increase --bins.")
        print(f"Dedup enabled: {n_levels} levels "
              f"({args.n_q} semantic + 1 uniqueness token).")
    args.n_q = n_levels
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

    dl_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, **dl_kwargs)

    # Prefix trie over item multitokens for constrained beam-search generation.
    trie = build_item_trie(raw_codes[:num_items], args.n_q, args.bins, args.device)

    model = torch.compile(model)

    print(f"Training on {len(train_seqs)} sequences, Val: {len(val_seqs)}, Test: {len(test_seqs)}.")

    use_amp = args.device == 'cuda'
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_amp else torch.amp.autocast('cuda', enabled=False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

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
            with autocast_ctx:
                logits = model(src_flat, src_padding_mask, tgt_offset)
                loss = sum(loss_fn(logits[q], tgt_raw[:, q]) for q in range(args.n_q))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            avg_loss = total_loss / len(train_loader)
            metrics = run_eval(model, val_loader, item_offset_padded, trie, args)
            val_recall_10 = metrics[10][0]
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Recall@10: {val_recall_10:.4f}")
            if val_recall_10 > best_val_recall:
                best_val_recall = val_recall_10
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest Val Recall@10: {best_val_recall:.4f}")

    print("\n--- Evaluation on Test Set ---")
    metrics = run_eval(model, test_loader, item_offset_padded, trie, args)
    print(f"Recall@5: {metrics[5][0]:.4f}")
    print(f"NDCG@5: {metrics[5][1]:.4f}")
    print(f"Recall@10: {metrics[10][0]:.4f}")
    print(f"NDCG@10: {metrics[10][1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--n_q", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help='Batch size for beam-search evaluation (expands x num_beams)')
    parser.add_argument("--num_beams", type=int, default=50,
                        help='Beam width for generative retrieval at eval time')
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--dedup", action='store_true',
                        help='Load the uniqueness-token codes '
                             '(item_codes_c<c>_dedup.pt) so colliding items get '
                             'distinct ids (paper §2.2 / TIGER). Requires '
                             'train_vae.py to have been run with --dedup.')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
