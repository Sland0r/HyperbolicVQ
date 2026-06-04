"""Paper-faithful Recall@10 evaluation for Hierarchy Modeling (§4.1 / Appendix C.1).

Standalone alternative to ``NLP/eval_recall.py``. Differences from the original:

- Recall@10 is measured by GENERATING ranked multitoken sequences with beam
  search (beam_size=10) — matching the paper's "we generate k ranked guesses"
  and "recall@10". The original's default per-position top-k metric is dropped;
  pass --greedy_positionwise only if you explicitly want that diagnostic.
- The seq2seq downstream model is unchanged (4+4 layers, d=256, ff=1024,
  8 heads, tied embeddings, 100 epochs, Adam lr=1e-3).
- Reads ``split_mode`` from the training config so train/test pairs match the
  trained model.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NLP_2.train_paper import HRQModel
from NLP_2.dataset_paper import WordNetHierarchyDataset

# cuDNN's SDPA backend fails to build an execution plan for the seq2seq
# Transformer's attention on this GPU/cuDNN build ("No valid execution plans
# built"). Disable it so SDPA falls back to flash / mem-efficient / math.
torch.backends.cuda.enable_cudnn_sdp(False)

# Reserve index 0 as BOS; codebook indices are shifted to 1..bins
BOS_TOKEN = 0


class Seq2SeqTransformer(nn.Module):
    """Seq2seq transformer (Appendix C.1): 4 enc + 4 dec layers, d_model=256,
    ff=1024, 8 heads, tied encoder/decoder embeddings."""

    def __init__(self, vocab_size=257, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, n_q=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb_enc = nn.Parameter(torch.randn(1, n_q, d_model))
        self.pos_emb_dec = nn.Parameter(torch.randn(1, n_q + 1, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        # Tie embeddings (paper)
        self.fc_out.weight = self.embedding.weight
        self.n_q = n_q
        self.vocab_size = vocab_size

    def forward(self, src, tgt, teacher_forcing=False):
        B = src.size(0)
        if teacher_forcing:
            bos = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=src.device)
            dec_input = torch.cat([bos, tgt[:, :-1]], dim=1)
            src_emb = self.embedding(src) + self.pos_emb_enc
            dec_emb = self.embedding(dec_input) + self.pos_emb_dec[:, :self.n_q, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.n_q).to(src.device)
            return self.fc_out(self.transformer(src_emb, dec_emb, tgt_mask=tgt_mask))
        else:
            device = src.device
            src_emb = self.embedding(src) + self.pos_emb_enc
            memory = self.transformer.encoder(src_emb)
            dec_input = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
            all_logits = []
            for _ in range(self.n_q):
                seq_len = dec_input.size(1)
                dec_emb = self.embedding(dec_input) + self.pos_emb_dec[:, :seq_len, :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                out = self.transformer.decoder(dec_emb, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1, :])
                all_logits.append(logits.unsqueeze(1))
                next_token = logits.argmax(dim=-1, keepdim=True)
                dec_input = torch.cat([dec_input, next_token], dim=1)
            return torch.cat(all_logits, dim=1)

    def beam_search_batch(self, srcs, beam_size=10):
        """Batched beam search. srcs: (B, n_q) 1-indexed.
        Returns (B, beam_size, n_q) top predicted multitokens (1-indexed)."""
        device = srcs.device
        B = srcs.size(0)
        vocab_size = self.vocab_size

        src_emb = self.embedding(srcs) + self.pos_emb_enc
        memory = self.transformer.encoder(src_emb)

        beam_seqs = torch.full((B, 1, 1), BOS_TOKEN, dtype=torch.long, device=device)
        beam_scores = torch.zeros(B, 1, device=device)

        for _ in range(self.n_q):
            n_beams = beam_seqs.size(1)
            seq_len = beam_seqs.size(2)
            flat_B = B * n_beams

            flat_seqs = beam_seqs.view(flat_B, seq_len)
            tgt_emb = self.embedding(flat_seqs) + self.pos_emb_dec[:, :seq_len, :].expand(flat_B, -1, -1)
            mem = memory.unsqueeze(1).expand(-1, n_beams, -1, -1).reshape(flat_B, self.n_q, -1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            out = self.transformer.decoder(tgt_emb, mem, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).view(B, n_beams, vocab_size)

            candidate_scores = beam_scores.unsqueeze(-1) + log_probs
            flat_scores = candidate_scores.view(B, -1)
            topk_scores, topk_flat_idx = torch.topk(flat_scores, k=beam_size, dim=-1)

            beam_idx = topk_flat_idx // vocab_size
            token_idx = topk_flat_idx % vocab_size

            prev_seqs = torch.gather(beam_seqs, 1, beam_idx.unsqueeze(-1).expand(-1, -1, seq_len))
            beam_seqs = torch.cat([prev_seqs, token_idx.unsqueeze(-1)], dim=-1)
            beam_scores = topk_scores

        return beam_seqs[:, :, 1:]  # strip BOS -> (B, beam_size, n_q)


def compute_baselines(dataset, k=10):
    """No-model leakage baselines (popularity@k, graph-composition@k)."""
    from collections import Counter, defaultdict

    train = dataset.train_pairs
    test = dataset.test_pairs
    n = len(test)
    if n == 0:
        return 0.0, 0.0

    gfreq = Counter(v for _, v in train)
    top_set = set(v for v, _ in gfreq.most_common(k))
    pop_hits = sum(1 for _, v in test if v in top_set)

    parents = defaultdict(set)
    for u, v in train:
        parents[u].add(v)

    sys.setrecursionlimit(1_000_000)
    reach_cache = {}

    def reach(u):
        if u in reach_cache:
            return reach_cache[u]
        reach_cache[u] = set()
        acc = set(parents[u])
        for p in parents[u]:
            acc |= reach(p)
        reach_cache[u] = acc
        return acc

    comp_hits = 0
    for u, v in test:
        cand = sorted(reach(u), key=lambda x: -gfreq[x])[:k]
        if v in cand:
            comp_hits += 1

    return pop_hits / n * 100, comp_hits / n * 100


def load_config(checkpoint_dir):
    config_path = os.path.join(checkpoint_dir, 'config.py')
    cfg = {}
    with open(config_path) as f:
        exec(f.read(), cfg)
    return {k: v for k, v in cfg.items() if not k.startswith('_')}


def run_evaluation(checkpoint_dir, epochs=100, batch_size=128,
                   beam_search=False, teacher_forcing=False):
    cfg = load_config(checkpoint_dir)
    c = cfg['c']
    embed_dim = cfg['embed_dim']
    n_q = cfg['n_q']
    bins = cfg['bins']
    split_mode = cfg.get('split_mode', 'closure')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_path = os.path.join(checkpoint_dir, 'logs.txt')
    log_file = open(log_path, 'a')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("")
    log("=" * 50)
    log("EVALUATION")
    log("=" * 50)
    log(f"Evaluating model with c={c}, split_mode={split_mode}...")

    dataset = WordNetHierarchyDataset(num_negatives=1, split_mode=split_mode)
    hrq_model = HRQModel(
        vocab_size=dataset.vocab_size, embed_dim=embed_dim, n_q=n_q, bins=bins, c=c,
        new_method=cfg.get('new_method', False), hste=cfg.get('hste', False),
        gradient_correction=cfg.get('gradient_correction', False),
    ).to(device)
    model_path = os.path.join(checkpoint_dir, 'model.pt')
    if os.path.exists(model_path):
        hrq_model.load_state_dict(torch.load(model_path, map_location=device))
    hrq_model.eval()

    # ── Extract discrete multitokens for every noun ──────────────────
    log("Extracting discrete tokens for all concepts...")
    all_noun_indices = torch.arange(dataset.vocab_size, device=device)
    with torch.no_grad():
        all_tokens = []
        batch_sz = 1024
        for i in range(0, dataset.vocab_size, batch_sz):
            batch_idx = all_noun_indices[i:i + batch_sz]
            _, _, codes, _, _ = hrq_model(batch_idx)
            codes = codes.squeeze(-1) if len(codes.shape) > 2 else codes
            if codes.size(1) == batch_idx.size(0) and codes.size(0) != batch_idx.size(0):
                codes = codes.transpose(0, 1)
            all_tokens.append(codes)
        all_tokens = torch.cat(all_tokens, dim=0)  # (vocab_size, n_q)

    n_unique = torch.unique(all_tokens, dim=0).shape[0]
    log(f"  Unique token sequences: {n_unique} / {all_tokens.shape[0]} total concepts")

    train_pairs = dataset.train_pairs
    test_pairs = dataset.test_pairs
    log(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    pop_baseline, comp_baseline = compute_baselines(dataset, k=10)
    log(f"  [baseline] global-popularity@10:   {pop_baseline:.1f}%")
    log(f"  [baseline] graph-composition@10:   {comp_baseline:.1f}%")

    all_tokens = all_tokens + 1  # shift 0..bins-1 -> 1..bins (0 = BOS)

    # ── Train seq2seq transformer (paper: 100 epochs, Adam lr=1e-3) ──
    log(f"Training Seq2Seq model ({epochs} epochs, teacher_forcing={teacher_forcing})...")
    transformer = Seq2SeqTransformer(vocab_size=bins + 1, n_q=n_q).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Pre-build index tensors once (avoids per-batch host->device transfer).
    train_src = torch.tensor([u for u, _ in train_pairs], device=device)
    train_tgt = torch.tensor([v for _, v in train_pairs], device=device)
    n_train = train_src.size(0)
    use_amp = device == 'cuda'

    for ep in tqdm(range(epochs), desc="Seq2Seq training"):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = torch.zeros((), device=device)
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            src_tokens = all_tokens[train_src[idx]]
            tgt_tokens = all_tokens[train_tgt[idx]]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = transformer(src_tokens, tgt_tokens, teacher_forcing=teacher_forcing)
                loss = criterion(logits.view(-1, bins + 1), tgt_tokens.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            n_batches += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}/{epochs}, avg loss: {epoch_loss.item() / n_batches:.4f}")

    # ── Evaluate Recall@10 ───────────────────────────────────────────
    transformer.eval()
    k = 10
    eval_batch_size = 512
    hits = 0

    if beam_search:
        log(f"Evaluating Recall@{k} (beam search, beam_size={k})...")
        with torch.no_grad():
            for i in tqdm(range(0, len(test_pairs), eval_batch_size), desc="Recall eval"):
                batch_pairs = test_pairs[i:i + eval_batch_size]
                src_indices = torch.tensor([u for u, v in batch_pairs], device=device)
                tgt_indices = torch.tensor([v for u, v in batch_pairs], device=device)
                src_tokens_batch = all_tokens[src_indices]
                tgt_tokens_batch = all_tokens[tgt_indices]
                candidates = transformer.beam_search_batch(src_tokens_batch, beam_size=k)
                matches = (candidates == tgt_tokens_batch.unsqueeze(1)).all(dim=-1).any(dim=-1)
                hits += matches.sum().item()
        decode_mode = "beam search"
    else:
        log(f"Evaluating Recall@{k} (per-position top-{k} diagnostic)...")
        with torch.no_grad():
            for i in tqdm(range(0, len(test_pairs), eval_batch_size), desc="Recall eval"):
                batch_pairs = test_pairs[i:i + eval_batch_size]
                src_indices = torch.tensor([u for u, v in batch_pairs], device=device)
                tgt_indices = torch.tensor([v for u, v in batch_pairs], device=device)
                src_tokens_batch = all_tokens[src_indices]
                tgt_tokens_batch = all_tokens[tgt_indices]
                logits = transformer(src_tokens_batch, tgt_tokens_batch, teacher_forcing=False)
                topk_preds = logits.topk(k, dim=-1).indices
                position_hits = (topk_preds == tgt_tokens_batch.unsqueeze(-1)).any(dim=-1)
                hits += position_hits.all(dim=-1).sum().item()
        decode_mode = "per-position top-k"

    recall_k = hits / len(test_pairs) * 100
    model_type = "HRQ" if c > 0 else "RQ"
    log("")
    log("-" * 50)
    log("RESULTS")
    log("-" * 50)
    log(f"  Model type:              {model_type} (c={c})")
    log(f"  Split mode:              {split_mode}")
    log(f"  Decode mode:             {decode_mode}")
    log(f"  Total concepts:          {all_tokens.shape[0]}")
    log(f"  Unique token sequences:  {n_unique}")
    log(f"  Train pairs:             {len(train_pairs)}")
    log(f"  Test pairs:              {len(test_pairs)}")
    log(f"  Recall@{k}:               {recall_k:.1f}%")
    log(f"  Popularity@{k} baseline:  {pop_baseline:.1f}%")
    log(f"  Composition@{k} baseline: {comp_baseline:.1f}%")
    log("=" * 50)
    log_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Checkpoint subdirectory name under checkpoint/nlp_2/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--teacher_forcing', action='store_true')
    parser.add_argument('--greedy_positionwise', action='store_true',
                        help='Use the old per-position top-k diagnostic instead '
                             'of paper-faithful beam-search generation')
    args = parser.parse_args()
    run_evaluation(
        checkpoint_dir='/home/acolombo/VAEs/checkpoint/nlp_2/' + args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beam_search=not args.greedy_positionwise,
        teacher_forcing=args.teacher_forcing,
    )
