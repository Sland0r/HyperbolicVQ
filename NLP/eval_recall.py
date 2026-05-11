import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NLP.train_hierarchy import HRQModel, WordNetHierarchyDataset

# Reserve index 0 as BOS; actual codebook indices are shifted to 1..bins
BOS_TOKEN = 0

class Seq2SeqTransformer(nn.Module):
    """Seq2seq transformer matching Appendix C of the HRQ paper:
    4 encoder + 4 decoder layers, d_model=256, ff=1024, 8 heads,
    tied encoder/decoder embeddings."""

    def __init__(self, vocab_size=257, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, n_q=4):
        super().__init__()
        # vocab_size = bins + 1 (for BOS token at index 0)
        self.embedding = nn.Embedding(vocab_size, d_model)
        # n_q + 1 positions: BOS + n_q target tokens
        self.pos_emb_enc = nn.Parameter(torch.randn(1, n_q, d_model))
        self.pos_emb_dec = nn.Parameter(torch.randn(1, n_q + 1, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Tie encoder/decoder embeddings (paper: "The embeddings of the
        # encoder and decoder are tied")
        self.fc_out.weight = self.embedding.weight

        self.n_q = n_q
        self.vocab_size = vocab_size

    def forward(self, src, tgt, teacher_forcing=True):
        """Forward pass for training.
        src: (B, n_q) source codebook indices (already shifted to 1..bins)
        tgt: (B, n_q) target codebook indices (already shifted to 1..bins)
        teacher_forcing: If True, use ground truth targets as decoder input.
                         If False, use greedy autoregressive decoding.
        Returns logits (B, n_q, vocab_size) predicting tgt at each position.
        """
        if teacher_forcing:
            B = src.size(0)
            # Decoder input: [BOS, tgt[0], tgt[1], ..., tgt[n_q-2]]  (length n_q)
            bos = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=src.device)
            dec_input = torch.cat([bos, tgt[:, :-1]], dim=1)  # (B, n_q)

            src_emb = self.embedding(src) + self.pos_emb_enc
            dec_emb = self.embedding(dec_input) + self.pos_emb_dec[:, :self.n_q, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.n_q).to(src.device)
            out = self.transformer(src_emb, dec_emb, tgt_mask=tgt_mask)
            return self.fc_out(out)
        else:
            # Autoregressive forward pass (greedy)
            B = src.size(0)
            device = src.device
            src_emb = self.embedding(src) + self.pos_emb_enc
            memory = self.transformer.encoder(src_emb)

            dec_input = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
            all_logits = []

            for t in range(self.n_q):
                seq_len = dec_input.size(1)
                dec_emb = self.embedding(dec_input) + self.pos_emb_dec[:, :seq_len, :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                
                out = self.transformer.decoder(dec_emb, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1, :])  # (B, vocab_size)
                all_logits.append(logits.unsqueeze(1))
                
                # Greedy choice for the next input (gradients won't flow through argmax,
                # but the loss will be computed on all_logits)
                next_token = logits.argmax(dim=-1, keepdim=True)
                dec_input = torch.cat([dec_input, next_token], dim=1)

            return torch.cat(all_logits, dim=1)  # (B, n_q, vocab_size)

    def beam_search(self, src, beam_size=10):
        """Autoregressively generate top beam_size candidate sequences.
        Args:
            src: (1, n_q) source multitoken (1-indexed)
            beam_size: number of beams to keep
        Returns:
            candidates: (beam_size, n_q) top predicted multitokens (1-indexed)
            scores: (beam_size,) log-probability scores
        """
        device = src.device
        vocab_size = self.vocab_size

        # Encode source once
        src_emb = self.embedding(src) + self.pos_emb_enc  # (1, n_q, d_model)
        memory = self.transformer.encoder(src_emb)         # (1, n_q, d_model)

        # Start with BOS token
        beam_seqs = torch.full((1, 1), BOS_TOKEN, dtype=torch.long, device=device)  # (1, 1)
        beam_scores = torch.zeros(1, device=device)

        for t in range(self.n_q):
            n_beams = beam_seqs.size(0)
            seq_len = beam_seqs.size(1)  # t + 1 (including BOS)

            tgt_emb = self.embedding(beam_seqs) + self.pos_emb_dec[:, :seq_len, :]
            mem = memory.expand(n_beams, -1, -1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            out = self.transformer.decoder(tgt_emb, mem, tgt_mask=tgt_mask)

            # Get logits for the last position
            logits = self.fc_out(out[:, -1, :])  # (n_beams, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)  # (n_beams, vocab_size)

            # Expand scores: (n_beams, vocab_size)
            candidate_scores = beam_scores.unsqueeze(1) + log_probs

            # Flatten and pick top beam_size
            flat_scores = candidate_scores.view(-1)
            topk_scores, topk_flat_idx = torch.topk(flat_scores, k=beam_size)

            beam_idx = topk_flat_idx // vocab_size
            token_idx = topk_flat_idx % vocab_size

            beam_seqs = torch.cat([
                beam_seqs[beam_idx],
                token_idx.unsqueeze(1)
            ], dim=1)  # (beam_size, t+2) — BOS + generated so far
            beam_scores = topk_scores

        # Strip the BOS column
        return beam_seqs[:, 1:], beam_scores  # (beam_size, n_q), (beam_size,)

    def beam_search_batch(self, srcs, beam_size=10):
        """Batched beam search for B sources simultaneously.
        Args:
            srcs: (B, n_q) source multitokens (1-indexed)
            beam_size: number of beams per source
        Returns:
            all_candidates: (B, beam_size, n_q) top predicted multitokens (1-indexed)
        """
        device = srcs.device
        B = srcs.size(0)
        vocab_size = self.vocab_size

        # Encode all sources at once
        src_emb = self.embedding(srcs) + self.pos_emb_enc  # (B, n_q, d_model)
        memory = self.transformer.encoder(src_emb)          # (B, n_q, d_model)

        # Initialize: all beams start with BOS
        # beam_seqs: (B, 1, 1) — each source has 1 beam with [BOS]
        beam_seqs = torch.full((B, 1, 1), BOS_TOKEN, dtype=torch.long, device=device)
        beam_scores = torch.zeros(B, 1, device=device)  # (B, 1)

        for t in range(self.n_q):
            n_beams = beam_seqs.size(1)
            seq_len = beam_seqs.size(2)  # t + 1 (BOS + generated tokens)
            flat_B = B * n_beams

            # beam_seqs: (B, n_beams, seq_len) → (B*n_beams, seq_len)
            flat_seqs = beam_seqs.view(flat_B, seq_len)
            tgt_emb = self.embedding(flat_seqs) + self.pos_emb_dec[:, :seq_len, :].expand(flat_B, -1, -1)
            mem = memory.unsqueeze(1).expand(-1, n_beams, -1, -1).reshape(flat_B, self.n_q, -1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            out = self.transformer.decoder(tgt_emb, mem, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1, :])  # (B*n_beams, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(B, n_beams, vocab_size)

            # Expand: (B, n_beams, vocab_size)
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs
            # Flatten beams*vocab: (B, n_beams*vocab_size)
            flat_scores = candidate_scores.view(B, -1)
            topk_scores, topk_flat_idx = torch.topk(flat_scores, k=beam_size, dim=-1)  # (B, beam_size)

            beam_idx = topk_flat_idx // vocab_size  # (B, beam_size)
            token_idx = topk_flat_idx % vocab_size  # (B, beam_size)

            # Gather previous seqs: use beam_idx to select which beams to keep
            prev_seqs = torch.gather(
                beam_seqs, 1,
                beam_idx.unsqueeze(-1).expand(-1, -1, seq_len)
            )  # (B, beam_size, seq_len)
            beam_seqs = torch.cat([prev_seqs, token_idx.unsqueeze(-1)], dim=-1)  # (B, beam_size, seq_len+1)
            beam_scores = topk_scores

        # Strip the BOS column
        return beam_seqs[:, :, 1:]  # (B, beam_size, n_q)


def run_evaluation(hrq_model_path="hrq_model.pt", c=1.0, embed_dim=16, n_q=4, bins=128, epochs=10, teacher_forcing=False, batch_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating model with c={c}...")

    dataset = WordNetHierarchyDataset(num_negatives=1)
    hrq_model = HRQModel(vocab_size=dataset.vocab_size, embed_dim=embed_dim, n_q=n_q, bins=bins, c=c).to(device)
    if os.path.exists(hrq_model_path):
        hrq_model.load_state_dict(torch.load(hrq_model_path, map_location=device))
    hrq_model.eval()

    # ── Extract discrete multitokens for every noun ──────────────────
    print("Extracting discrete tokens for all concepts...")
    all_noun_indices = torch.arange(dataset.vocab_size, device=device)
    with torch.no_grad():
        all_tokens = []
        batch_sz = 1024
        for i in range(0, dataset.vocab_size, batch_sz):
            batch_idx = all_noun_indices[i:i+batch_sz]
            _, _, codes, _ = hrq_model(batch_idx)
            codes = codes.squeeze(-1) if len(codes.shape) > 2 else codes
            if codes.size(1) == batch_idx.size(0) and codes.size(0) != batch_idx.size(0):
                codes = codes.transpose(0, 1)
            all_tokens.append(codes)
        all_tokens = torch.cat(all_tokens, dim=0)  # (vocab_size, n_q)

    n_unique = torch.unique(all_tokens, dim=0).shape[0]
    print(f"  Unique token sequences: {n_unique} / {all_tokens.shape[0]} total concepts")

    # ── Train/test split (paper: 85/15 random split of hypernymy pairs) ──
    pairs = dataset.hypernymy_pairs
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.85)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # ── Shift codebook indices: 0 is reserved for BOS ────────────────
    all_tokens = all_tokens + 1  # shift from 0..bins-1 to 1..bins

    # ── Train seq2seq transformer (paper: 100 epochs, Adam lr=1e-3) ──
    print("Training Sequence-to-Sequence model...")
    transformer = Seq2SeqTransformer(vocab_size=bins + 1, n_q=n_q).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for ep in tqdm(range(epochs), desc="Seq2Seq training"):
        random.shuffle(train_pairs)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            src_idx = torch.tensor([u for u, v in batch], device=device)
            tgt_idx = torch.tensor([v for u, v in batch], device=device)
            src_tokens = all_tokens[src_idx]  # (B, n_q), values in 1..bins
            tgt_tokens = all_tokens[tgt_idx]  # (B, n_q), values in 1..bins

            optimizer.zero_grad()
            logits = transformer(src_tokens, tgt_tokens, teacher_forcing=teacher_forcing)  # (B, n_q, vocab_size)
            loss = criterion(logits.view(-1, bins + 1), tgt_tokens.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}/{epochs}, avg loss: {epoch_loss / n_batches:.4f}")

    # ── Evaluate Recall@10 via beam search ────────────────────────────
    # For each test pair (u, v), use beam search to generate top-10
    # candidate multitokens for the hypernym of u, then check if v's
    # multitoken appears in the beam.
    print("Evaluating Recall@10...")
    hits = 0
    transformer.eval()
    k = 10

    eval_batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(test_pairs), eval_batch_size), desc="Recall eval"):
            batch_pairs = test_pairs[i:i+eval_batch_size]
            src_indices = torch.tensor([u for u, v in batch_pairs], device=device)
            tgt_indices = torch.tensor([v for u, v in batch_pairs], device=device)
            src_tokens_batch = all_tokens[src_indices]   # (B, n_q)
            tgt_tokens_batch = all_tokens[tgt_indices]   # (B, n_q)

            # Batched beam search: (B, k, n_q)
            candidates = transformer.beam_search_batch(src_tokens_batch, beam_size=k)
            # Check matches: (B, k, n_q) vs (B, 1, n_q)
            matches = (candidates == tgt_tokens_batch.unsqueeze(1)).all(dim=-1).any(dim=-1)  # (B,)
            hits += matches.sum().item()

    recall_k = hits / len(test_pairs) * 100
    model_type = "HRQ" if c > 0 else "RQ"
    print(f"\n=========================================")
    print(f"  Recall@{k} for {model_type} (c={c}): {recall_k:.1f}%")
    print(f"  (unique codes: {n_unique}, test pairs: {len(test_pairs)})")
    print(f"=========================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--model_path', type=str, default='hrq_model.pt')
    parser.add_argument('--teacher_forcing', action='store_true', help='Enable teacher forcing during training')
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--n_q', type=int, default=4)
    parser.add_argument('--bins', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    run_evaluation(
        hrq_model_path=args.model_path,
        c=args.c,
        embed_dim=args.embed_dim,
        n_q=args.n_q,
        bins=args.bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        teacher_forcing=args.teacher_forcing
    )
