import torch
import math

def accumulate_train_codes(codes, train_codes_hist, train_codes_hist_last10,
                           total_train_codes, total_train_codes_last10,
                           bins, batch_idx, last_10_percent_start, device, max_q):
    if train_codes_hist is None:
        train_codes_hist = torch.zeros(max_q, bins, device=device)
        train_codes_hist_last10 = torch.zeros(max_q, bins, device=device)
        
    codes_count = codes.shape[1] * codes.shape[2]
    total_train_codes += codes_count
    
    is_last10 = batch_idx > last_10_percent_start
    if is_last10:
        total_train_codes_last10 += codes_count
        
    for q_idx in range(codes.shape[0]):
        codes_q = codes[q_idx].flatten()
        ones = torch.ones_like(codes_q, dtype=torch.float32)
        train_codes_hist[q_idx].scatter_add_(0, codes_q, ones)
        if is_last10:
            train_codes_hist_last10[q_idx].scatter_add_(0, codes_q, ones)
            
    return train_codes_hist, train_codes_hist_last10, total_train_codes, total_train_codes_last10

def compute_train_ppl(train_codes_hist, total_train_codes):
    ppl = []
    if train_codes_hist is not None and total_train_codes > 0:
        probs = train_codes_hist / train_codes_hist.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
        ppl = torch.exp2(entropy).tolist()
    return ppl

def accumulate_val_codes(codes, codes_hist, bins, device, max_q):
    if codes_hist is None:
        codes_hist = torch.zeros(max_q, bins, device=device)
    for q in range(codes.shape[0]):
        codes_hist[q].scatter_add_(0, codes[q].flatten(), torch.ones(codes[q].numel(), device=device))
    return codes_hist

def compute_val_ppl(codes_hist):
    val_ppls = []
    if codes_hist is not None:
        for q in range(codes_hist.shape[0]):
            sizes = codes_hist[q].long().tolist()
            sizes = [s / sum(sizes) for s in sizes]
            entropy = -sum(p * math.log2(p) for p in sizes if p > 0)
            perplexity = 2 ** entropy
            val_ppls.append(perplexity)
    return val_ppls
