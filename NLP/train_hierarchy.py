import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from torch.utils.data import DataLoader
import argparse
import sys
import os

# Add parent dir to path to import academicodec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from academicodec.quantization.vq import ResidualVectorQuantizer
from NLP.wordnet_dataset import WordNetHierarchyDataset

class HRQModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_q=4, bins=256, c=1.0):
        super().__init__()
        # Continuous embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Hyperbolic residual quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=embed_dim,
            n_q=n_q,
            bins=bins,
            c=c
        )
        self.c = c

    def forward(self, x):
        # x is shape [batch_size]
        emb = self.embedding(x)
        # RVQ expects [batch, channels, seq_len]
        emb_unsqueezed = emb.unsqueeze(-1)
        quantized, codes, _, penalty = self.quantizer(emb_unsqueezed, sample_rate=0)
        return emb, quantized.squeeze(-1), codes, penalty
         

def exp_map0(v, c):
    if c == 0.0:
        return v
    norm = v.norm(dim=-1, keepdim=True)
    sqrt_c = c ** 0.5
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm.clamp_min(1e-5))
    return v * scale

def poincare_distance(x, y, c):
    """Compute poincaré distance (NOT squared) between x and y.

    When c == 0 returns Euclidean distance.  When c > 0, maps tangent-space
    vectors onto the Poincaré ball first and returns the geodesic distance.
    """
    if c > 0.0:
        x = exp_map0(x, c)
        y = exp_map0(y, c)

    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)

    sq_dist = (x2 + y2 - 2 * xy).clamp_min(0.0)

    if c == 0.0:
        return (sq_dist + 1e-8).sqrt()  # eps prevents ∇sqrt(0) = ∞

    denom = ((1 - c * x2) * (1 - c * y2)).clamp_min(1e-6)
    arg = 1 + 2 * c * sq_dist / denom
    dist = (1 / (c**0.5)) * torch.acosh(arg.clamp_min(1.0 + 1e-5))
    return dist

def train(args):
    dataset = WordNetHierarchyDataset(num_negatives=50)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = HRQModel(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        n_q=args.n_q,
        bins=args.bins,
        c=args.c
    ).to(args.device)
    
    if args.c > 0:
        manifold_params = []
        euclidean_params = []
        for p in model.parameters():
            if hasattr(p, "manifold"):
                manifold_params.append(p)
            else:
                euclidean_params.append(p)
        print(f"Manifold params: {len(manifold_params)}, Euclidean params: {len(euclidean_params)}")

        # Paper (Appendix C.1): Riemannian SGD for HRQ
        param_groups = []
        if manifold_params:
            param_groups.append({"params": manifold_params, "lr": args.warmup_lr})
        if euclidean_params:
            param_groups.append({"params": euclidean_params, "lr": args.warmup_lr})
        optimizer = geoopt.optim.RiemannianSGD(param_groups, lr=args.warmup_lr)
    else:
        # Paper (Appendix C.1): SGD for RQ
        optimizer = optim.SGD(model.parameters(), lr=args.warmup_lr)
    
    for epoch in range(args.epochs):
        # Paper (Appendix C.1): warmup lr=0.01 for first 20 epochs, then lr=1.0
        if epoch == args.warmup_epochs:
            new_lr = args.lr
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            print(f"Warmup done — switching lr to {new_lr}")

        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            u = batch['u'].to(args.device)
            v = batch['v'].to(args.device)
            negatives = batch['negatives'].to(args.device)
            
            u_emb, u_quant, u_codes, u_commit = model(u)
            v_emb, v_quant, v_codes, v_commit = model(v)
            
            B, num_neg = negatives.shape
            neg_flat = negatives.view(-1)
            neg_emb, neg_quant, _, _ = model(neg_flat)
            neg_emb = neg_emb.view(B, num_neg, -1)
            
            # Paper: Contrastive loss is calculated on continuous embeddings E_theta
            pos_dist = poincare_distance(u_emb, v_emb, args.c).squeeze(-1)
            
            u_expanded = u_emb.unsqueeze(1).expand(-1, num_neg, -1).reshape(B*num_neg, -1)
            neg_dist = poincare_distance(u_expanded, neg_emb.reshape(B*num_neg, -1), args.c).view(B, num_neg)
            
            # Add self-distance d(u, u) = 0 to negatives to match the paper H'(u) definition
            self_dist = torch.zeros((B, 1), device=args.device)
            
            # Paper: log-softmax contrastive loss (no temperature scaling)
            logits = torch.cat([-pos_dist.unsqueeze(1), -neg_dist, -self_dist], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=args.device)
            
            ce_loss = nn.CrossEntropyLoss()(logits, labels)
            # VQ commitment loss (already computed inside the quantizer)
            loss = ce_loss + u_commit.mean() + v_commit.mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Save the model
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--n_q', type=int, default=4)
    parser.add_argument('--bins', type=int, default=256)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--warmup_lr', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='hrq_model.pt')
    
    args = parser.parse_args()
    train(args)
