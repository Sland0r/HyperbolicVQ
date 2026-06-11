import sys, types, torch
sys.path.insert(0, '/gpfs/home4/acolombo/VAEs')
from academicodec.quantization.core_vq import ResidualVectorQuantization

torch.manual_seed(0)

def build(**kw):
    base = dict(num_quantizers=4, dim=8, codebook_size=16, c=1.0,
                kmeans_init=False, new_method=True, ema=False,
                commitment_weight=0.0, codebook_weight=0.0)
    base.update(kw)
    return ResidualVectorQuantization(**base)

def run(label, **kw):
    torch.manual_seed(0)
    rvq = build(**kw)
    rvq.train(); rvq.diag = True
    x = (0.3 * torch.randn(2, 8, 5)).requires_grad_(True)

    qgrads = {}
    orig_fwd = type(rvq.layers[0]).forward
    def patched(self, inp, _idx=[0]):
        q, ind, loss = orig_fwd(self, inp)
        i = _idx[0]; _idx[0] += 1
        if q.requires_grad:
            q.register_hook(lambda g, i=i: qgrads.__setitem__(i, g.norm().item()))
        return q, ind, loss
    for l in rvq.layers:
        l.forward = types.MethodType(patched, l)

    qout, idx, losses, dist = rvq(x, n_q=4)
    qout.sum().backward()
    print(f"--- {label} ---")
    print("  per-layer q-node grads:", {k: round(v,4) for k,v in qgrads.items()} or "none (all detached)")
    print("  grad at residual_in per layer:",
          [None if g != g else round(g, 4) for g in (rvq.diag_data["grad_in"] if rvq.diag_data else [])])
    print("  encoder grad norm:", round(x.grad.norm().item(), 6))
    cb = [l._codebook.embed.grad for l in rvq.layers]
    print("  codebook grads from recon:", [None if g is None else round(g.norm().item(),6) for g in cb])
    return qout

# 1) forward equality in eval mode: flag must not change the forward pass
torch.manual_seed(0); rvq_a = build()
torch.manual_seed(0); rvq_b = build(block_hste_pt=True)
rvq_a.eval(); rvq_b.eval()
xe = 0.3 * torch.randn(2, 8, 5)
with torch.no_grad():
    qa, ia, _, _ = rvq_a(xe.clone(), n_q=4)
    qb, ib, _, _ = rvq_b(xe.clone(), n_q=4)
print("eval forward identical:", torch.allclose(qa, qb), "| indices identical:", torch.equal(ia, ib))

# 1b) training forward value identical too (STE only reroutes gradients)
torch.manual_seed(0); rvq_a = build()
torch.manual_seed(0); rvq_b = build(block_hste_pt=True)
rvq_a.train(); rvq_b.train()
qa, ia, _, _ = rvq_a(xe.clone(), n_q=4)
qb, ib, _, _ = rvq_b(xe.clone(), n_q=4)
print("train forward value identical:", torch.allclose(qa.detach(), qb.detach()))

# 2) gradient routing
run("block_hste_pt (new_method)", block_hste_pt=True)
run("block_hste_pt + gyration_only", block_hste_pt=True, gyration_only=True)
run("block_hste (existing, identity hop)", block_hste=True)
run("block_hste_pt (standard mode)", block_hste_pt=True, new_method=False)
run("block_hste_pt + tangent_proj (codebook_dim=4)", block_hste_pt=True,
    codebook_dim=4, tangent_proj=True)

# 3) mutual exclusion
try:
    build(block_hste=True, block_hste_pt=True)
    print("ERROR: mutual exclusion not enforced")
except ValueError as e:
    print("mutual exclusion OK:", e)
