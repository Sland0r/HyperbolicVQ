import torch
import geoopt

c = 1.0
ball = geoopt.PoincareBall(c=c)
p = geoopt.ManifoldParameter(torch.randn(10, 512), manifold=ball)

opt = geoopt.optim.RiemannianAdam([p], lr=0.1)

for _ in range(10):
    loss = p.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

norms = p.norm(dim=-1)
print(f"Max norm: {norms.max().item()}, Min norm: {norms.min().item()}")
