import torch
import geoopt

c = torch.tensor(1.0)
ball = geoopt.PoincareBall(c=c)

u = torch.tensor([0.2, 0.3])
v = torch.tensor([-0.1, 0.4])
w = torch.tensor([0.5, -0.2])

def mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-5)

gyr_w = geoopt.manifolds.poincare.math.gyration(u, v, w, c)
print("geoopt gyration:", gyr_w)

gyr_manual = mobius_add(-mobius_add(u, v, c), mobius_add(u, mobius_add(v, w, c), c), c)
print("manual gyration:", gyr_manual)
