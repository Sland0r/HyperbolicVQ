import torch
import torch.nn as nn
import geoopt
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=0, world_size=1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = geoopt.ManifoldParameter(torch.randn(10, 10), manifold=geoopt.PoincareBall(c=1.0))

m = Model()
print("Before DDP:")
for p in m.parameters():
    print(type(p), isinstance(p, geoopt.ManifoldParameter), hasattr(p, 'manifold'))

m = DDP(m)
print("After DDP:")
for p in m.parameters():
    print(type(p), isinstance(p, geoopt.ManifoldParameter), hasattr(p, 'manifold'))
