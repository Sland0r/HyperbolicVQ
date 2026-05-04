import torch
import geoopt
ball = geoopt.PoincareBall(c=1.0)
x = torch.tensor([0.5, 0.0])
y = torch.tensor([0.0, 0.5])
v = ball.logmap(x, y)
v_transp = ball.transp0(x, v)
print("v_transp:", v_transp)
v_transp_back = ball.transp(torch.zeros_like(x), x, v_transp)
print("v_transp_back:", v_transp_back)
print("v:", v)
