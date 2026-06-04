import torch
x = torch.tensor([1.0], requires_grad=True)
code1 = torch.tensor([1.1])
code2 = torch.tensor([0.2])

q1 = x + (code1 - x).detach()
r1 = x - q1

q2 = r1 + (code2 - r1).detach()
r2 = r1 - q2

out = q1 + q2
loss = out.sum()
loss.backward()
print("Gradient of x:", x.grad)
