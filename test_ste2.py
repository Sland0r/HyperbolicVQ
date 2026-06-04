import torch
def test_rvq(detach_residual=False):
    x = torch.tensor([1.0], requires_grad=True)
    code1 = torch.tensor([1.1])
    code2 = torch.tensor([0.2])

    q1 = x + (code1 - x).detach()
    r1 = x - q1
    if detach_residual:
        r1 = r1.detach()

    q2 = r1 + (code2 - r1).detach()
    r2 = r1 - q2

    out = q1 + q2
    loss = (out - 1.5)**2
    loss.backward()
    return x.grad.item()

print("Without detach:", test_rvq(False))
print("With detach:", test_rvq(True))
