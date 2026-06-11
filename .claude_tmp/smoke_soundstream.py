import sys, torch
sys.path.insert(0, '/gpfs/home4/acolombo/VAEs')
from academicodec.models.encodec.net3 import SoundStream

torch.manual_seed(0)
model = SoundStream(
    n_filters=32, D=512, ratios=[6, 5, 4, 2], sample_rate=24000, bins=1024,
    target_bandwidths=[1, 2, 4, 8, 12], c=1.0, ema=True, kmeans_init=False,
    codebook_dim=64, new_method=True, block_hste_pt=True, encoder_scale=-1,
    tangent_proj=True, code_max_radius=0.9, uniform=True)
model.train()
x = torch.randn(2, 1, 7200)  # 0.3 s @ 24 kHz, multiple of hop 240
out = model(x)
# net3 forward returns recon + quantization info; backprop recon + commit
print("forward outputs:", [type(o) for o in out] if isinstance(out, tuple) else type(out))
recon = out[0] if isinstance(out, tuple) else out
loss = torch.nn.functional.mse_loss(recon, x)
if isinstance(out, tuple):
    for o in out[1:]:
        if torch.is_tensor(o) and o.requires_grad and o.dim() <= 1:
            loss = loss + o.sum()
loss.backward()
enc_grads = [p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None]
print("recon shape:", recon.shape, "| loss:", round(loss.item(), 4))
print("encoder params with grad: %d/%d, total grad norm %.4f" % (
    len(enc_grads), len(list(model.encoder.parameters())),
    sum(g**2 for g in enc_grads) ** 0.5))
nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
print("any NaN grads:", nan)
