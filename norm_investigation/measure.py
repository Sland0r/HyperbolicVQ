"""Shared norm-measurement helpers (CPU)."""
import os, glob, re, math
import torch

# --- on-ball maps (replicated from core_vq, no geoopt import needed) ---
def exp_map0(v, c):
    norm = v.norm(dim=-1, keepdim=True)
    sc = c ** 0.5
    scale = torch.tanh(sc * norm) / (sc * norm.clamp_min(1e-5))
    return v * scale

def project(x, c, eps=1e-5):
    max_norm = (1.0 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.where(norm > max_norm, x * (max_norm / norm), x)

def to_ball(x, c):
    """tangent encoder output -> initial on-ball residual r0 (no shell/scale/proj:
    valid for nlp/rec/image defaults). For c==0 returns x unchanged."""
    if c and c > 0:
        return project(exp_map0(x, c), c)
    return x

# --- checkpoint resolution & loading ---
def resolve(d, domain):
    if domain in ('nlp', 'rec'):
        return os.path.join(d, 'model.pt'), os.path.join(d, 'config.py')
    if domain == 'image':
        ck = sorted(glob.glob(d + '/best_*.pth'), key=lambda p: int(re.findall(r'best_(\d+)', p)[0])) \
             or sorted(glob.glob(d + '/latest.pth'))
        return (ck[-1] if ck else None), os.path.join(d, 'config.py')
    if domain == 'audio':
        ck = sorted(glob.glob(d + '/best_*.pth') + glob.glob(d + '/*/best_*.pth'),
                    key=lambda p: int(re.findall(r'best_(\d+)', p)[0]))
        if not ck:
            ck = glob.glob(d + '/latest.pth') + glob.glob(d + '/*/latest.pth')
        cf = glob.glob(d + '/config.py') + glob.glob(d + '/*/config.py')
        return (ck[-1] if ck else None), (cf[0] if cf else None)

def read_config(cfg):
    d = {}
    for line in open(cfg):
        line = line.split('#')[0].strip()
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            try: d[m.group(1)] = eval(m.group(2), {}, {})
            except Exception: d[m.group(1)] = m.group(2)
    return d

def load_sd(ckpt, domain):
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    if domain == 'audio':
        return obj['soundstream']
    if isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        return obj['model']
    return obj

# --- codebook norms ---
def codebook_norms(sd):
    """Return list over layers of dict(mean,std,radius_frac) of L2 norm of embed rows.
    radius_frac only meaningful for hyperbolic (=norm, since 1/sqrt(c)=1 at c=1)."""
    layers = {}
    for k, v in sd.items():
        m = re.match(r'.*quantizer\.vq\.layers\.(\d+)\._codebook\.embed$', k)
        if m and hasattr(v, 'shape'):
            layers[int(m.group(1))] = v.float()
    out = []
    for i in sorted(layers):
        e = layers[i]
        n = e.norm(dim=-1)
        out.append({'mean': float(n.mean()), 'std': float(n.std()),
                    'min': float(n.min()), 'max': float(n.max()), 'bins': e.shape[0], 'dim': e.shape[1]})
    return out
