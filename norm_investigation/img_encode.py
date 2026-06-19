"""GPU: mean L2 norm of the on-ball encoder output (r0) over the image test set.
Computes one value per distinct image checkpoint dir. Writes results_img.json."""
import os, sys, json, glob
import torch
from torchvision import datasets, transforms
sys.path.insert(0, 'norm_investigation')
sys.path.insert(0, 'egs/MNIST_VQVAE')
import mapping as M
from measure import resolve, read_config, to_ball
from mnist_vqvae import VQVAE2D

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# distinct dirs from both image tables
dirs = sorted(set(list(M.img_recon.values()) + list(M.img_abl.values())))

_loaders = {}
def get_val(dataset, data_dir):
    key = dataset
    if key in _loaders: return _loaders[key]
    tf = transforms.ToTensor()
    if dataset == 'mnist':
        vd = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
        ch, sz = 1, 28
    elif dataset == 'cifar100':
        vd = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tf)
        ch, sz = 3, 32
    else:
        raise ValueError(dataset)
    dl = torch.utils.data.DataLoader(vd, batch_size=256, shuffle=False, num_workers=4)
    _loaders[key] = (dl, ch, sz)
    return _loaders[key]

MODELSIZE = {'mnist': 'small', 'cifar100': 'large', 'emnist': 'medium'}
FLAGS = ['new_method','approx','hste','hste_riemannian','hste_clip','gradient_correction',
         'block_hste_pt','A5','A4_v2','a6','a7','a6_1','a7_1','a8','ema','kmeans_init',
         'uniform','exponential_lambda','dot_product_weight','entailment_cone_weight',
         'gyration_weight','codebook_weight','commitment_weight','threshold_ema_dead_code','full_grid']

out = {}
for d in dirs:
    full = M.ckdir('image', d)
    ck, cf = resolve(full, 'image')
    cfg = read_config(cf)
    ds = cfg['dataset']; c = float(cfg['c'])
    dl, ch, sz = get_val(ds, cfg['data_dir'])
    kw = {k: cfg[k] for k in FLAGS if k in cfg}
    model = VQVAE2D(D=cfg['D'], n_q=cfg['n_q'], bins=cfg['bins'], c=c,
                    in_channels=ch, img_size=sz, size=MODELSIZE[ds], **kw).to(dev)
    sd = torch.load(ck, map_location=dev, weights_only=False)['model']
    model.load_state_dict(sd, strict=False)
    model.eval()
    tot = 0.0; cnt = 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(dev)
            z = model.encoder(x)                 # (B, D, N)  tokens flattened
            z = z.permute(0, 2, 1).reshape(-1, z.shape[1])   # (B*N, D)
            r0 = to_ball(z, c)
            n = r0.norm(dim=-1)
            tot += float(n.sum()); cnt += n.numel()
    out[d] = {'enc_mean': tot / cnt, 'c': c, 'dataset': ds, 'ckpt': ck, 'n_vec': cnt}
    print(f"{d:40s} ds={ds} c={c} enc_mean={out[d]['enc_mean']:.4f}", flush=True)

json.dump(out, open('norm_investigation/results_img.json', 'w'), indent=1)
print("WROTE results_img.json")
