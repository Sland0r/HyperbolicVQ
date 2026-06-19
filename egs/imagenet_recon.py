"""Print reconstructions from the Euclidean and hyperbolic ImageNet RQ-VAE checkpoints.

Loads a fixed set of ImageNet val images, runs them through both trained RQVAE
generators, and saves a comparison grid: row 0 = originals, row 1 = Euclidean
recon, row 2 = hyperbolic recon.
"""
import os, sys, glob
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

sys.path.insert(0, "/home/acolombo/VAEs")
from egs.imagenet_rqvae_bench import RQVAE

VAL_DIR = "/scratch-shared/acolombo/imagenet/ILSVRC/Data/CLS-LOC/val"
EUC_CKPT = "/home/acolombo/VAEs/checkpoint/imagenet_rqvae_euc/latest.pth"
HYP_CKPT = "/home/acolombo/VAEs/checkpoint/imagenet_rqvae_hyp/latest.pth"
OUT = "/home/acolombo/VAEs/figures/imagenet_recon_euc_vs_hyp.png"
N = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
])

# Fixed, evenly-spaced val images for a reproducible panel.
all_imgs = sorted(glob.glob(os.path.join(VAL_DIR, "*.JPEG")))
picks = [all_imgs[i] for i in range(0, N * 500, 500)][:N]
print("images:", [os.path.basename(p) for p in picks], flush=True)
x = torch.stack([tfm(Image.open(p).convert("RGB")) for p in picks]).to(device)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    print(f"{ckpt_path}: c={a['c']} n_q={a['n_q']} bins={a['bins']} dim={a['dim']} "
          f"gstep={ckpt.get('global_step')}", flush=True)
    g = RQVAE(n_q=a["n_q"], bins=a["bins"], dim=a["dim"], c=a["c"]).to(device)
    g.load_state_dict(ckpt["g"])
    g.eval()
    return g


rows = [x.cpu()]
for tag, path in [("euclidean", EUC_CKPT), ("hyperbolic", HYP_CKPT)]:
    g = load_model(path)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x_hat, _ = g(x)
    x_hat = x_hat.float().clamp(-1, 1).cpu()
    rec_l1 = F.l1_loss(x_hat, x.cpu()).item()
    print(f"  {tag} recon L1 = {rec_l1:.4f}", flush=True)
    rows.append(x_hat)
    del g
    torch.cuda.empty_cache()

# rows: [orig, euc, hyp] each (N,3,256,256) in [-1,1]; lay out row-by-row.
grid = torch.cat(rows, dim=0)
grid = (grid + 1) / 2  # -> [0,1]
torchvision.utils.save_image(grid, OUT, nrow=N)
print("saved grid (rows: original / euclidean / hyperbolic) ->", OUT, flush=True)
