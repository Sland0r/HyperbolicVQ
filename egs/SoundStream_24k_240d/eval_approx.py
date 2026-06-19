"""On-ball residual (--approx) recompute at fixed full depth n_q=12.

Loads one checkpoint, runs encoder -> RVQ forward with approx=True at full
bandwidth, and reports the mean squared hyperbolic distance between the
recomposed code aggregate and the encoder output (0 = exact on the ball).
The block-level STE hop is training-only, so the eval forward value equals the
plain Mobius aggregate -- the number is a property of the residual aggregation,
not the gradient estimator. One invocation per model checkpoint.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import soundfile as sf

# reuse the exact model builder / arg surface from eval_rd
from eval_rd import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--out_json', required=True)
    p.add_argument('--valid_path', default='/home/acolombo/VAEs/dataset/LibriTTS/dev-clean')
    p.add_argument('--n_files', type=int, default=100)
    p.add_argument('--max_sec', type=float, default=10.0)
    p.add_argument('--sr', type=int, default=24000)
    p.add_argument('--bins', type=int, default=1024)
    p.add_argument('--ratios', type=int, nargs='+', default=[6, 5, 4, 2])
    p.add_argument('--target_bandwidths', type=float, nargs='+', default=[1, 2, 4, 8, 12])
    p.add_argument('--full_bw', type=float, default=12.0)  # n_q=12 at full depth
    p.add_argument('--c', type=float, default=0.0)
    p.add_argument('--codebook_dim', type=int, default=512)
    p.add_argument('--threshold_ema_dead_code', type=int, default=2)
    p.add_argument('--new_method', action='store_true')
    p.add_argument('--hste', action='store_true')
    p.add_argument('--hste_riemannian', action='store_true')
    p.add_argument('--gradient_correction', action='store_true')
    p.add_argument('--tangent_proj', action='store_true')
    p.add_argument('--encoder_scale', type=float, default=1.0)
    p.add_argument('--encoder_shell', type=float, default=0.0)
    p.add_argument('--code_max_radius', type=float, default=0.0)
    args = p.parse_args()

    if args.c <= 0.0:
        # Euclidean has no ball; residual is undefined. Emit an explicit marker.
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as fh:
            json.dump({'name': args.name, 'ckpt': args.ckpt, 'c': args.c,
                       'mean_residual': None, 'note': 'euclidean (no ball)'}, fh, indent=2)
        print(f'[{args.name}] euclidean -- residual undefined')
        return

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args)
    model.quantizer.approx = True  # turn on the on-ball residual diagnostic
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['soundstream'] if 'soundstream' in ck else ck
    model.load_state_dict(sd)
    model.to(dev).eval()
    frame_rate = model.frame_rate
    hop = int(np.prod(args.ratios))
    print(f'[{args.name}] loaded {args.ckpt} (epoch={ck.get("epoch")}) c={args.c} d={args.codebook_dim}')

    files = sorted(glob.glob(args.valid_path + '/**/*.wav', recursive=True))
    rng = np.random.RandomState(1234)
    rng.shuffle(files)

    dists = []
    n_done = 0
    with torch.no_grad():
        for f in files:
            if n_done >= args.n_files:
                break
            wav, sr = sf.read(f, dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(-1)
            if len(wav) < args.sr:
                continue
            wav = wav[: int(args.max_sec * args.sr)]
            T = (len(wav) // hop) * hop
            x = torch.from_numpy(wav[:T]).to(dev).view(1, 1, -1)

            e = model.encoder(x)
            e = model.pre_quant_bn(e)
            # force full depth (n_q=12); validation=True path returns distance
            _, _, _, _, distance = model.quantizer(e, frame_rate, args.full_bw, validation=True)
            dists.append(float(distance))
            n_done += 1
            if n_done % 20 == 0:
                print(f'  {n_done}/{args.n_files} files  running mean={np.mean(dists):.5f}', flush=True)

    mean_res = float(np.mean(dists))
    out = {'name': args.name, 'ckpt': args.ckpt, 'c': args.c,
           'codebook_dim': args.codebook_dim, 'n_files': n_done,
           'full_bw': args.full_bw, 'mean_residual': mean_res,
           'std_residual': float(np.std(dists))}
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'[{args.name}] mean on-ball residual (n_q=12) = {mean_res:.5f}  ->  {args.out_json}')


if __name__ == '__main__':
    main()
