"""Rate-distortion evaluation: sweep RVQ depth (n_q = 1,2,4,8,12) on a fixed
dev-clean subset and compute, per operating point:

  rate:        nominal kbps = frame_rate * n_q * log2(bins) / 1000
               entropy kbps = frame_rate * sum_i H(code_i) / 1000  (empirical)
  distortion:  PESQ-wb (24k->16k resample), STOI, SI-SDR, log-mel L1 distance
  bottleneck:  perplexity + unique codes per layer (at full depth)

Encode once at full bandwidth, decode prefixes codes[:n_q] (greedy RVQ).
Writes a JSON with all numbers. One invocation per model checkpoint.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import soundfile as sf
import torchaudio
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn

from academicodec.models.encodec.net3 import SoundStream


def build_model(args):
    return SoundStream(n_filters=32, D=512,
                       target_bandwidths=args.target_bandwidths,
                       exponential_lambda=0.0, uniform=True,
                       threshold_ema_dead_code=args.threshold_ema_dead_code,
                       codebook_weight=1.0, commitment_weight=0.25,
                       dot_product_weight=0.0, entailment_cone_weight=0.0,
                       gyration_weight=0.0, ratios=args.ratios, decay=0.99,
                       sample_rate=args.sr, bins=args.bins, c=args.c, ema=False,
                       kmeans_init=False, pre_quant_batchnorm=False, remove=0,
                       codebook_dim=args.codebook_dim, new_method=args.new_method,
                       approx=False, hste=args.hste, hste_riemannian=args.hste_riemannian,
                       gradient_correction=args.gradient_correction,
                       encoder_scale=args.encoder_scale, encoder_shell=args.encoder_shell,
                       code_max_radius=args.code_max_radius, embed_init_scale=1.0,
                       tangent_proj=args.tangent_proj)


def si_sdr(ref, est, eps=1e-8):
    ref = ref - ref.mean()
    est = est - est.mean()
    s = (est * ref).sum() / (ref.pow(2).sum() + eps) * ref
    return float(10 * torch.log10(s.pow(2).sum() / ((est - s).pow(2).sum() + eps)))


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
    p.add_argument('--nq_sweep', type=int, nargs='+', default=[1, 2, 4, 8, 12])
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

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['soundstream'] if 'soundstream' in ck else ck
    model.load_state_dict(sd)
    model.to(dev).eval()
    frame_rate = model.frame_rate
    hop = int(np.prod(args.ratios))
    print(f'[{args.name}] loaded {args.ckpt} (epoch={ck.get("epoch")}) frame_rate={frame_rate}')

    files = sorted(glob.glob(args.valid_path + '/**/*.wav', recursive=True))
    rng = np.random.RandomState(1234)
    rng.shuffle(files)

    resamp16 = torchaudio.transforms.Resample(args.sr, 16000).to(dev)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr, n_fft=1024, hop_length=256, n_mels=64).to(dev)

    max_nq = max(args.nq_sweep)
    bits_per_cb = math.log2(args.bins)
    # accumulators
    metrics = {nq: {'pesq': [], 'stoi': [], 'sisdr': [], 'meldist': []} for nq in args.nq_sweep}
    hist = torch.zeros(max_nq, args.bins, dtype=torch.float64)
    n_codes_total = 0
    n_done = 0

    with torch.no_grad():
        for f in files:
            if n_done >= args.n_files:
                break
            wav, sr = sf.read(f, dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(-1)
            if len(wav) < args.sr:  # skip < 1 s
                continue
            wav = wav[: int(args.max_sec * args.sr)]
            T = (len(wav) // hop) * hop
            x = torch.from_numpy(wav[:T]).to(dev).view(1, 1, -1)

            codes = model.encode(x, target_bw=float(max_nq))  # (max_nq, 1, N)
            n = codes.shape[-1]
            n_codes_total += n
            for q in range(codes.shape[0]):
                hist[q].scatter_add_(0, codes[q].flatten().cpu().to(torch.int64),
                                     torch.ones(n, dtype=torch.float64))

            ref = x.flatten()
            ref16 = resamp16(ref.unsqueeze(0)).squeeze(0).cpu().numpy()
            mel_ref = torch.log(melspec(ref) + 1e-7)

            for nq in args.nq_sweep:
                est = model.decode(codes[:nq]).flatten()
                L = min(len(est), len(ref))
                e, r = est[:L], ref[:L]
                est16 = resamp16(e.unsqueeze(0)).squeeze(0).cpu().numpy()
                L16 = min(len(est16), len(ref16))
                try:
                    pq = pesq_fn(16000, ref16[:L16], est16[:L16], 'wb')
                except Exception:
                    pq = float('nan')
                st = stoi_fn(r.cpu().numpy(), e.cpu().numpy(), args.sr, extended=False)
                sd_ = si_sdr(r, e)
                md = float((torch.log(melspec(e) + 1e-7) - mel_ref[..., :]).abs().mean()
                           if mel_ref.shape == torch.log(melspec(e) + 1e-7).shape
                           else (torch.log(melspec(e) + 1e-7)[..., :mel_ref.shape[-1]]
                                 - mel_ref[..., :melspec(e).shape[-1]]).abs().mean())
                m = metrics[nq]
                m['pesq'].append(pq); m['stoi'].append(float(st))
                m['sisdr'].append(sd_); m['meldist'].append(md)
            n_done += 1
            if n_done % 20 == 0:
                print(f'  {n_done}/{args.n_files} files', flush=True)

    # per-layer entropy (bits) and perplexity
    probs = (hist / hist.sum(dim=1, keepdim=True).clamp_min(1)).numpy()
    H = [float(-(p[p > 0] * np.log2(p[p > 0])).sum()) for p in probs]
    ppl = [float(2 ** h) for h in H]
    uniq = [int((hist[q] > 0).sum()) for q in range(max_nq)]

    out = {'name': args.name, 'ckpt': args.ckpt, 'n_files': n_done,
           'frame_rate': frame_rate, 'bins': args.bins,
           'entropy_bits_per_layer': H, 'ppl_per_layer': ppl, 'unique_per_layer': uniq,
           'points': []}
    for nq in args.nq_sweep:
        m = metrics[nq]
        pesq_a = [v for v in m['pesq'] if not math.isnan(v)]
        out['points'].append({
            'n_q': nq,
            'kbps_nominal': frame_rate * nq * bits_per_cb / 1000,
            'kbps_entropy': frame_rate * sum(H[:nq]) / 1000,
            'pesq': float(np.mean(pesq_a)) if pesq_a else None,
            'stoi': float(np.mean(m['stoi'])),
            'sisdr': float(np.mean(m['sisdr'])),
            'meldist': float(np.mean(m['meldist'])),
        })
        pt = out['points'][-1]
        print(f"  n_q={nq:2d}  {pt['kbps_nominal']:5.1f} kbps (entropy {pt['kbps_entropy']:5.2f})"
              f"  PESQ={pt['pesq'] if pt['pesq'] else float('nan'):.3f}  STOI={pt['stoi']:.3f}"
              f"  SI-SDR={pt['sisdr']:6.2f}  melL1={pt['meldist']:.4f}")

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'[{args.name}] wrote {args.out_json}')


if __name__ == '__main__':
    main()
