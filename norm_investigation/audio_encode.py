"""GPU: mean L2 norm of the on-ball initial residual r0 over LibriTTS dev-clean.
Uses the *real* instantiated quantizer (project_in + restored _enc_scale buffer),
so it is faithful to the tangent_proj / encoder_scale=-1 audio pipeline.
One value per distinct audio checkpoint dir. Writes results_audio.json."""
import os, sys, json, glob, random
import torch
import soundfile as sf
sys.path.insert(0, 'norm_investigation')
import mapping as M
from measure import resolve, read_config, exp_map0, project
from academicodec.models.encodec.net3 import SoundStream

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MAXLEN = 24000
NFILES = 400

dirs = sorted(set(list(M.audio_d64.values()) + list(M.audio_dim.values()) + list(M.audio_abl.values())))

# fixed deterministic sample of dev-clean files
_files = None
def dev_files(path):
    global _files
    if _files is None:
        fs = sorted(glob.glob(path + '/**/*.wav', recursive=True))
        random.Random(0).shuffle(fs)
        _files = fs[:NFILES]
    return _files

def load_wav(fp):
    data, sr = sf.read(fp, dtype='float32')
    w = torch.from_numpy(data)
    if w.ndim > 1: w = w.mean(-1)
    if w.numel() >= MAXLEN: w = w[:MAXLEN]
    else: w = torch.nn.functional.pad(w, (0, MAXLEN - w.numel()))
    return w  # (MAXLEN,)

def r0_norms(model, wav):
    rvq = model.quantizer.vq
    c = rvq.c
    e = model.encoder(wav)              # (B, 512, T)
    x = e.permute(0, 2, 1)              # (B, T, 512)
    if c and c > 0:
        if rvq.requires_projection and rvq.tangent_proj:
            x = rvq.project_in(x)
        x = rvq._shape_tangent(x)
        r0 = project(exp_map0(x, c), c)
    else:
        r0 = rvq.project_in(x) if rvq.requires_projection else x
    return r0.reshape(-1, r0.shape[-1]).norm(dim=-1)

CTOR_DEFAULTS = dict(a8=False, target_max_recon=0.0, encoder_scale_ema=0.0,
                     a6=False, a7=False, a6_1=False, a7_1=False, A5=False, A4_v2=False,
                     gyration_only=False, block_hste=False, hste_clip=False, remove=0,
                     pre_quant_batchnorm=False)
PASS = ['target_bandwidths','exponential_lambda','uniform','threshold_ema_dead_code',
        'codebook_weight','commitment_weight','dot_product_weight','entailment_cone_weight',
        'gyration_weight','ratios','decay','bins','c','ema','kmeans_init','pre_quant_batchnorm',
        'remove','codebook_dim','new_method','approx','hste','hste_riemannian','hste_clip',
        'gyration_only','block_hste','block_hste_pt','gradient_correction','A5','A4_v2',
        'a6','a7','a6_1','a7_1','a8','encoder_scale','encoder_scale_ema','encoder_shell',
        'code_max_radius','target_max_recon','embed_init_scale','tangent_proj']

def main():
  out = {}
  for d in dirs:
    full = M.ckdir('audio', d)
    ck, cf = resolve(full, 'audio')
    cfg = read_config(cf)
    kw = dict(CTOR_DEFAULTS)
    for k in PASS:
        if k in cfg: kw[k] = cfg[k]
    c = float(cfg['c'])
    model = SoundStream(n_filters=32, D=512, sample_rate=int(cfg.get('sr', 24000)), **kw).to(dev)
    sd = torch.load(ck, map_location=dev, weights_only=False)['soundstream']
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    files = dev_files(cfg['valid_data_path'])
    tot = 0.0; cnt = 0
    with torch.no_grad():
        B = 16
        for i in range(0, len(files), B):
            wavs = torch.stack([load_wav(f) for f in files[i:i+B]]).unsqueeze(1).to(dev)  # (B,1,L)
            n = r0_norms(model, wavs)
            tot += float(n.sum()); cnt += n.numel()
    out[d] = {'enc_mean': tot / cnt, 'c': c, 'codebook_dim': cfg.get('codebook_dim'),
              'tangent_proj': cfg.get('tangent_proj'), 'ckpt': ck, 'n_vec': cnt,
              'missing_keys': len(miss)}
    print(f"{d:34s} c={c} cbdim={cfg.get('codebook_dim')} tproj={cfg.get('tangent_proj')} "
          f"enc_mean={out[d]['enc_mean']:.4f} (miss={len(miss)})", flush=True)

  json.dump(out, open('norm_investigation/results_audio.json', 'w'), indent=1)
  print("WROTE results_audio.json")

if __name__ == '__main__':
    main()
