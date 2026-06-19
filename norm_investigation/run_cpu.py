"""CPU pass: codebook norms for ALL mapped checkpoints + NLP/rec encoding norms.
Writes norm_investigation/results_cpu.json."""
import os, sys, json
import torch, torch.nn as nn
sys.path.insert(0, 'norm_investigation')
import mapping as M
from measure import resolve, read_config, load_sd, codebook_norms, to_ball

RES = {}  # key -> {label, dir, c, codebook:[...], enc_mean:float|None}

def do_codebook(domain, name, label, table, enc=None):
    d = M.ckdir(domain, name)
    ck, cf = resolve(d, domain)
    cfg = read_config(cf) if cf and os.path.exists(cf) else {}
    c = float(cfg.get('c', 0.0))
    sd = load_sd(ck, domain)
    cb = codebook_norms(sd)
    entry = {'label': label, 'table': table, 'dir': d, 'ckpt': ck,
             'c': c, 'dim': cfg.get('embed_dim', cfg.get('D')), 'bins': cfg.get('bins'),
             'codebook': cb, 'enc_mean': None}
    if enc == 'nlp':
        emb = sd['embedding.weight'].float()
        r0 = to_ball(emb, c)
        entry['enc_mean'] = float(r0.norm(dim=-1).mean())
        entry['enc_n'] = emb.shape[0]
    elif enc == 'rec':
        entry['enc_mean'] = rec_enc_norm(sd, c)
    RES.setdefault(table, []).append(entry)
    return entry

# --- rec encoder forward over Beauty catalog ---
_REC_EMB = None
def beauty_embeddings():
    global _REC_EMB
    if _REC_EMB is None:
        from rec_1.amazon_dataset import prepare_data
        _, _, _, _, ef = prepare_data('Beauty')
        _REC_EMB = torch.load(ef, map_location='cpu').float()
    return _REC_EMB

def rec_enc_norm(sd, c):
    enc = nn.Sequential(nn.Linear(768,512), nn.ReLU(), nn.Linear(512,256), nn.ReLU(),
                        nn.Linear(256,128), nn.ReLU(), nn.Linear(128,32))
    with torch.no_grad():
        enc[0].weight.copy_(sd['encoder.0.weight']); enc[0].bias.copy_(sd['encoder.0.bias'])
        enc[2].weight.copy_(sd['encoder.2.weight']); enc[2].bias.copy_(sd['encoder.2.bias'])
        enc[4].weight.copy_(sd['encoder.4.weight']); enc[4].bias.copy_(sd['encoder.4.bias'])
        enc[6].weight.copy_(sd['encoder.6.weight']); enc[6].bias.copy_(sd['encoder.6.bias'])
        x = beauty_embeddings()
        z = enc(x)
        r0 = to_ball(z, c)
        return float(r0.norm(dim=-1).mean())

# ---------------- run all tables ----------------
for cfg_name, cells in M.nlp_grid.items():
    for cell, name in zip(M.nlp_cells, cells):
        do_codebook('nlp', name, f"{cfg_name} | {cell}", 'tab:nlp_grid', enc='nlp')
for k, v in M.nlp_abl.items():  do_codebook('nlp', v, k, 'tab:abl_nlp', enc='nlp')
for k, v in M.nlp_riem.items(): do_codebook('nlp', v, k, 'tab:abl_riem', enc='nlp')
for k, v in M.rec_beauty.items(): do_codebook('rec', v, k, 'tab:rec_beauty', enc='rec')
for k, v in M.rec_abl.items():    do_codebook('rec', v, k, 'tab:abl_rec', enc='rec')
# image / audio: codebook norms only here (encoding via GPU jobs)
for (cfg_name, ds), v in M.img_recon.items(): do_codebook('image', v, f"{cfg_name} | {ds}", 'tab:img_recon')
for (cfg_name, ds), v in M.img_abl.items():   do_codebook('image', v, f"{cfg_name} | {ds}", 'tab:abl_image')
for k, v in M.audio_d64.items():  do_codebook('audio', v, k, 'tab:audio_d64')
for (cfg_name, dim), v in M.audio_dim.items(): do_codebook('audio', v, f"{cfg_name} | d{dim}", 'tab:audio_dim')
for k, v in M.audio_abl.items():  do_codebook('audio', v, k, 'tab:abl_audio')

json.dump(RES, open('norm_investigation/results_cpu.json','w'), indent=1)
n = sum(len(x) for x in RES.values())
print(f"done: {n} checkpoints across {len(RES)} tables")
for t, rows in RES.items():
    print(f"  {t}: {len(rows)} rows; enc_mean set on {sum(r['enc_mean'] is not None for r in rows)}")
