"""Assemble final markdown report: per thesis table, experiment name,
per-codebook mean L2 norm, and mean encoding (r0) L2 norm over the eval set."""
import os, sys, json
sys.path.insert(0, 'norm_investigation')
import mapping as M

cpu = json.load(open('norm_investigation/results_cpu.json'))
img = json.load(open('norm_investigation/results_img.json')) if os.path.exists('norm_investigation/results_img.json') else {}
aud = json.load(open('norm_investigation/results_audio.json')) if os.path.exists('norm_investigation/results_audio.json') else {}

def cbcells(cb):
    return " / ".join(f"{l['mean']:.3f}" for l in cb) if cb else "—"

def enc_for(entry):
    if entry['enc_mean'] is not None:
        return f"{entry['enc_mean']:.3f}"
    # image/audio: look up by dir basename mapping
    d = entry['dir']
    # image keys are the relative subpath used in mapping; audio keys are dir name
    rel_img = d.replace('checkpoint/', '')
    if rel_img in img: return f"{img[rel_img]['enc_mean']:.3f}"
    base = os.path.basename(d)
    if base in aud: return f"{aud[base]['enc_mean']:.3f}"
    return "(pending)"

L = []
def emit(title, table_key, eval_label):
    rows = cpu.get(table_key, [])
    L.append(f"\n### {title}\n")
    L.append(f"*Eval set for encoding norm: {eval_label}. Codebook column = per-stage mean ‖embed‖ (stage 1→n_q).*\n")
    L.append("| Experiment | c | dim×bins | mean ‖encoding‖ | per-stage mean ‖codebook‖ |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        dim = r['dim'] if r['dim'] is not None else (r['codebook'][0]['dim'] if r['codebook'] else '?')
        bins = r['bins'] if r['bins'] is not None else (r['codebook'][0]['bins'] if r['codebook'] else '?')
        dimbins = f"{dim}×{bins}"
        label = r['label'].replace(' | ', ' · ')  # avoid literal '|' breaking the markdown table
        L.append(f"| {label} | {r['c']:.0f} | {dimbins} | {enc_for(r)} | {cbcells(r['codebook'])} |")

emit("Results — Table 6.1 WordNet capacity grid (tab:nlp_grid)", 'tab:nlp_grid', "all 82,115 synset embeddings (transductive)")
emit("Results — Table 6.2 Recommendation, Amazon Beauty (tab:rec_beauty)", 'tab:rec_beauty', "full Beauty item catalogue")
emit("Results — Table 6.3 Image reconstruction (tab:img_recon) [also used by 6.3 hierarchy & generation]", 'tab:img_recon', "MNIST/CIFAR-100 test set (10k imgs), all latent positions")
emit("Results — Table 6.4a Audio d64, 3/10 epochs (tab:audio_d64)", 'tab:audio_d64', "LibriTTS dev-clean (400 clips)")
emit("Results — Table 6.4b Audio dimension sweep (tab:audio_dim)", 'tab:audio_dim', "LibriTTS dev-clean (400 clips)")
emit("Ablation — Table 7.2 WordNet gradient-routing (tab:abl_nlp)", 'tab:abl_nlp', "all 82,115 synset embeddings")
emit("Ablation — Table 7.2 Riemannian-discount (tab:abl_riem)", 'tab:abl_riem', "all 82,115 synset embeddings")
emit("Ablation — Table 7.3 Recommendation routers (tab:abl_rec)", 'tab:abl_rec', "full Beauty item catalogue")
emit("Ablation — Table 7.3 Audio routers (tab:abl_audio)", 'tab:abl_audio', "LibriTTS dev-clean (400 clips)")
emit("Ablation — Table 7.3 Image full-sum router (tab:abl_image)", 'tab:abl_image', "MNIST/CIFAR-100 test set")

open('norm_investigation/REPORT.md', 'w').write("\n".join(L))
print("\n".join(L))
