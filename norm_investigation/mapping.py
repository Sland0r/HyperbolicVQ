"""Thesis-1 (results + ablations) experiment -> checkpoint mapping.

Each entry: (table, row_label, ckpt_dir, domain).
"one representative per row": seed 42 / standard cell, full grid only where the
table *is* a grid (tab:nlp_grid, tab:audio_dim). config.py in each dir is read
at runtime for curvature c and dims.
"""
import os

CK = "checkpoint"

# ---- NLP (WordNet) -------------------------------------------------------
NLP = "nlp_2"
# capacity grid: cells d8b128 d8b256 d16b64 d16b128 d16b256 d32b128 d32b256
nlp_grid = {
 "Euclidean":  ["23821261_euc_d8b128_s-1p0","23821262_euc_d8b256_s-1p0","23717785_euccw1d16b64_s-1p0",
                "23711830_euccw1_s-1p0","23821263_euc_d16b256_s-1p0","23717793_euccw1d32b128_s-1p0","23821264_euc_d32b256_s-1p0"],
 "Vanilla":    ["23821265_van_d8b128_s-1p0","23821266_van_d8b256_s-1p0","23717787_hypcw1d16b64_s-1p0",
                "23711047_hypcw1_s-1p0","23821267_van_d16b256_s-1p0","23717794_hypcw1d32b128_s-1p0","23821268_van_d32b256_s-1p0"],
 "A4 (no gc)": ["23824887_a4nogc_d8b128_s-1p0","23824888_a4nogc_d8b256_s-1p0","23717789_a4nogccw1d16b64_s-1p0",
                "23711829_a4nogccw1_s-1p0","23824889_a4nogc_d16b256_s-1p0","23717795_a4nogccw1d32b128_s-1p0","23824890_a4nogc_d32b256_s-1p0"],
 "A4 + gc":    ["23823320_a4gc_d8b128_s-1p0","23823321_a4gc_d8b256_s-1p0","23823322_a4gc_d16b64_s-1p0",
                "23710509_a4cw1_s-1p0","23823323_a4gc_d16b256_s-1p0","23823324_a4gc_d32b128_s-1p0","23823325_a4gc_d32b256_s-1p0"],
}
nlp_cells = ["d8/b128","d8/b256","d16/b64","d16/b128","d16/b256","d32/b128","d32/b256"]

# ablation routing family at d16/b128, cw=1.0
nlp_abl = {
 "Euclidean":        "23711830_euccw1_s-1p0",
 "Vanilla":          "23711047_hypcw1_s-1p0",
 "Vanilla + gc":     "23823905_vangc_d16b128_s-1p0",
 "Per-layer d-HSTE (A3)":"23819072_a3riem_cw1_s-1p0",
 "A3 + gc":          "23819074_a3gc_cw1_s-1p0",
 "A4 (no gc)":       "23711829_a4nogccw1_s-1p0",
 "A4 + gc":          "23710509_a4cw1_s-1p0",
 "Strict gc (A4 v2)":"23768934_strictgc_r1_s-1p0",
 "A5 (no gc)":       "23739271_a5cw1_s-1p0",
 "A5 + gc":          "23739270_a5gccw1_s-1p0",
 "Keep-first (a6)":  "23776347_a6_r1_s-1p0",
 "Keep-last (a7)":   "23776350_a7_r1_s-1p0",
 "a6.1 (A4(+)a6)":   "23784105_a6_1_r1_s-1p0",
 "a7.1 (A4(+)a7)":   "23784108_a7_1_r1_s-1p0",
 "Full-sum (a8)":    "23906813_a8_cw1_s-1p0",
}
# riemannian-discount ablation (cw1, no gc)
nlp_riem = {
 "A3 discount ON":  "23819072_a3riem_cw1_s-1p0",
 "A3 discount OFF": "23906816_a3noriem_cw1_s-1p0",
 "A4 discount ON":  "23711829_a4nogccw1_s-1p0",
 "A4 discount OFF": "23906815_a4noriem_cw1_s-1p0",
}

# ---- Recommendation (Amazon Beauty) -------------------------------------
REC = "rec_1"
rec_beauty = {  # tab:rec_beauty
 "Euclidean":  "23967147",
 "Vanilla":    "23967149",
 "A4 (+gc)":   "23967151",
}
rec_abl = {  # tab:abl_rec
 "A4 (+gc) ref":   "23967151",
 "A5":             "23935320",
 "Keep-first (a6)":"23908020",
 "Keep-last (a7)": "23908021",
 "Full-sum (a8)":  "23908022",
 "Strict gc (A4v2)":"23908023",
}

# ---- Image (MNIST / CIFAR-100), seed 42 ---------------------------------
img_recon = {  # tab:img_recon / tab:img_hier / tab:img_gen (shared quantizers)
 ("Euclidean","mnist"):   "mnist_new/euclidean/23519915",
 ("Vanilla","mnist"):     "mnist_new/c1/23519920",
 ("A4 (+gc)","mnist"):    "mnist_new/c1_blockpt/23698256",
 ("Euclidean","cifar"):   "cifar_new/euclidean/23519938",
 ("Vanilla","cifar"):     "cifar_new/c1/23519939",
 ("A4 (+gc)","cifar"):    "cifar_new/c1_blockpt/23698257",
}
img_abl = {  # tab:abl_image  (a8 router study) seed 42
 ("Euclidean","mnist"):"mnist_new/euclidean/23519915",
 ("Vanilla","mnist"):  "mnist_new/c1/23519920",
 ("A4 (+gc)","mnist"): "mnist_new/c1_blockpt/23698256",
 ("a6.1","mnist"):     "mnist_new/c1_a61_s42/23784111",
 ("a7.1","mnist"):     "mnist_new/c1_a71_s42/23784115",
 ("a8","mnist"):       "mnist_new/a8/23935317",
 ("Euclidean","cifar"):"cifar_new/euclidean/23519938",
 ("Vanilla","cifar"):  "cifar_new/c1/23519939",
 ("A4 (+gc)","cifar"): "cifar_new/c1_blockpt/23698257",
 ("a6.1","cifar"):     "cifar_new/c1_a61_s42/23784118",
 ("a7.1","cifar"):     "cifar_new/c1_a71_s42/23784121",
 ("a8","cifar"):       "cifar_new/a8/23924479",
}

# ---- Audio (SoundStream LibriTTS 24k) -----------------------------------
SS = "soundstream"
audio_d64 = {  # tab:audio_d64
 "Euclidean 3ep":   "euc_d64_3ep",
 "Euclidean 10ep":  "euc_d64_10ep",
 "Vanilla 3ep":     "hyp_std_d64_3ep",
 "A4 (riem+gc) 3ep":"blockptste_riem_gc_d64_3ep",
 "A4 (riem+gc) 10ep":"blockptste_riem_gc_d64_10ep",
}
audio_dim = {  # tab:audio_dim  (3 epochs)
 ("Euclidean","8"):"euc_d8_3ep", ("Euclidean","32"):"euc_d32_3ep",
 ("Euclidean","64"):"euc_d64_3ep", ("Euclidean","128"):"euc_d128_3ep",
 ("A4 (+gc)","8"):"blockptste_riem_gc_d8_3ep", ("A4 (+gc)","32"):"blockptste_riem_gc_d32_3ep",
 ("A4 (+gc)","64"):"blockptste_riem_gc_d64_3ep", ("A4 (+gc)","128"):"blockptste_riem_gc_d128_3ep",
 ("A4 (no gc)","8"):"blockptste_riem_nogc_d8_3ep", ("A4 (no gc)","32"):"blockptste_riem_nogc_d32_3ep",
 ("A4 (no gc)","64"):"blockptste_riem_nogc_d64_3ep", ("A4 (no gc)","128"):"blockptste_riem_nogc_d128_3ep",
}
audio_abl = {  # tab:abl_audio  (n_q=12, d64, 3ep)
 "A4 (riem+gc) ref":"blockptste_riem_gc_d64_3ep",
 "Keep-first (a6)": "a6_d64_3ep",
 "Keep-last (a7)":  "a7_d64_3ep",
 "Per-layer d-HSTE (A3)":"nmhste_riem_d64_3ep",
 "Full-sum (a8)":   "a8_d64_3ep",
 "Strict gc (A4v2)":"a4v2_d64_3ep",
}


def ckdir(domain, name):
    if domain == "nlp":  return os.path.join(CK, NLP, name)
    if domain == "rec":  return os.path.join(CK, REC, name)
    if domain == "image":return os.path.join(CK, name)
    if domain == "audio":return os.path.join(CK, SS, name)
    raise ValueError(domain)
