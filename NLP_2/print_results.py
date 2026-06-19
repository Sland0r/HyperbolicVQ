import os

# Define the runs to gather
run_groups = {
    'standard --new+hste': [
        '23362361_m-new_hste_s-1p0_c1.0_k4_b128_h16_closure',
        '23362362_m-new_hste_s-0p3_c1.0_k4_b128_h16_closure',
        '23362363_m-new_hste_s-0p1_c1.0_k4_b128_h16_closure',
        '23362364_m-new_hste_s-0p05_c1.0_k4_b128_h16_closure',
    ],
    '+ --hste_riemannian': [
        '23365119_m-new_hste_s-1p0_c1.0_k4_b128_h16_closure_riem',
        '23365120_m-new_hste_s-0p3_c1.0_k4_b128_h16_closure_riem',
        '23365121_m-new_hste_s-0p1_c1.0_k4_b128_h16_closure_riem',
        '23365122_m-new_hste_s-0p05_c1.0_k4_b128_h16_closure_riem',
    ],
    '+ --gradient_correction': [
        '23651588_m-new_hste_s-1p0_c1.0_k4_b128_h16_closure_gc',
        '23651589_m-new_hste_s-0p3_c1.0_k4_b128_h16_closure_gc',
        '23651590_m-new_hste_s-0p1_c1.0_k4_b128_h16_closure_gc',
        '23651591_m-new_hste_s-0p05_c1.0_k4_b128_h16_closure_gc',
    ],
    '+ --block_hste_pt + --hste_riem + --grad_corr': [
        '23697575_block_pt_s-1p0',
        '23697576_block_pt_s-0p3',
        '23697577_block_pt_s-0p1',
        '23697578_block_pt_s-0p05',
    ]
}

base_dir = "/home/acolombo/VAEs/checkpoint/nlp_2"

results = {group: {} for group in run_groups}

for group, run_ids in run_groups.items():
    for run_id in run_ids:
        path = os.path.join(base_dir, run_id)
        config_file = os.path.join(path, "config.py")
        logs_file = os.path.join(path, "logs.txt")
        
        scale = None
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                for line in f:
                    if line.startswith("embed_init_scale"):
                        scale = line.split("=")[1].strip()
                        break
        
        recall = None
        if os.path.exists(logs_file):
            with open(logs_file, "r") as f:
                for line in f:
                    if "Recall@10:" in line:
                        recall = line.split(":")[1].strip().replace('%', '')
                        break
                        
        if scale and recall:
            results[group][float(scale)] = recall

print("### New Experiments Results Table")
print("| init scale | standard `--new+hste` | + `--hste_riemannian` | + `--gradient_correction` | + `--block_hste_pt` + riem + gc |")
print("|---|---|---|---|---|")

scales = [1.0, 0.3, 0.1, 0.05]
for scale in scales:
    row = [str(scale)]
    for group in run_groups:
        row.append(results[group].get(scale, "-"))
    print("| " + " | ".join(row) + " |")
