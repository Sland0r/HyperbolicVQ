import os

run_groups = {
    'euclidean': [
        '23671904_m-none_s-1p0_c0.0_k4_b128_h16_closure_t4euc',
        '23671908_m-none_s-0p3_c0.0_k4_b128_h16_closure_t4euc',
        '23671912_m-none_s-0p1_c0.0_k4_b128_h16_closure_t4euc',
        '23671918_m-none_s-0p05_c0.0_k4_b128_h16_closure_t4euc',
    ],
    'vanilla_hyp': [
        '23671905_m-none_s-1p0_c1.0_k4_b128_h16_closure_t4van',
        '23671909_m-none_s-0p3_c1.0_k4_b128_h16_closure_t4van',
        '23671913_m-none_s-0p1_c1.0_k4_b128_h16_closure_t4van',
        '23671919_m-none_s-0p05_c1.0_k4_b128_h16_closure_t4van',
    ],
    'nm+hste+riem': [
        '23671906_m-new_hste_s-1p0_c1.0_k4_b128_h16_closure_t4riem',
        '23671910_m-new_hste_s-0p3_c1.0_k4_b128_h16_closure_t4riem',
        '23671914_m-new_hste_s-0p1_c1.0_k4_b128_h16_closure_t4riem',
        '23671920_m-new_hste_s-0p05_c1.0_k4_b128_h16_closure_t4riem',
    ],
    'nm+hste+gc': [
        '23671907_m-new_hste_s-1p0_c1.0_k4_b128_h16_closure_t4gc',
        '23671911_m-new_hste_s-0p3_c1.0_k4_b128_h16_closure_t4gc',
        '23671917_m-new_hste_s-0p1_c1.0_k4_b128_h16_closure_t4gc',
        '23671923_m-new_hste_s-0p05_c1.0_k4_b128_h16_closure_t4gc',
    ],
    'block_pt+riem+gc': [
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

print("### Four-Method Head-to-Head + New Block_PT")
print("| init scale | euclidean | vanilla hyp | nm+hste+riem | nm+hste+gc | block_pt+riem+gc |")
print("|---|---|---|---|---|---|")

scales = [1.0, 0.3, 0.1, 0.05]
for scale in scales:
    row = [str(scale)]
    for group in run_groups:
        row.append(results[group].get(scale, "-"))
    print("| " + " | ".join(row) + " |")
