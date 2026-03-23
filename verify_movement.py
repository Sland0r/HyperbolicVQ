import torch
import sys

path = "/home/acolombo/VAEs/checkpoint/soundstream/latest.pth"
try:
    ckpt = torch.load(path, map_location='cpu')
except Exception as e:
    print(f"Could not load {path}: {e}")
    sys.exit(1)

model_state = ckpt['soundstream']

# Try to find identical codebook at layer 0
codebook_key = "quantizer.vq.layers.0._codebook.embed"
if codebook_key in model_state:
    embed = model_state[codebook_key]
    print(f"Loaded embed with shape: {embed.shape}")
    print(f"First 5 elements of row 0: {embed[0, :5]}")
    
    # Are all columns identical to their init?
    # Kaiming uniform init has max/min bounds, we can't easily recreate, but we can look for "exact" identical weights 
    # across two different codebooks to see if they updated.
    
    codebook_1 = model_state.get("quantizer.vq.layers.1._codebook.embed")
    if codebook_1 is not None:
        print(f"Are layer 0 and layer 1 identical? {torch.allclose(embed, codebook_1)}")
else:
    print(f"Key {codebook_key} not found. Available keys starting with 'quantizer':")
    for k in model_state.keys():
        if k.startswith("quantizer"):
            print(k)

