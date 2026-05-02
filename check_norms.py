import torch
import sys

# Check codebook norms from saved checkpoints
for job_id in ["22421827", "22421828"]:
    path = f"/home/acolombo/VAEs/checkpoint/soundstream/{job_id}/latest.pth"
    try:
        state = torch.load(path, map_location="cpu")
        print(f"\n=== Job {job_id} ===")
        for key, val in state.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if "embed" in k and "avg" not in k and torch.is_tensor(v):
                        norm = v.norm(dim=-1)
                        print(f"  {k}: shape={v.shape}, norm min={norm.min():.4f}, max={norm.max():.4f}, mean={norm.mean():.4f}")
    except Exception as e:
        print(f"  {job_id}: {e}")
