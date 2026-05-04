import torch

def mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-5)

def gyration(u, v, w, c):
    u_plus_v = mobius_add(u, v, c)
    v_plus_w = mobius_add(v, w, c)
    u_plus_v_plus_w = mobius_add(u, v_plus_w, c)
    return mobius_add(-u_plus_v, u_plus_v_plus_w, c)

c = torch.tensor(1.0)
q1 = torch.tensor([0.2, 0.3])
q2 = torch.tensor([0.1, -0.4])

# Suppose z is exactly q1 + q2
z = mobius_add(q1, q2, c)

# Raw residual
r_raw = mobius_add(-q1, z, c)
print("r_raw:", r_raw)
print("q2:", q2)
# r_raw and q2 should be identical because -q1 + (q1 + q2) = q2
# Let's verify
print("Diff:", (r_raw - q2).norm())

# Now suppose z is NOT exactly q1 + q2, but z = q1 + (q2 + r_2)
r2 = torch.tensor([-0.05, 0.05])
z = mobius_add(q1, mobius_add(q2, r2, c), c)

r_raw = mobius_add(-q1, z, c)  # this is q2 + r2
print("\nNew r_raw (should be q2 + r2):", r_raw)
print("q2 + r2:", mobius_add(q2, r2, c))

# If we define r_aligned = gyr[-q1, z] r_raw
r_aligned = gyration(-q1, z, r_raw, c)
print("\nr_aligned:", r_aligned)

# How to reconstruct r_raw from r_aligned without z?
# We need gyr[q1, -z] r_aligned
r_raw_recon = gyration(q1, -z, r_aligned, c)
print("r_raw_recon (with z):", r_raw_recon)

# Try approximating z with q1 + q2
z_approx = mobius_add(q1, q2, c)
r_raw_recon_approx = gyration(q1, -z_approx, r_aligned, c)
print("r_raw_recon_approx (with z_approx):", r_raw_recon_approx)

