"""Boundary / large-displacement stress test for HyperbolicSTE.backward.

test_hste.py only checks the *well-inside-the-ball* regime (radius ~0.15R).
This script probes the two regimes the STE backward actually hits during
training of --new_method + --hste runs:

  (A) base points approaching the ball boundary  (c*||p||^2 -> 1)
  (B) large quantization displacement            (||q - x|| large / antipodal)

For each regime we compare HyperbolicSTE.backward against an INDEPENDENT
reference built from geoopt's exact parallel transport + the exact conformal
factors (the same definitional Riemannian-pullback math the STE implements):

    grad_r        = grad_output / lambda_q^2          # Euclidean -> Riemannian at q
    grad_r_at_x   = PT_{q->x}(grad_r)                 # transport q -> x
    grad_e_at_x   = grad_r_at_x * lambda_x^2          # Riemannian -> Euclidean at x

and we report the *magnitude amplification* lambda_x^2 / lambda_q^2 that the STE
applies, since that is the quantity that explodes as x -> boundary.

Run:
    export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
    python test_hste_boundary.py
"""

import math
import torch

from academicodec.quantization.core_vq import (
    HyperbolicSTE,
    conformal_factor,
    parallel_transport_1,
)

try:
    from geoopt.manifolds.stereographic.math import parallel_transport as geoopt_pt
    HAVE_GEOOPT = True
except ImportError:
    HAVE_GEOOPT = False


def hste_backward(x, q, c, grad_output):
    """Run exactly the path the autograd Function takes for grad wrt x."""
    x = x.clone().detach().requires_grad_(True)
    q = q.clone().detach()
    out = HyperbolicSTE.apply(x, q, c)
    out.backward(grad_output)
    return x.grad


def reference_backward(x, q, c, grad_output):
    """Independent reference: same Riemannian-pullback math via geoopt PT,
    exact (unclamped) conformal factors, float64."""
    lam_q = conformal_factor(q, c)           # 2 / (1 - c||q||^2)
    lam_x = conformal_factor(x, c)
    grad_r = grad_output / lam_q.pow(2)
    grad_r_at_x = geoopt_pt(q, x, grad_r, k=torch.tensor(-float(c), dtype=x.dtype))
    return grad_r_at_x * lam_x.pow(2)


def sample_at_radius(n, d, frac, c, generator):
    """n points whose Poincare radius is `frac` of the ball radius R=1/sqrt(c)."""
    R = 1.0 / math.sqrt(c)
    v = torch.randn(n, d, generator=generator, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    return v * (frac * R)


def amplification(x, q, c):
    """The scalar magnitude factor lambda_x^2 / lambda_q^2 applied by the STE."""
    lam_q = conformal_factor(q, c)
    lam_x = conformal_factor(x, c)
    return (lam_x.pow(2) / lam_q.pow(2)).squeeze(-1)


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def boundary_sweep(c, d=8, n=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    print(f"\n{'='*92}\n(A) BOUNDARY SWEEP  c={c}  (ball radius R={1/math.sqrt(c):.4f}, dim={d})\n{'='*92}")
    print(f"  q sits at 0.30R (fixed); x radius swept toward the boundary.")
    print(f"  grad_output ~ unit-norm rows.\n")
    print(f"  {'x_frac':>8} | {'||grad_x||_max':>15} | {'amp=λx²/λq² max':>17} | "
          f"{'PT1 relerr':>11} | {'HSTE vs ref':>12} | {'finite':>6}")
    print(f"  {'-'*8}-+-{'-'*15}-+-{'-'*17}-+-{'-'*11}-+-{'-'*12}-+-{'-'*6}")
    q = sample_at_radius(n, d, 0.30, c, g)
    for xf in (0.30, 0.60, 0.90, 0.99, 0.999, 0.9999, 0.99999):
        x = sample_at_radius(n, d, xf, c, g)
        go = torch.randn(n, d, generator=g, dtype=torch.float64)
        go = go / go.norm(dim=-1, keepdim=True)

        gh = hste_backward(x, q, c, go)
        amp = amplification(x, q, c)
        finite = torch.isfinite(gh).all().item()
        gmax = gh.norm(dim=-1).max().item()

        if HAVE_GEOOPT:
            # PT-only relative error (unit tangent vector transported q->x)
            v = torch.randn(n, d, generator=g, dtype=torch.float64)
            pt1 = parallel_transport_1(q, x, v, c)
            ptref = geoopt_pt(q, x, v, k=torch.tensor(-float(c)))
            pt_re = rel_err(pt1, ptref)
            ref = reference_backward(x, q, c, go)
            hste_re = rel_err(gh, ref)
            print(f"  {xf:>8} | {gmax:>15.3e} | {amp.max().item():>17.3e} | "
                  f"{pt_re:>11.2e} | {hste_re:>12.2e} | {str(bool(finite)):>6}")
        else:
            print(f"  {xf:>8} | {gmax:>15.3e} | {amp.max().item():>17.3e} | "
                  f"{'n/a':>11} | {'n/a':>12} | {str(bool(finite)):>6}")


def displacement_sweep(c, d=8, n=2048, seed=1):
    g = torch.Generator().manual_seed(seed)
    print(f"\n{'='*92}\n(B) DISPLACEMENT SWEEP  c={c}  (x fixed at 0.50R)\n{'='*92}")
    print(f"  q radius swept; direction RANDOM (so q and x can be near-antipodal -> large geodesic).\n")
    print(f"  {'q_frac':>8} | {'geo-dist max':>13} | {'||grad_x||_max':>15} | "
          f"{'PT1 relerr':>11} | {'HSTE vs ref':>12} | {'finite':>6}")
    print(f"  {'-'*8}-+-{'-'*13}-+-{'-'*15}-+-{'-'*11}-+-{'-'*12}-+-{'-'*6}")
    x = sample_at_radius(n, d, 0.50, c, g)
    for qf in (0.10, 0.50, 0.90, 0.99, 0.999, 0.9999):
        q = sample_at_radius(n, d, qf, c, g)
        go = torch.randn(n, d, generator=g, dtype=torch.float64)
        go = go / go.norm(dim=-1, keepdim=True)

        gh = hste_backward(x, q, c, go)
        finite = torch.isfinite(gh).all().item()
        gmax = gh.norm(dim=-1).max().item()

        # geodesic distance d_c(x,q) = (2/sqrt c) atanh( sqrt c * ||(-x) mobius+ q|| )
        # use a stable proxy via mobius: report max over batch
        with torch.no_grad():
            from academicodec.quantization.core_vq import mobius_add
            diff = mobius_add(-x, q, c)
            sc = math.sqrt(c)
            gd = (2 / sc) * torch.atanh((sc * diff.norm(dim=-1)).clamp_max(1 - 1e-12))
        if HAVE_GEOOPT:
            v = torch.randn(n, d, generator=g, dtype=torch.float64)
            pt1 = parallel_transport_1(q, x, v, c)
            ptref = geoopt_pt(q, x, v, k=torch.tensor(-float(c)))
            pt_re = rel_err(pt1, ptref)
            ref = reference_backward(x, q, c, go)
            hste_re = rel_err(gh, ref)
            print(f"  {qf:>8} | {gd.max().item():>13.3e} | {gmax:>15.3e} | "
                  f"{pt_re:>11.2e} | {hste_re:>12.2e} | {str(bool(finite)):>6}")
        else:
            print(f"  {qf:>8} | {gd.max().item():>13.3e} | {gmax:>15.3e} | "
                  f"{'n/a':>11} | {'n/a':>12} | {str(bool(finite)):>6}")


def main():
    torch.set_default_dtype(torch.float64)
    if not HAVE_GEOOPT:
        print("WARNING: geoopt not available -> reference columns skipped.")
    for c in (1.0, 0.5):   # 1.0 = NLP/rec default; 0.5 exercises the clamp_max(1-1e-5) edge
        boundary_sweep(c)
        displacement_sweep(c)
    print("\nLegend: amp is the magnitude factor the STE multiplies the gradient by; "
          "it blows up as x -> boundary even when PT is exact.")


if __name__ == "__main__":
    main()
