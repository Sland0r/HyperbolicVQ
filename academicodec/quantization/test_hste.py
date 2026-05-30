"""Numerical checks for the hyperbolic straight-through estimator.

Run:
    export PYTHONPATH="/home/acolombo/VAEs:${PYTHONPATH}"
    python academicodec/quantization/test_hste.py

Covers:
  1. parallel_transport_1 is the *exact* Poincaré-ball parallel transport:
     it matches geoopt's reference transport (an isometry that both rescales
     AND rotates), whereas the old parallel_transport — a pure conformal
     rescale — preserves the Riemannian norm but gets the *direction* wrong.
  2. HyperbolicSTE forward is identity (returns q) and feeds no gradient to
     the codebook (grad wrt q is None).
  3. HyperbolicSTE gradient wrt x converges to the plain Euclidean STE
     gradient as q -> x, and genuinely differs in *direction* once q != x
     (i.e. it is no longer the degenerate scalar rescale that the approximate
     transport produced).
"""

import torch

from academicodec.quantization.core_vq import (
    HyperbolicSTE,
    conformal_factor,
    parallel_transport,
    parallel_transport_1,
    project,
)


def riemannian_norm_sq(p, v, c):
    """Squared Poincaré-ball norm of tangent vector v at base point p."""
    return (conformal_factor(p, c) ** 2) * v.pow(2).sum(dim=-1, keepdim=True)


def euclidean_ste_grad(x, q, grad_output):
    """Gradient of the standard STE  x + (q - x).detach()  wrt x."""
    x = x.clone().detach().requires_grad_(True)
    out = x + (q - x).detach()
    out.backward(grad_output)
    return x.grad


def hste_grad(x, q, c, grad_output):
    """Gradient of HyperbolicSTE wrt x (and the None grad wrt q)."""
    x = x.clone().detach().requires_grad_(True)
    q = q.clone().detach().requires_grad_(True)
    out = HyperbolicSTE.apply(x, q, c)
    out.backward(grad_output)
    return x.grad, q.grad


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)  # tighter tolerances
    c = 1.3
    N, D = 64, 8

    # Points kept well inside the ball (radius 1/sqrt(c) ~= 0.877) so the
    # Poincaré ops are numerically clean; v is an arbitrary tangent vector.
    x = project(0.15 * torch.randn(N, D), c)
    y = project(0.15 * torch.randn(N, D), c)
    v = 0.3 * torch.randn(N, D)

    # ---- 1. exact PT vs reference (geoopt) -----------------------------
    pt_exact = parallel_transport_1(x, y, v, c)
    pt_approx = parallel_transport(x, y, v, c)

    # Both should preserve the Riemannian norm (the approx is a pure conformal
    # rescale, so it does too) -- report, don't discriminate on this.
    src = riemannian_norm_sq(x, v, c)
    print(f"[1] Riemannian-norm preservation max|Δ‖v‖²|: "
          f"exact={ (riemannian_norm_sq(y, pt_exact, c) - src).abs().max():.2e}  "
          f"approx={ (riemannian_norm_sq(y, pt_approx, c) - src).abs().max():.2e}")
    assert (riemannian_norm_sq(y, pt_exact, c) - src).abs().max() < 1e-8

    try:
        from geoopt.manifolds.stereographic.math import parallel_transport as g_pt
        ref = g_pt(x, y, v, k=torch.tensor(-c))  # our +c convention == geoopt k=-c
        exact_rel = (pt_exact - ref).norm() / ref.norm()
        approx_rel = (pt_approx - ref).norm() / ref.norm()
        print(f"    vs geoopt reference: exact rel.err={exact_rel:.3e}  "
              f"approx rel.err={approx_rel:.3e}")
        assert exact_rel < 1e-6, "parallel_transport_1 must match exact geoopt PT"
        assert approx_rel > 1e-2, "approx PT is expected to get the direction wrong"
    except ImportError:
        print("    (geoopt not available; skipped reference comparison)")

    # ---- 2. forward identity + no codebook gradient --------------------
    g = torch.randn(N, D)
    fwd = HyperbolicSTE.apply(x, y, c)
    assert torch.allclose(fwd, y), "HyperbolicSTE forward must return q unchanged"
    _, gq = hste_grad(x, y, c, g)
    assert gq is None, "HyperbolicSTE must not backprop into the codebook (q)"
    print("[2] forward is identity (returns q) and grad wrt q is None  ✓")

    # ---- 3. compare against Euclidean STE ------------------------------
    # As q -> x the hyperbolic STE must collapse to the Euclidean STE.
    print("[3] hste vs Euclidean STE grad as q -> x:")
    for eps in (1e-1, 1e-2, 1e-3, 1e-5):
        q = project(x + eps * torch.randn(N, D), c)
        ge = euclidean_ste_grad(x, q, g)
        gh, _ = hste_grad(x, q, c, g)
        rel = (gh - ge).norm() / ge.norm().clamp_min(1e-12)
        cos = (gh.flatten() @ ge.flatten()) / (
            gh.flatten().norm() * ge.flatten().norm()).clamp_min(1e-12)
        print(f"    eps={eps:<6g}  rel‖Δgrad‖={rel.item():.3e}  "
              f"cos(grad_h,grad_e)={cos.item():.6f}")

    # Tight limit: q == x exactly -> identical gradients.
    gh0, _ = hste_grad(x, x, c, g)
    ge0 = euclidean_ste_grad(x, x, g)
    assert torch.allclose(gh0, ge0, atol=1e-9), \
        "at q==x the hyperbolic STE must equal the Euclidean STE"
    print("    q==x: hste grad == Euclidean STE grad  ✓")

    # Far apart: the exact-PT gradient must actually differ in *direction*,
    # not just be a rescale of the Euclidean grad (which is what the old
    # approximate transport produced).
    q_far = project(x + 0.5 * torch.randn(N, D), c)
    gh_far, _ = hste_grad(x, q_far, c, g)
    ge_far = euclidean_ste_grad(x, q_far, g)
    gh_u = gh_far / gh_far.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ge_u = ge_far / ge_far.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dir_diff = (gh_u - ge_u).norm(dim=-1).max().item()
    print(f"    far q: max row direction diff = {dir_diff:.3e} "
          f"(>0 => genuine transport, not a scalar rescale)")
    assert dir_diff > 1e-3, \
        "exact-PT STE should rotate the gradient, not merely rescale it"

    print("\nAll HyperbolicSTE checks passed.")


if __name__ == "__main__":
    main()
