# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core vector quantization implementation."""
import math
import typing as tp

import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torch import nn
import geoopt
from hypll.manifolds.poincare_ball import PoincareBall as HypllPoincareBall
from hypll.manifolds.poincare_ball import Curvature
from hypll.nn.modules.linear import HLinear
from hypll.tensors.manifold_tensor import ManifoldTensor

# geoopt API changed between versions; provide a lightweight fallback for
# parallel transport used in several code paths. This is an approximation
# based on conformal factor scaling and avoids depending on internal
# geoopt modules which may not be present in all versions.
def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Approximate parallel transport of tangent vector v from base point x to y.

    Uses the ratio of conformal factors λ_c^x / λ_c^y as a simple scaling.
    This is not exact parallel transport on the Poincaré ball but is a
    stable, numerically-safe fallback that preserves magnitude ordering and
    avoids hard dependency on geoopt internals.
    """
    lam_x = conformal_factor(x, c)
    lam_y = conformal_factor(y, c)
    # Avoid division by zero
    ratio = (lam_x / lam_y).clamp(min=1e-6)
    return v * ratio


def parallel_transport_1(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Exact parallel transport of tangent vector v from base point x to y.

    On the Poincaré ball the parallel transport along the geodesic from x to y is

        PT_{x->y}(v) = (λ_c^x / λ_c^y) · gyr[y, ⊖x] v

    where ⊖x = -x, λ is the conformal factor and gyr is the gyration operator.
    Unlike ``parallel_transport`` (a magnitude-only conformal rescale), this is a
    true isometry of the tangent spaces: it preserves the Riemannian norm
    λ_c^x ||v|| and rotates the vector via gyration.

    The gyration is evaluated with the simplified closed form (matching geoopt's
    ``_gyration`` under the sign map k = -c, since our Möbius denominator is
    1 + 2c<u,v> + c²||u||²||v||²). The Möbius-based ``gyration`` helper is *not*
    reused here: it composes several ``mobius_add`` calls whose ``clamp_min``
    corrupts the result when a base point sits near the ball boundary or when v
    is a large tangent vector (||v|| outside the ball) — exactly the regime the
    STE backward pass hits. This closed form is a single clamped division and
    stays an exact isometry there. Needs no geoopt internals.
    """
    gyr = gyration_transport(x, y, v, c)

    lam_x = conformal_factor(x, c)
    lam_y = conformal_factor(y, c)
    ratio = (lam_x / lam_y).clamp(min=1e-6)
    return ratio * gyr


def gyration_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """gyr[y, ⊖x] v — the rotation part of PT_{x->y}, WITHOUT the conformal
    λ_x/λ_y coefficient. Gyrations are orthogonal maps, so this preserves the
    EUCLIDEAN norm of v exactly (used by the --gyration_only STE backward)."""
    diff = y - x
    x_sq = x.pow(2).sum(dim=-1, keepdim=True)
    y_sq = y.pow(2).sum(dim=-1, keepdim=True)
    xw = (x * v).sum(dim=-1, keepdim=True)
    yw = (y * v).sum(dim=-1, keepdim=True)
    diff_w = yw - xw
    x_diff = (x * diff).sum(dim=-1, keepdim=True)
    diff_sq = diff.pow(2).sum(dim=-1, keepdim=True)

    c2 = c * c

    # Numerator terms
    a = -c2 * yw * x_sq - c * xw + 2 * c2 * (x_sq + x_diff) * xw
    b = c2 * xw * y_sq - c * yw
    a_plus_b = a + b

    # Stabilized difference avoiding catastrophic cancellation when y is close to x
    a_minus_b = c * (1 - c * x_sq) * diff_w - c2 * xw * diff_sq

    # Algebraically equivalent to a * y + b * (-x) but numerically stable
    num = 0.5 * a_plus_b * diff + 0.5 * a_minus_b * (y + x)

    # Stabilized denominator avoiding cancellation
    one_minus_c_x_sq = 1 - c * x_sq
    d = one_minus_c_x_sq.pow(2) - 2 * c * one_minus_c_x_sq * x_diff + c2 * x_sq * diff_sq
    d = d.clamp_min(1e-15)

    return v + 2 * num / d


import sys
sys.path.insert(0, '/home/acolombo/music')
from hyp_modules import HyperbolicEntailmentConeLoss

from academicodec.quantization.distrib import broadcast_tensors


def check_nan(x, msg):
    if torch.is_tensor(x) and torch.isnan(x).any():
        print(f"NaN DETECTED: {msg}", flush=True)
        import sys
        sys.exit(1) # Stop immediately so we can see the trace and print
    return x


def assert_finite(x: tp.Any, name: str):
    if not torch.is_tensor(x):
        return x
    if torch.isfinite(x).all():
        return x

    nan_count = torch.isnan(x).sum().item()
    posinf_count = torch.isposinf(x).sum().item()
    neginf_count = torch.isneginf(x).sum().item()
    finite_mask = torch.isfinite(x)
    bad_idx = (~finite_mask).nonzero(as_tuple=False)
    first_bad = bad_idx[0].tolist() if bad_idx.numel() > 0 else []
    finite_vals = x[finite_mask]
    finite_min = finite_vals.min().item() if finite_vals.numel() > 0 else float("nan")
    finite_max = finite_vals.max().item() if finite_vals.numel() > 0 else float("nan")
    raise RuntimeError(
        "Non-finite tensor detected at "
        f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, "
        f"first_bad_index={first_bad}, finite_min={finite_min:.6g}, finite_max={finite_max:.6g}"
    )

def mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True) # "mobius_add x2"
    y2 = y.pow(2).sum(dim=-1, keepdim=True) # "mobius_add y2"
    xy = (x * y).sum(dim=-1, keepdim=True) # "mobius_add xy"
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y # "mobius_add num"
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2 # "mobius_add denom"
    return num / denom.clamp_min(1e-5) # "mobius_add result"

def mobius_sub(x, y, c):
    return mobius_add(x, -y, c)

def hyperbolic_distance_sq(x, y, c):
    m_add = mobius_sub(x, y, c) # "hyperbolic_distance_sq m_add"
    norm = m_add.norm(dim=-1, keepdim=True).clamp_min(1e-5) # "hyperbolic_distance_sq norm"
    sqrt_c = c ** 0.5
    arg = (sqrt_c * norm).clamp(min=0.0, max=1 - 1e-3) # "hyperbolic_distance_sq arg"
    dist = (2 / sqrt_c) * torch.atanh(arg) # "hyperbolic_distance_sq dist"
    return dist.pow(2) # "hyperbolic_distance_sq result"

def pairwise_hyperbolic_distance_sq(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True) # "pairwise_hyperbolic_distance_sq x2"
    y2 = y.pow(2).sum(dim=-1, keepdim=True) # "pairwise_hyperbolic_distance_sq y2"
    xy = x @ y.t() # "pairwise_hyperbolic_distance_sq xy"
    sq_dist = (x2 + y2.t() - 2 * xy).clamp_min(0.0) # "pairwise_hyperbolic_distance_sq sq_dist"
    denom = ((1 - c * x2) @ (1 - c * y2).t()).clamp_min(1e-6) # "pairwise_hyperbolic_distance_sq denom"
    arg = 1 + 2 * c * sq_dist / denom # "pairwise_hyperbolic_distance_sq arg"
    dist = (1 / (c ** 0.5)) * torch.acosh(arg.clamp_min(1.0 + 1e-5)) # "pairwise_hyperbolic_distance_sq dist"
    return dist.pow(2) # "pairwise_hyperbolic_distance_sq result"

def exp_map0(v, c):
    norm = v.norm(dim=-1, keepdim=True) # "exp_map0 norm"
    sqrt_c = c ** 0.5
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm.clamp_min(1e-5)) # "exp_map0 scale"
    return v * scale # "exp_map0 result"

def log_map0(y, c):
    norm = y.norm(dim=-1, keepdim=True) # "log_map0 norm"
    sqrt_c = c ** 0.5
    scale = torch.atanh((sqrt_c * norm).clamp_max(1 - 1e-5)) / (sqrt_c * norm.clamp_min(1e-5)) # "log_map0 scale"
    return y * scale # "log_map0 result"

def project(x, c, eps=1e-5):
    """Project x onto the open Poincaré ball of radius 1/sqrt(c)."""
    max_norm = (1.0 - eps) / (c ** 0.5) 
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15) # "project norm"
    return torch.where(norm > max_norm, x * (max_norm / norm), x) # "project result"

def exp_map(x, v, c):
    cx2 = (c * x.pow(2).sum(dim=-1, keepdim=True)).clamp_max(1 - 1e-5) # "exp_map c*x2"
    lambda_x = 2 / (1 - cx2) # "exp_map lambda_x"
    return project(mobius_add(x, exp_map0(lambda_x * v / 2, c), c), c) # "exp_map result"

def log_map(x, y, c):
    cx2 = (c * x.pow(2).sum(dim=-1, keepdim=True)).clamp_max(1 - 1e-5) # "log_map c*x2"
    lambda_x = 2 / (1 - cx2) # "log_map lambda_x"
    return log_map0(mobius_add(-x, y, c), c) * 2 / lambda_x # "log_map result"


def conformal_factor(x, c):
    """Conformal factor λ_c^x = 2 / (1 - c ||x||^2)."""
    # Clamp c||x||^2 (not the raw ||x||^2): the ball is c||x||^2 < 1, so for
    # c != 1 clamping ||x||^2 at 1 either wrongly caps valid points (c<1) or
    # never protects (c>1). Curvature-aware clamp is correct for all c.
    cx2 = (c * x.pow(2).sum(dim=-1, keepdim=True)).clamp_max(1 - 1e-5)
    return 2.0 / (1.0 - cx2)



def gyration(u, v, w, c):
    """
    Computes gyr[u, v]w = -(u ⊕_c v) ⊕_c (u ⊕_c (v ⊕_c w))
    """
    u_plus_v = mobius_add(u, v, c)
    v_plus_w = mobius_add(v, w, c)
    u_plus_v_plus_w = mobius_add(u, v_plus_w, c)
    return mobius_add(-u_plus_v, u_plus_v_plus_w, c)


def weighted_midpoint_op(x, w, c):
    """Weighted midpoint operation [x, w]_c (Eq. 43).
    [x, w]_c = w * λ_c^x * x / (1 + sqrt(1 + c * w^2 * (λ_c^x)^2 * ||x||^2))
    """
    lam = conformal_factor(x, c)            # (... , 1)
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)  # (... , 1)
    num = w * lam * x
    denom = 1.0 + torch.sqrt((1.0 + c * w**2 * lam**2 * x_sq_norm).clamp_min(1e-10))
    return num / denom

def einstein_midpoint(z, w, c):
    """Einstein midpoint of points z with indicator weights w (Eq. 41).
    μ = (1/2) ⊗_c ( Σ w_i λ_c^{z_i} z_i / Σ |w_i| (λ_c^{z_i} - 1) )
    Args:
        z: (N, D) points on the Poincaré ball
        w: (N, K) one-hot assignment weights (w_ij = 1 if z_i -> c_j)
        c: curvature
    Returns:
        (K, D) midpoints, one per centroid
    """
    lam = conformal_factor(z, c)  # (N, 1)
    # Numerator: Σ_i w_ij * λ_c^{z_i} * z_i  for each centroid j
    weighted_z = lam * z          # (N, D) element-wise
    # w.T is (K, N), weighted_z is (N, D)
    num = w.t() @ weighted_z      # (K, D)
    # Denominator: Σ_i |w_ij| * (λ_c^{z_i} - 1)  for each centroid j
    den = w.t() @ (lam - 1.0)     # (K, 1)
    den = den.clamp_min(1e-8)
    # The argument to the half-Möbius scaling: num / den
    v = num / den                 # (K, D)
    # (1/2) ⊗_c v  =  exp_map0( (1/2) * log_map0(v, c), c )
    # But for the Poincaré ball, s ⊗_c v = tanh(s * atanh(√c ||v||)) / (√c ||v||) * v
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half_scaled = torch.tanh(0.5 * torch.atanh((sqrt_c * v_norm).clamp_max(1 - 1e-5))) / (sqrt_c * v_norm)
    mu = half_scaled * v
    return project(mu, c)

class HyperbolicSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, q, c, riemannian=False, gyration_only=False):
        ctx.save_for_backward(x, q)
        ctx.c = c
        ctx.riemannian = riemannian
        ctx.gyration_only = gyration_only
        return q

    @staticmethod
    def backward(ctx, grad_output):
        x, q = ctx.saved_tensors
        c = ctx.c

        if ctx.gyration_only:
            # Pure gyration transport: rotate the gradient from q to x the way the
            # geometry dictates, but drop ALL conformal (lambda/"gamma") coefficients
            # — no /lambda_q^2 conversion, no PT lambda_q/lambda_x ratio, no
            # *lambda_x^2 reconversion. Gyrations are orthogonal, so the Euclidean
            # gradient magnitude is preserved exactly (like the Euclidean STE, which
            # trains); only the direction is hyperbolically corrected.
            return (gyration_transport(q, x, grad_output, c),
                    None, None, None, None)

        # conformal factors (clamp c*||.||^2, not the raw norm — correct for all c)
        cq2 = (c * q.pow(2).sum(dim=-1, keepdim=True)).clamp_max(1 - 1e-5)
        cx2 = (c * x.pow(2).sum(dim=-1, keepdim=True)).clamp_max(1 - 1e-5)
        lambda_q = 2 / (1 - cq2)
        lambda_x_ = 2 / (1 - cx2)

        # Euclidean -> Riemannian
        grad_r = grad_output / (lambda_q ** 2)
        # transport tangent vector (exact Poincaré-ball PT, an isometry)
        grad_r_at_x = parallel_transport_1(
            q, x,
            grad_r,
            c
        )
        if ctx.riemannian:
            # Geometry-exact discount: return the *Riemannian* gradient at x
            # (skip the Riemannian->Euclidean ×λ_x² re-conversion that explodes
            # near the boundary). This is what a Riemannian optimizer consumes;
            # the conformal amplification is cancelled at the source.
            grad_x = grad_r_at_x
        else:
            # Riemannian -> Euclidean (standard STE; ×λ_x² amplifies near boundary)
            grad_x = grad_r_at_x * (lambda_x_ ** 2)

        return grad_x, None, None, None, None

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    if val == 0:
        return d
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    #assert_finite(moving_avg, "ema_inplace/moving_avg(before)")
    #assert_finite(new, "ema_inplace/new")
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    #assert_finite(moving_avg, "ema_inplace/moving_avg(after)")


def laplace_smoothing(x, n_categories: int, epsilon: float=1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int=10, c: float=0.):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if c > 0:
            dists = -pairwise_hyperbolic_distance_sq(samples, means, c)
        else:
            diffs = rearrange(samples, "n d -> n () d") - rearrange(means,
                                                                    "c d -> () c d")
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if c > 0:
            new_means = project(new_means, c)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
            self,
            dim: int,
            codebook_size: int,
            kmeans_init: int=False,
            kmeans_iters: int=10,
            decay: float=0.99,
            epsilon: float=1e-5,
            threshold_ema_dead_code: int=2,
            c: float=0.,
            ema: bool=True,
            gyration_weight: float=0.,
            code_max_radius: float=0.,
            embed_init_scale: float=1.0, ):
        super().__init__()
        self.c = c
        self.decay = decay
        self.ema = ema
        self.gyration_weight = gyration_weight
        # If >0 (c>0 only), cap codebook embeddings at this fraction of the ball
        # radius. Keeps codes off the boundary, where hyperbolic_distance_sq's
        # atanh clamp saturates and zeroes the commit/codebook-loss gradients.
        self.code_max_radius = code_max_radius
        init_fn: tp.Union[
            tp.Callable[..., torch.Tensor],
            tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        # Scale the random init toward the origin (no-op for kmeans_init, which
        # starts from zeros). Keeps codes away from the boundary regime where
        # hyperbolic_distance_sq's atanh clamp saturates.
        if embed_init_scale != 1.0:
            embed = embed * embed_init_scale

        # if not kmeans_init:
        #     # Normalize random init to zero-mean, unit-variance
        #     embed = (embed - embed.mean()) / embed.std().clamp_min(1e-5)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        if self.c > 0:
            if not kmeans_init:
                # Codes are direct Poincare points. Raw high-dim kaiming rows have
                # norm ~sqrt(2), so project() would pin EVERY code on the boundary
                # (radius ~1), where hyperbolic_distance_sq's atanh clamp saturates
                # and zeroes the commit/codebook-loss gradient. Instead place codes
                # in the interior: keep the (diverse) kaiming directions but spread
                # the radii uniformly in (0, r_max], with r_max set by code_max_radius.
                r_max = self.code_max_radius if self.code_max_radius > 0 else 0.5
                r_max = min(r_max, 1 - 1e-3)
                directions = embed / embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                radii = torch.rand(codebook_size, 1) * (r_max / (self.c ** 0.5))
                embed = directions * radii
            # Ensure random initialization is on the manifold when k-means init is disabled.
            embed = project(embed, self.c)

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        if not self.ema:
            if self.c > 0:
                self.embed = geoopt.ManifoldParameter(embed, manifold=geoopt.PoincareBall(c=self.c))
            else:
                self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def _project_embed_inplace_(self):
        if self.c <= 0:
            return
        with torch.no_grad():
            e = project(self.embed.data, self.c)
            if self.code_max_radius > 0:
                max_norm = self.code_max_radius / (self.c ** 0.5)
                norm = e.norm(dim=-1, keepdim=True)
                e = e * (max_norm / norm.clamp_min(1e-8)).clamp_max(1.0)
            self.embed.data.copy_(e)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size,
                                     self.kmeans_iters, self.c)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())
        if not self.ema:
            # Also sync the codebook embeddings which are nn.Parameter when not using EMA
            broadcast_tensors([self.embed])

    def replace_(self, samples, mask):
        #assert_finite(samples, "replace_/samples")
        #assert_finite(self.embed, "replace_/embed(before)")
        #also add some noise
        samples = samples + torch.randn_like(samples) * 0.01
        modified_codebook = torch.where(
            mask[..., None], # true when codebook is dead
            sample_vectors(samples, self.codebook_size), self.embed)
        if self.c > 0:
            modified_codebook = project(modified_codebook, self.c)
        #assert_finite(modified_codebook, "replace_/modified_codebook")
        self.embed.data.copy_(modified_codebook)
        #assert_finite(self.embed, "replace_/embed(after)")

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        #assert_finite(self.cluster_size, "expire_codes_/cluster_size(before)")
        #assert_finite(batch_samples, "expire_codes_/batch_samples(before)")
        expired_codes = self.cluster_size < self.threshold_ema_dead_code # number of clusters = codebook size
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d") # likely not necessary, already in that form
        #assert_finite(batch_samples, "expire_codes_/batch_samples(flat)")
        self.replace_(batch_samples, mask=expired_codes)
        #assert_finite(self.embed, "expire_codes_/embed(after replace)")
        broadcast_tensors(self.buffers())
        #assert_finite(self.embed, "expire_codes_/embed(after broadcast)")

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        if self.c > 0:
            dist = -pairwise_hyperbolic_distance_sq(x, self.embed, self.c)
            if self.gyration_weight > 0:
                sqrt_c = self.c ** 0.5
                x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                e_norm = self.embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                cos_alpha = (x @ self.embed.t()) / (x_norm @ e_norm.t())
                cos_alpha = cos_alpha.clamp(-1 + 1e-6, 1 - 1e-6)
                sin_alpha = (1 - cos_alpha.pow(2)).clamp_min(0).sqrt()
                x_poincare_norm = (2.0 / sqrt_c) * torch.atanh((sqrt_c * x_norm).clamp_max(1 - 1e-5))
                e_poincare_norm = (2.0 / sqrt_c) * torch.atanh((sqrt_c * e_norm).clamp_max(1 - 1e-5))
                penalty = self.c * (x_poincare_norm @ e_poincare_norm.t()) * sin_alpha
                dist = dist - self.gyration_weight * penalty
        else:
            embed = self.embed.t()
            dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed +
                     embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        self._project_embed_inplace_()
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        self._project_embed_inplace_()
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        self._project_embed_inplace_()
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # (everything, dim)
        #assert_finite(x, "EuclideanCodebook.forward/x(preprocess)")

        self.init_embed_(x)
        #assert_finite(self.embed, "EuclideanCodebook.forward/embed(after init)")

        embed_ind = self.quantize(x) # indices of the closest centroid
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape) # back to normal shape
        quantize = self.dequantize(embed_ind) # quantized x
        #assert_finite(quantize, "EuclideanCodebook.forward/quantize")

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            #if self.ema:
            self.expire_codes_(x) # move unused codes close to random samples
            #assert_finite(self.embed, "EuclideanCodebook.forward/embed(after expire)")
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            #assert_finite(self.cluster_size, "EuclideanCodebook.forward/cluster_size(after ema)")

            if not self.ema:
                # Skip EMA: codebook is nn.Parameter, updated via optimizer
                # TODO: might add reset for dead codes here
                pass
            elif self.c > 0:
                # Einstein midpoint EMA update (Eq. 41-43)
                # 1. Compute Einstein midpoint μ_j of assigned samples (Eq. 41)
                with torch.no_grad():
                    mu = einstein_midpoint(x, embed_onehot, self.c)
                    # 2. Weighted midpoint EMA (Eq. 42):
                    #    c_j^{t+1} = proj( [c_j, β]_c  ⊕_c  [μ_j, 1-β]_c )
                    #    where β = decay
                    old_part = weighted_midpoint_op(self.embed, self.decay, self.c)
                    new_part = weighted_midpoint_op(mu, 1.0 - self.decay, self.c)
                    embed_normalized = project(mobius_add(old_part, new_part, self.c), self.c)
                self.embed.data.copy_(embed_normalized)

            else:
                embed_sum = x.t() @ embed_onehot
                #assert_finite(embed_sum, "EuclideanCodebook.forward/embed_sum")
                ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size,
                                      self.epsilon) * self.cluster_size.sum())
                #assert_finite(cluster_size, "EuclideanCodebook.forward/cluster_size(smoothed)")
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                #assert_finite(embed_normalized, "EuclideanCodebook.forward/embed_normalized(euclidean)")
                self.embed.data.copy_(embed_normalized)
                #assert_finite(self.embed, "EuclideanCodebook.forward/embed(after update)")

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
            self,
            dim: int,
            codebook_size: int,
            codebook_dim: tp.Optional[int]=None,
            decay: float=0.99,
            epsilon: float=1e-5,
            kmeans_init: bool=True,
            kmeans_iters: int=50,
            threshold_ema_dead_code: int=2,
            codebook_weight: float=1.0,
            commitment_weight: float=0.25,
            c: float=0.,
            remove: int=0,
            ema: bool=False,
            hste: bool=False,
            hste_riemannian: bool=False,
            gyration_only: bool=False,
            block_hste: bool=False,
            block_hste_pt: bool=False,
            gyration_weight: float=0.,
            code_max_radius: float=0.,
            embed_init_scale: float=1.0, ):
        super().__init__()
        self.c = c
        self.ema = ema
        self.hste = hste
        self.hste_riemannian = hste_riemannian
        self.gyration_only = gyration_only
        # Block-level STE (--block_hste / --block_hste_pt): the per-layer quantize is
        # returned fully DETACHED (codebooks learn from the codebook loss only, as
        # with the other STEs); the encoder's through-quantizer gradient is provided
        # once, at block level, by ResidualVectorQuantization.forward (identity STE
        # in tangent space for block_hste, one HSTE transport on the ball for
        # block_hste_pt).
        self.block_ste = block_hste or block_hste_pt

        _codebook_dim: int = default(codebook_dim, dim)

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            c=c,
            ema=ema,
            gyration_weight=gyration_weight,
            code_max_radius=code_max_radius,
            embed_init_scale=embed_init_scale)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def forward(self, x): # quantizes x, computes loss depending on distance to codes, properly propagates gradients
        device = x.device
        quantize, embed_ind = self._codebook(x)

        # Save pre-STE quantize (has gradient path to embed) for codebook loss
        quantize_raw = quantize

        if self.training:
            # if self.c > 0:
            #     diff = mobius_sub(quantize, x, self.c)
            #     quantize = project(mobius_add(x, diff.detach(), self.c), self.c)
            # else:
            if self.block_ste:
                quantize = quantize_raw.detach()
            elif self.hste:
                quantize = HyperbolicSTE.apply(x, quantize, self.c, self.hste_riemannian,
                                               self.gyration_only)
            else:
                quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.c > 0:
                    commit_loss = hyperbolic_distance_sq(quantize.detach(), x, self.c).mean()
                else:
                    commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if not self.ema:
                # Codebook loss: drive codebook embeddings toward residuals
                # Use quantize_raw (pre-STE) so gradients flow to embed
                if self.c > 0:
                    codebook_loss = hyperbolic_distance_sq(x.detach(), quantize_raw, self.c).mean()
                else:
                    codebook_loss = F.mse_loss(x.detach(), quantize_raw)
                loss = loss + codebook_loss * self.codebook_weight

        return quantize, embed_ind, loss

class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.c = kwargs.get("c", 0.0)
        self.dot_product_weight = kwargs.pop("dot_product_weight", 0.0)
        self.entailment_cone_weight = kwargs.pop("entailment_cone_weight", 0.0)
        self.gyration_weight = kwargs.get("gyration_weight", 0.0)
        if self.entailment_cone_weight > 0 and self.c > 0:
            self.entailment_cone_loss_fn = HyperbolicEntailmentConeLoss(K=0.1, c=self.c)
        self.new_method = kwargs.pop("new_method", True)
        self.gradient_correction = kwargs.pop("gradient_correction", False)
        # Shaping of the tangent vector before exp_map0 (c>0 only). Both push
        # residuals off the origin so the nearest-code argmax discriminates by
        # direction; at the origin all same-radius codes are ~equidistant -> collapse.
        #   encoder_scale: constant multiplier (the encoder can absorb this).
        #   encoder_shell > 0: L2-normalise each vector so exp_map0 lands on a fixed
        #     ball-radius = encoder_shell/sqrt(c). Removes the magnitude DOF so the
        #     encoder cannot collapse residuals to the origin. Takes precedence.
        self.encoder_scale = kwargs.pop("encoder_scale", 1.0)
        self.encoder_shell = kwargs.pop("encoder_shell", 0.0)
        # When True (c>0 only), the codebook_dim bottleneck is a EUCLIDEAN nn.Linear
        # applied in tangent space BEFORE exp_map0 (and after log_map0 on decode),
        # instead of a hyperbolic HLinear on the ball. So exp_map, the codebooks, and
        # all quantization happen in the low-dim Poincare ball; no hyperbolic layers.
        self.tangent_proj = kwargs.pop("tangent_proj", False)
        # Block-level tangent STE (--block_hste): one straight-through wrapping the
        # whole RVQ block, out = x_tan + (log_map0(mobius_sum) - x_tan).detach().
        # The decoder gradient reaches the encoder with factor exactly 1: no
        # per-layer hops, hence no sum-over-paths of conformal factors at all.
        # Per-layer codes are detached (see VectorQuantization.block_ste); commit
        # losses keep their direct differentiable path to the encoder.
        self.block_hste = kwargs.get("block_hste", False)
        # Block-level PT-STE (--block_hste_pt): like --block_hste, but the single
        # block-level hop is a HyperbolicSTE transport on the ball, from the full
        # Möbius sum Q back to the initial residual r0, applied BEFORE project_out /
        # log_map0. Since Möbius left translations are isometries and (new_method)
        # r0 = L_{q1}...L_{qN}(r_N) while Q = L_{q1}...L_{qN}(0), the STE lie
        # d(Q, r0) = d(r_N, 0) — the final quantization error, the same size as a
        # per-layer STE at the last layer, paid in ONE transport between two macro
        # points (endpoint conformal ratio lambda_r0/lambda_Q, no compounding).
        # The hste_riemannian / gyration_only variants apply to the hop.
        self.block_hste_pt = kwargs.get("block_hste_pt", False)
        if self.block_hste and self.block_hste_pt:
            raise ValueError("--block_hste and --block_hste_pt are mutually exclusive")
        self.hste_riemannian = kwargs.get("hste_riemannian", False)
        self.gyration_only = kwargs.get("gyration_only", False)
        # First-batch tangent-norm auto-calibration (active when encoder_scale<=0):
        # a single global scale is fitted once so the median residual lands at
        # _enc_target, then frozen in a buffer (saved/restored with the model).
        # Unlike encoder_shell this preserves per-vector magnitude variation.
        self._enc_target = 0.5
        self.register_buffer("_enc_scale", torch.ones(1))
        self.register_buffer("_enc_calibrated", torch.zeros(1))
        # --- optional per-quantizer diagnostics (off by default; no hot-path cost) ---
        # When self.diag is True, forward() records, per residual layer, the max
        # Poincaré radius fraction sqrt(c)*||.|| (1.0 == ball boundary) of the
        # residual fed in and the quantized output, and registers a backward hook
        # capturing the gradient norm arriving at each layer's residual input.
        # Used to test the "boundary -> grad explosion" hypothesis during training.
        self.diag = False
        self.diag_data = None
        self.layers = nn.ModuleList()
        for i in range(num_quantizers):
            layer_kwargs = kwargs.copy()
            self.layers.append(VectorQuantization(**layer_kwargs))
        self.remove = kwargs.get("remove", 0)
        dim = kwargs.get("dim", 256)
        codebook_dim = kwargs.get("codebook_dim", dim)
        _codebook_dim: int = default(codebook_dim, dim)

        self.requires_projection = _codebook_dim != dim
        if self.block_hste and self.requires_projection and not self.tangent_proj:
            raise ValueError(
                "--block_hste needs the encoder tangent and the quantized output in "
                "the same (tangent) space: use --tangent_proj with codebook_dim != "
                "dimension, or codebook_dim == dimension.")
        if self.requires_projection and self.c > 0 and not self.tangent_proj:
            hyp_manifold = HypllPoincareBall(c=Curvature(self.c))
            self.project_in = HLinear(dim, _codebook_dim, manifold=hyp_manifold, bias=True)
            self.project_out = HLinear(_codebook_dim, dim, manifold=hyp_manifold, bias=True)
        elif self.requires_projection:
            # Euclidean projection. For c>0 + tangent_proj this is applied in tangent
            # space (before exp_map0 / after log_map0); for c==0 it is the ordinary
            # pre-quantization bottleneck.
            self.project_in = nn.Linear(dim, _codebook_dim)
            self.project_out = nn.Linear(_codebook_dim, dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

    # ---- diagnostics (only active when self.diag is True) ------------------
    def _diag_reset(self, n_q):
        self.diag_data = {
            'res_frac': [float('nan')] * n_q,   # max sqrt(c)*||residual_in||  (1.0 == boundary)
            'q_frac':   [float('nan')] * n_q,   # max sqrt(c)*||quantized||
            'grad_in':  [float('nan')] * n_q,   # ||grad|| arriving at residual_in (filled in backward)
        }

    def _diag_radius(self, t):
        # t: (b, n, d) on the Poincaré ball; report max radius fraction over batch
        sqrt_c = self.c ** 0.5
        with torch.no_grad():
            return (sqrt_c * t.norm(dim=-1)).max().item()

    def _diag_layer_in(self, i, residual):
        self.diag_data['res_frac'][i] = self._diag_radius(residual)
        if residual.requires_grad:
            residual.register_hook(
                lambda g, i=i: self.diag_data['grad_in'].__setitem__(
                    i, g.detach().norm().item()))

    def _shape_tangent(self, x):
        """Shape the encoder tangent vector before exp_map0 (c>0 only)."""
        if self.encoder_shell > 0:
            sqrt_c = self.c ** 0.5
            target_norm = math.atanh(min(self.encoder_shell, 1 - 1e-6)) / sqrt_c
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8) * target_norm
        elif self.encoder_scale <= 0:
            # One-time first-batch auto-calibration of a single global tangent
            # scale so the median residual lands at radius self._enc_target. High-dim
            # encoder outputs have ||v|| ~ sqrt(d), so exp_map0 otherwise saturates on
            # the boundary where hyperbolic_distance_sq's atanh clamp zeroes the
            # gradient. The scalar is frozen after the first batch and restored from
            # checkpoints; preserves per-vector magnitude variation.
            if self.training and self._enc_calibrated.item() == 0:
                with torch.no_grad():
                    sqrt_c = self.c ** 0.5
                    med = x.norm(dim=-1).median().clamp_min(1e-8)
                    target_norm = math.atanh(min(self._enc_target, 1 - 1e-6)) / sqrt_c
                    self._enc_scale.fill_(float(target_norm / med))
                    self._enc_calibrated.fill_(1)
            x = x * self._enc_scale
        elif self.encoder_scale != 1.0:
            x = x * self.encoder_scale
        return x

    def encode(self, x, n_q: tp.Optional[int] = None, st: int = 0):
        """Encode input to discrete code indices.

        Args:
            x: (B, D, N) input tensor.
            n_q: number of quantizer layers to use (default: all).
            st: starting quantizer index.
        Returns:
            codes: (used_n_q, B, N) code indices.
        """
        x = rearrange(x, "b d n -> b n d")
        n_q = n_q or len(self.layers)

        if self.c > 0:
            if self.requires_projection and self.tangent_proj:
                x = self.project_in(x)  # Euclidean D->d in tangent space, BEFORE exp_map0
            x = self._shape_tangent(x)
            residual = project(exp_map0(x, self.c), self.c)
            if self.requires_projection and not self.tangent_proj:
                residual = self.project_in(ManifoldTensor(residual, manifold=self.project_in.manifold)).tensor
        else:
            residual = x
            residual = self.project_in(residual)

        all_indices = []

        if self.new_method and self.c > 0:
            for layer in self.layers[st:n_q]:
                indices = layer._codebook.encode(residual)
                quantized = layer._codebook.decode(indices)
                residual = project(mobius_add(-quantized, residual, self.c), self.c)
                all_indices.append(indices)

        else:  # Standard mode
            for layer in self.layers[st:n_q]:
                indices = layer._codebook.encode(residual)
                quantized = layer._codebook.decode(indices)
                if self.c > 0:
                    #residual = project(mobius_add(-quantized, residual, self.c), self.c)
                    residual = project(mobius_sub(residual, quantized, self.c), self.c)
                else:
                    residual = residual - quantized
                all_indices.append(indices)

        return torch.stack(all_indices)

    def decode(self, codes):
        """Decode code indices back to a quantized representation.

        Args:
            codes: (n_q, B, N) code indices.
        Returns:
            quantized_out: (B, D, N) quantized output (tangent-space for c > 0).
        """
        n_q = codes.shape[0]
        all_quantized = []
        for i in range(n_q):
            quantized = self.layers[i]._codebook.decode(codes[i])  # (B, N, D)
            all_quantized.append(quantized)

        if self.new_method and self.c > 0:
            quantized_out = all_quantized[-1]
            for q in reversed(all_quantized[:-1]):
                quantized_out = project(mobius_add(q, quantized_out, self.c), self.c)

        elif self.c > 0:  # Standard hyperbolic
            quantized_out = all_quantized[0]
            for q in all_quantized[1:]:
                quantized_out = project(mobius_add(quantized_out, q, self.c), self.c)

        else:  # Standard Euclidean
            quantized_out = sum(all_quantized)

        # Apply output projection and map back to tangent space for hyperbolic
        if self.c > 0:
            if self.requires_projection and not self.tangent_proj:
                quantized_out = self.project_out(ManifoldTensor(quantized_out, manifold=self.project_out.manifold)).tensor
            quantized_out = log_map0(quantized_out, self.c)
            if self.requires_projection and self.tangent_proj:
                quantized_out = self.project_out(quantized_out)  # Euclidean d->D in tangent space, AFTER log_map0
        else:
            quantized_out = self.project_out(quantized_out)

        return rearrange(quantized_out, "b n d -> b d n")

    def forward(self, x, n_q: tp.Optional[int]=None, approx: bool=False):
        x = rearrange(x, "b d n -> b n d")
        if self.c > 0:
            if self.requires_projection and self.tangent_proj:
                x = self.project_in(x)  # Euclidean D->d in tangent space, BEFORE exp_map0
            x = self._shape_tangent(x)
            residual = project(exp_map0(x, self.c), self.c)
            if self.requires_projection and not self.tangent_proj:
                residual = self.project_in(ManifoldTensor(residual, manifold=self.project_in.manifold)).tensor
        else:
            residual = x
            residual = self.project_in(residual)

        # initial residual r0 on the ball (post project_in), base point of the
        # block-level PT-STE hop
        r0 = residual

        quantized_out = torch.zeros_like(residual)
        all_losses = []
        all_indices = []
        all_quantized = []
        distance = torch.tensor(0.0, device=x.device)

        #n_q = len(self.layers)
        n_q = n_q - self.remove

        if self.new_method and self.c > 0:
            if self.diag:
                self._diag_reset(n_q)
            for i, layer in enumerate(self.layers[:n_q]):
                if self.diag:
                    self._diag_layer_in(i, residual)
                quantized, indices, loss = layer(residual)
                all_quantized.append(quantized)
                if self.diag:
                    self.diag_data['q_frac'][i] = self._diag_radius(quantized)

                if self.entailment_cone_weight > 0:
                    q_flat = rearrange(quantized.detach(), "b n d -> (b n) d")
                    r_flat = rearrange(residual, "b n d -> (b n) d")
                    cone_loss = self.entailment_cone_loss_fn(q_flat, r_flat)
                    loss = loss + self.entailment_cone_weight * cone_loss

                if self.gyration_weight > 0 and self.c > 0 and self.training:
                    q_flat = rearrange(quantized, "b n d -> (b n) d")
                    r_flat = rearrange(residual, "b n d -> (b n) d")
                    q_norm = q_flat.norm(dim=-1).clamp_min(1e-8)
                    r_norm = r_flat.norm(dim=-1).clamp_min(1e-8)
                    cos_alpha = (q_flat * r_flat).sum(dim=-1) / (q_norm * r_norm)
                    cos_alpha = cos_alpha.clamp(-1 + 1e-6, 1 - 1e-6)
                    sin_alpha = (1 - cos_alpha.pow(2)).clamp_min(0).sqrt()
                    sqrt_c = self.c ** 0.5
                    q_poincare_norm = (2.0 / sqrt_c) * torch.atanh((sqrt_c * q_norm).clamp_max(1 - 1e-5))
                    r_poincare_norm = (2.0 / sqrt_c) * torch.atanh((sqrt_c * r_norm).clamp_max(1 - 1e-5))
                    gyration_loss = (self.c * q_poincare_norm * r_poincare_norm * sin_alpha).mean()
                    loss = loss + self.gyration_weight * gyration_loss

                residual = project(mobius_add(-quantized, residual, self.c), self.c)
                if self.training and self.gradient_correction:
                    residual = residual.detach()

                if self.dot_product_weight > 0:
                    q_log = log_map0(quantized, self.c).detach()
                    r_log = log_map0(residual, self.c)
                    dot_p_vec = ((q_log * r_log).sum(dim=-1) / q_log.norm(dim=-1).clamp_min(1e-5))
                    loss = loss + self.dot_product_weight * F.relu(-dot_p_vec).mean()

                all_indices.append(indices)
                all_losses.append(loss)

            quantized_out = all_quantized[-1]
            for q in reversed(all_quantized[:-1]):
                quantized_out = project(mobius_add(q, quantized_out, self.c), self.c)

            if self.block_hste_pt and self.training:
                # single block-level HSTE hop Q -> r0 on the ball (per-layer codes
                # are detached, so this is the only recon path to the encoder)
                quantized_out = HyperbolicSTE.apply(
                    r0, quantized_out, self.c, self.hste_riemannian,
                    self.gyration_only)

            if self.requires_projection and not self.tangent_proj:
                quantized_out = self.project_out(ManifoldTensor(quantized_out, manifold=self.project_out.manifold)).tensor

            if approx:
                diff = mobius_sub(mobius_add(quantized_out, residual, self.c), project(exp_map0(x, self.c), self.c), self.c)
                distance = hyperbolic_distance_sq(diff, torch.zeros_like(diff), self.c).mean()

            quantized_out = log_map0(quantized_out, self.c)
            if self.block_hste and self.training:
                # block-level identity STE in tangent coordinates (x = shaped tangent)
                quantized_out = x + (quantized_out - x).detach()
            if self.requires_projection and self.tangent_proj:
                quantized_out = self.project_out(quantized_out)  # Euclidean d->D in tangent space, AFTER log_map0

        else:
            for layer in self.layers[:n_q]:
                quantized, indices, loss = layer(residual)

                # Entailment Cone Loss: push residual into the cone of quantized
                if self.entailment_cone_weight > 0 and self.c > 0:
                    q_flat = rearrange(quantized.detach(), "b n d -> (b n) d")
                    r_flat = rearrange(residual, "b n d -> (b n) d")
                    cone_loss = self.entailment_cone_loss_fn(q_flat, r_flat)
                    loss = loss + self.entailment_cone_weight * cone_loss

                if self.gyration_weight > 0 and self.c > 0 and self.training:
                    q_flat = rearrange(quantized, "b n d -> (b n) d")
                    r_flat = rearrange(residual, "b n d -> (b n) d")
                    q_norm = q_flat.norm(dim=-1).clamp_min(1e-8)
                    r_norm = r_flat.norm(dim=-1).clamp_min(1e-8)
                    cos_alpha = (q_flat * r_flat).sum(dim=-1) / (q_norm * r_norm)
                    cos_alpha = cos_alpha.clamp(-1 + 1e-6, 1 - 1e-6)
                    sin_alpha = (1 - cos_alpha.pow(2)).clamp_min(0).sqrt()
                    sqrt_c = self.c ** 0.5
                    q_poincare_norm = (2.0 / sqrt_c) * torch.atanh((sqrt_c * q_norm).clamp_max(1 - 1e-5))
                    r_poincare_norm = (2.0 / sqrt_c) * torch.atanh((sqrt_c * r_norm).clamp_max(1 - 1e-5))
                    gyration_loss = (self.c * q_poincare_norm * r_poincare_norm * sin_alpha).mean()
                    loss = loss + self.gyration_weight * gyration_loss

                if self.c > 0:
                    residual = project(mobius_sub(residual, quantized, self.c), self.c)
                    if self.training and self.gradient_correction:
                        residual = residual.detach()
                    all_quantized.append(quantized)
                else:
                    residual = residual - quantized
                    quantized_out = quantized_out + quantized

                if self.dot_product_weight > 0:
                    if self.c > 0:
                        q_log = log_map0(quantized, self.c).detach()
                        r_log = log_map0(residual, self.c)
                        dot_p_vec = ((q_log * r_log).sum(dim=-1) / q_log.norm(dim=-1).clamp_min(1e-5))
                    else:
                        dot_p_vec = (quantized.detach() * residual).sum(dim=-1) / quantized.norm(dim=-1).clamp_min(1e-5).detach()
                    loss = loss + self.dot_product_weight * F.relu(-dot_p_vec).mean()

                all_indices.append(indices)
                all_losses.append(loss)
    
            if self.c > 0:
                # for q in reversed(all_quantized):
                #     quantized_out = project(mobius_add(q, quantized_out, self.c), self.c)
                quantized_out = all_quantized[0]
                for q in all_quantized[1:]:
                    quantized_out = project(mobius_add(quantized_out, q, self.c), self.c)

                if self.block_hste_pt and self.training:
                    # single block-level HSTE hop Q -> r0 on the ball (per-layer
                    # codes are detached, so this is the only recon path to the
                    # encoder)
                    quantized_out = HyperbolicSTE.apply(
                        r0, quantized_out, self.c, self.hste_riemannian,
                        self.gyration_only)

                if self.requires_projection and not self.tangent_proj:
                    quantized_out = self.project_out(ManifoldTensor(quantized_out, manifold=self.project_out.manifold)).tensor

                if approx:
                    diff = mobius_sub(mobius_add(quantized_out, residual, self.c), project(exp_map0(x, self.c), self.c), self.c)
                    distance = hyperbolic_distance_sq(diff, torch.zeros_like(diff), self.c).mean()

                quantized_out = log_map0(quantized_out, self.c)
                if self.block_hste and self.training:
                    # block-level identity STE in tangent coordinates (x = shaped tangent)
                    quantized_out = x + (quantized_out - x).detach()
                if self.requires_projection and self.tangent_proj:
                    quantized_out = self.project_out(quantized_out)  # Euclidean d->D in tangent space, AFTER log_map0
            else:
                quantized_out = self.project_out(quantized_out)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))

        
        quantized_out = rearrange(quantized_out, "b n d -> b d n")
        return quantized_out, out_indices, out_losses, distance
