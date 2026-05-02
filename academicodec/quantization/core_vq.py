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

def hyperbolic_distance_sq(x, y, c, max_dist=10.0):
    m_add = mobius_sub(x, y, c) # "hyperbolic_distance_sq m_add"
    norm = m_add.norm(dim=-1, keepdim=True).clamp_min(1e-5) # "hyperbolic_distance_sq norm"
    sqrt_c = c ** 0.5
    arg = (sqrt_c * norm).clamp(min=0.0, max=1 - 1e-3) # "hyperbolic_distance_sq arg"
    dist = (2 / sqrt_c) * torch.atanh(arg) # "hyperbolic_distance_sq dist"
    return dist.pow(2) # "hyperbolic_distance_sq result"

def pairwise_hyperbolic_distance_sq(x, y, c, max_dist=10.0):
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
    x2 = x.pow(2).sum(dim=-1, keepdim=True).clamp_max(1 - 1e-5) # "exp_map x2"
    lambda_x = 2 / (1 - c * x2) # "exp_map lambda_x"
    return project(mobius_add(x, exp_map0(lambda_x * v / 2, c), c), c) # "exp_map result"

def log_map(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True).clamp_max(1 - 1e-5) # "log_map x2"
    lambda_x = 2 / (1 - c * x2) # "log_map lambda_x"
    return log_map0(mobius_add(-x, y, c), c) * 2 / lambda_x # "log_map result"


def conformal_factor(x, c):
    """Conformal factor λ_c^x = 2 / (1 - c ||x||^2)."""
    x2 = x.pow(2).sum(dim=-1, keepdim=True).clamp_max(1 - 1e-5)
    return 2.0 / (1.0 - c * x2)

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
            ema: bool=True, ):
        super().__init__()
        self.c = c
        self.decay = decay
        self.ema = ema
        init_fn: tp.Union[
            tp.Callable[..., torch.Tensor],
            tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        # if not kmeans_init:
        #     # Normalize random init to zero-mean, unit-variance
        #     embed = (embed - embed.mean()) / embed.std().clamp_min(1e-5)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        if self.c > 0:
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
            self.embed.data.copy_(project(self.embed.data, self.c))

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
            ema: bool=True, ):
        super().__init__()
        self.c = c
        self.ema = ema
        
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
            ema=ema)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        # if hasattr(self.project_in, 'manifold'):
        #     x = self.project_in(ManifoldTensor(x, manifold=self.project_in.manifold)).tensor
        # else:
        #     x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        # if hasattr(self.project_out, 'manifold'):
        #     quantize = self.project_out(ManifoldTensor(quantize, manifold=self.project_out.manifold)).tensor
        # else:
        #     quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

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
            if self.c > 0:
                # Hyperbolic straight-through estimator
                quantize = exp_map(x, log_map(x, quantize, self.c).detach(), self.c)
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

        # if hasattr(self.project_out, 'manifold'):
        #     quantize = self.project_out(ManifoldTensor(quantize, manifold=self.project_out.manifold)).tensor
        # else:
        #     quantize = self.project_out(quantize)
        # quantize = rearrange(quantize, "b n d -> b d n")
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
        if self.entailment_cone_weight > 0 and self.c > 0:
            self.entailment_cone_loss_fn = HyperbolicEntailmentConeLoss(K=0.1, c=self.c)
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)])
        self.remove = kwargs.get("remove", 0)
        dim = kwargs.get("dim", 256)
        codebook_dim = kwargs.get("codebook_dim", dim)
        _codebook_dim: int = default(codebook_dim, dim)

        self.requires_projection = _codebook_dim != dim
        if self.requires_projection and self.c > 0:
            hyp_manifold = HypllPoincareBall(c=Curvature(self.c))
            self.project_in = HLinear(dim, _codebook_dim, manifold=hyp_manifold, bias=True)
            self.project_out = HLinear(_codebook_dim, dim, manifold=hyp_manifold, bias=True)
        elif self.requires_projection:
            self.project_in = nn.Linear(dim, _codebook_dim)
            self.project_out = nn.Linear(_codebook_dim, dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

    def forward(self, x, n_q: tp.Optional[int]=None, validation=False):
        x = rearrange(x, "b d n -> b n d")
        if self.c > 0:
            residual = project(exp_map0(x, self.c), self.c)
            if self.requires_projection:
                residual = self.project_in(ManifoldTensor(residual, manifold=self.project_in.manifold)).tensor
        else:
            residual = x
            residual = self.project_in(residual)

        quantized_out = torch.zeros_like(residual)
        all_losses = []
        all_indices = []
        all_dots = []
        all_quantized = []

        #n_q = len(self.layers)
        n_q = n_q - self.remove

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
                
            if self.c > 0:
                residual = project(mobius_add(-quantized, residual, self.c), self.c)
                #residual = project(mobius_sub(residual, quantized, self.c), self.c)
                q_log = log_map0(quantized, self.c).detach()
                r_log = log_map0(residual, self.c)
                dot_p_vec = ((q_log * r_log).sum(dim=-1) / q_log.norm(dim=-1).clamp_min(1e-5))
                all_quantized.append(quantized)
                #quantized_out = project(mobius_add(quantized_out, quantized, self.c), self.c)
                
            else:
                residual = residual - quantized
                dot_p_vec = (quantized.detach() * residual).sum(dim=-1) / quantized.norm(dim=-1).clamp_min(1e-5).detach()
                quantized_out = quantized_out + quantized
                
            # Used for logging
            dot_p_scalar = dot_p_vec.mean()
            
            if validation:
                all_dots.append(dot_p_vec.flatten())

            # Negative Dot Product -> Expansion (penalize negative dot products per-element)
            if self.dot_product_weight > 0:
                # ReLU(-x) is positive when x is negative, so it penalizes negative dot products
                loss = loss + self.dot_product_weight * F.relu(-dot_p_vec).mean()

            # Entailment Cone Loss: push residual into the cone of quantized
            if self.entailment_cone_weight > 0 and self.c > 0:
                # quantized.detach() is parent (u), residual is child (v)
                # Gradients only flow through the residual
                # Reshape from (B, N, D) to (B*N, D) for the cone loss
                q_flat = rearrange(quantized.detach(), "b n d -> (b n) d")
                r_flat = rearrange(residual, "b n d -> (b n) d")
                cone_loss = self.entailment_cone_loss_fn(q_flat, r_flat)
                loss = loss + self.entailment_cone_weight * cone_loss

            all_indices.append(indices)
            all_losses.append(loss)

        if self.c > 0:
            for q in reversed(all_quantized):
                quantized_out = project(mobius_add(q, quantized_out, self.c), self.c)
            if self.requires_projection:
                quantized_out = self.project_out(ManifoldTensor(quantized_out, manifold=self.project_out.manifold)).tensor
            else:
                quantized_out = self.project_out(quantized_out)
            quantized_out = log_map0(quantized_out, self.c)
        else:
            quantized_out = self.project_out(quantized_out)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        
        quantized_out = rearrange(quantized_out, "b n d -> b d n")
        if validation:
            return quantized_out, out_indices, out_losses, torch.stack(all_dots)
        return quantized_out, out_indices, out_losses

    def encode(self,
               x: torch.Tensor,
               n_q: tp.Optional[int]=None,
               st: tp.Optional[int]=None) -> torch.Tensor:
        if self.c > 0:
            residual = project(exp_map0(x, self.c), self.c)
        else:
            residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:  # 设置解码的起止layer
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            if self.c > 0:
                residual = project(mobius_add(-quantized, residual, self.c), self.c)
            else:
                residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        if self.c > 0:
            quantized_out = torch.zeros((q_indices.shape[1], q_indices.shape[2], self.layers[0].project_out.out_features), device=q_indices.device) # placeholder layout, may need permuting based on real dims or initialized to actual 0 tensor of correct shape. `quantized = layer.decode(indices)` returns the output shape. Let's initialize `quantized_out` to scalar 0.0 or a typed 0 tensor later if c > 0 depending on the first iteration.

        first = True
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            if first and self.c > 0:
                quantized_out = torch.zeros_like(quantized)
            first = False

            if self.c > 0:
                quantized_out = project(mobius_add(quantized_out, quantized, self.c), self.c)
            else:
                quantized_out = quantized_out + quantized
        
        if self.c > 0:
            quantized_out = log_map0(quantized_out, self.c)
            
        return quantized_out
