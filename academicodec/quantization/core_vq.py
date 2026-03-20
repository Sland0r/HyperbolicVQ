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

from academicodec.quantization.distrib import broadcast_tensors


def mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def hyperbolic_distance_sq(x, y, c, max_dist=10.0):
    m_add = mobius_add(-x, y, c)
    norm = m_add.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    sqrt_c = c ** 0.5
    arg = (sqrt_c * norm).clamp(min=0.0, max=1 - 1e-3)
    dist = (2 / sqrt_c) * torch.atanh(arg)
    return dist.clamp_max(max_dist).pow(2)

def pairwise_hyperbolic_distance_sq(x, y, c, max_dist=10.0):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = x @ y.t()
    sq_dist = (x2 + y2.t() - 2 * xy).clamp_min(0.0)
    denom = ((1 - c * x2) @ (1 - c * y2).t()).clamp_min(1e-6)
    arg = 1 + 2 * c * sq_dist / denom
    dist = (1 / (c ** 0.5)) * torch.acosh(arg.clamp_min(1.0 + 1e-5))
    return dist.clamp_max(max_dist).pow(2)

def exp_map0(v, c):
    norm = v.norm(dim=-1, keepdim=True)
    sqrt_c = c ** 0.5
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm.clamp_min(1e-15))
    return v * scale

def log_map0(y, c):
    norm = y.norm(dim=-1, keepdim=True)
    sqrt_c = c ** 0.5
    scale = torch.atanh((sqrt_c * norm).clamp_max(1 - 1e-5)) / (sqrt_c * norm.clamp_min(1e-15))
    return y * scale

def project(x, c, eps=1e-5):
    """Project x onto the open Poincaré ball of radius 1/sqrt(c)."""
    max_norm = 1.0 / (c ** 0.5) - eps
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.where(norm > max_norm, x * (max_norm / norm), x)

def exp_map(x, v, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True).clamp_max(1 - 1e-5)
    lambda_x = 2 / (1 - c * x2)
    return project(mobius_add(x, exp_map0(lambda_x * v / 2, c), c), c)

def log_map(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True).clamp_max(1 - 1e-5)
    lambda_x = 2 / (1 - c * x2)
    return log_map0(mobius_add(-x, y, c), c) * 2 / lambda_x


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


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

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

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

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], # true when codebook is dead
            sample_vectors(samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code # number of clusters = codebook size
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d") # likely not necessary, already in that form
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

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
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # (everything, dim)

        self.init_embed_(x)

        embed_ind = self.quantize(x) # indices of the closest centroid
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape) # back to normal shape
        quantize = self.dequantize(embed_ind) # quantized x

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)

            if not self.ema:
                # Skip EMA: codebook is nn.Parameter, updated via optimizer
                # TODO: might add reset for dead codes here
                pass
            elif self.c > 0:
                ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
                # Compute smoothed denominator for EMA
                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size,
                                      self.epsilon) * self.cluster_size.sum())

                # Map x to the tangent space of their assigned centroids
                embed_ind_flat = embed_ind.view(-1)
                embedded = self.embed[embed_ind_flat] # Contextual centroids per batch item
                tangent_x = log_map(embedded, x, self.c)
                
                # Sum the tangent vectors per centroid
                tangent_sum = torch.zeros_like(self.embed)
                tangent_sum.scatter_add_(0, repeat(embed_ind_flat, "n -> n d", d=self.embed.size(-1)), tangent_x)
                
                # The effective step size is scaled by (1 - decay) and averaged by the smoothed cluster size
                step = (1 - self.decay) * (tangent_sum / cluster_size.unsqueeze(1))
                
                # Move the active centroids along the geodesic by the step amount
                embed_normalized = exp_map(self.embed, step, self.c)
                embed_normalized = project(embed_normalized, self.c)
                self.embed.data.copy_(embed_normalized)

            else:
                ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
                embed_sum = x.t() @ embed_onehot
                ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size,
                                      self.epsilon) * self.cluster_size.sum())
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embed.data.copy_(embed_normalized)

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
            commitment_weight: float=1.,
            c: float=0.,
            ema: bool=True, ):
        super().__init__()
        self.c = c
        self.ema = ema
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)
                           if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)
                            if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

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
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x): # quantizes x, computes loss depending on distance to codes, properly propagates gradients
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        # Save pre-STE quantize (has gradient path to embed) for codebook loss
        quantize_raw = quantize

        if self.training:
            if self.c > 0:
                diff = mobius_add(-x, quantize, self.c)
                quantize = project(mobius_add(x, diff.detach(), self.c), self.c)
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
                #TODO: make the gradient riemannian
                if self.c > 0:
                    codebook_loss = hyperbolic_distance_sq(x.detach(), quantize_raw, self.c).mean()
                else:
                    codebook_loss = F.mse_loss(x.detach(), quantize_raw)
                loss = loss + codebook_loss

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.c = kwargs.get("c", 0.0)
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)])
        print("EMA", kwargs.get("ema", False))

    def forward(self, x, n_q: tp.Optional[int]=None):
        if self.c > 0:
            residual = project(exp_map0(x, self.c), self.c)
            quantized_out = torch.zeros_like(residual)
        else:
            residual = x
            quantized_out = 0.0

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            if self.c > 0:
                residual = project(mobius_add(-quantized, residual, self.c), self.c)
                quantized_out = project(mobius_add(quantized_out, quantized, self.c), self.c)
            else:
                residual = residual - quantized
                quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        if self.c > 0:
            quantized_out = log_map0(quantized_out, self.c)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
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
