# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Residual vector quantizer implementation."""
import math
import typing as tp
from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn

from academicodec.quantization.core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
            self,
            dimension: int=256,
            codebook_dim: tp.Optional[int]=None,
            n_q: int=8,
            bins: int=1024,
            decay: float=0.99,
            kmeans_init: bool=False,
            kmeans_iters: int=50,
            threshold_ema_dead_code: int=2,
            codebook_weight: float=1.0,
            commitment_weight: float=0.25,
            dot_product_weight: float=0.0,
            entailment_cone_weight: float=0.0,
            gyration_weight: float=0.0,
            c: float=0.0,
            remove: int=0,
            ema: bool=False,
            new_method: bool=True,
            approx: bool=False,
            hste: bool=False,
            hste_riemannian: bool=False,
            hste_clip: bool=False,
            gyration_only: bool=False,
            block_hste: bool=False,
            block_hste_pt: bool=False,
            gradient_correction: bool=False,
            A5: bool=False,
            A4_v2: bool=False,
            block_recon: bool=False,
            a6: bool=False,
            a7: bool=False,
            a6_1: bool=False,
            a7_1: bool=False,
            a8: bool=False,
            encoder_scale: float=1.0,
            encoder_scale_ema: float=0.0,
            encoder_shell: float=0.0,
            code_max_radius: float=0.0,
            embed_init_scale: float=1.0,
            tangent_proj: bool=False,
            ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.c = c
        self.remove = remove
        self.ema = ema
        self.new_method = new_method
        self.approx = approx
        self.hste = hste
        self.hste_riemannian = hste_riemannian
        self.hste_clip = hste_clip
        self.gyration_only = gyration_only
        self.block_hste = block_hste
        self.block_hste_pt = block_hste_pt
        self.gradient_correction = gradient_correction
        self.A5 = A5
        self.A4_v2 = A4_v2
        self.block_recon = block_recon
        self.a6 = a6
        self.a7 = a7
        self.a6_1 = a6_1
        self.a7_1 = a7_1
        self.a8 = a8
        self.encoder_scale = encoder_scale
        self.encoder_scale_ema = encoder_scale_ema
        self.encoder_shell = encoder_shell
        self.code_max_radius = code_max_radius
        self.embed_init_scale = embed_init_scale
        self.tangent_proj = tangent_proj
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            codebook_weight=codebook_weight,
            commitment_weight=commitment_weight,
            dot_product_weight=dot_product_weight,
            entailment_cone_weight=entailment_cone_weight,
            gyration_weight=gyration_weight,
            c=self.c,
            remove=self.remove,
            ema=self.ema,
            new_method=self.new_method,
            hste=self.hste,
            hste_riemannian=self.hste_riemannian,
            hste_clip=self.hste_clip,
            gyration_only=self.gyration_only,
            block_hste=self.block_hste,
            block_hste_pt=self.block_hste_pt,
            gradient_correction=self.gradient_correction,
            A5=self.A5,
            A4_v2=self.A4_v2,
            block_recon=self.block_recon,
            a6=self.a6,
            a7=self.a7,
            a6_1=self.a6_1,
            a7_1=self.a7_1,
            a8=self.a8,
            encoder_scale=self.encoder_scale,
            encoder_scale_ema=self.encoder_scale_ema,
            encoder_shell=self.encoder_shell,
            code_max_radius=self.code_max_radius,
            embed_init_scale=self.embed_init_scale,
            tangent_proj=self.tangent_proj)

    def forward(self,
                x: torch.Tensor,
                sample_rate: int,
                bandwidth: tp.Optional[float]=None,
                nq = 0, validation=False) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        if nq == 0:
            n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        else:
            n_q = nq
            
        quantized, codes, commit_loss, distance = self.vq(x, n_q=n_q, approx=self.approx)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return quantized, codes, bw, torch.mean(commit_loss), distance

    def get_num_quantizers_for_bandwidth(
            self, sample_rate: int, bandwidth: tp.Optional[float]=None) -> int:
        """Return n_q based on specified target bandwidth.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.:
            n_q = int(max(1, math.floor(bandwidth / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return bandwidth per quantizer for a given input sample rate.
        """
        return math.log2(self.bins) * sample_rate / 1000

    def encode(self,
               x: torch.Tensor,
               sample_rate: int,
               bandwidth: tp.Optional[float]=None,
               st: tp.Optional[int]=None) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        st = st or 0
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes)
        return quantized
