import math
import random

import numpy as np
import torch.nn as nn
from academicodec.modules.seanet import SEANetDecoder
from academicodec.modules.seanet import SEANetEncoder
from academicodec.quantization import ResidualVectorQuantizer


# Generator
class SoundStream(nn.Module):
    def __init__(self,
                 n_filters,
                 D,
                 target_bandwidths=[7.5, 15],
                 exponential_lambda=0.4,
                 uniform: bool=False,
                 ratios=[8, 5, 4, 2],
                 sample_rate=24000,
                 bins=1024,
                 normalize=False,
                 threshold_ema_dead_code=2,
                 codebook_weight: float=1.0,
                 commitment_weight: float=0.25,
                 dot_product_weight: float=0.0,
                 entailment_cone_weight: float=0.0,
                 gyration_weight: float=0.0,
                 c: float=0.0,
                 ema: bool=True,
                 decay: float=0.99,
                 kmeans_init: bool=True,
                 pre_quant_batchnorm: bool=False,
                 remove: int=0,
                 codebook_dim: int=None,
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
                 a6: bool=False,
                 a7: bool=False,
                 a6_1: bool=False,
                 a7_1: bool=False,
                 a8: bool=False,
                 encoder_scale: float=1.0,
                 encoder_scale_ema: float=0.0,
                 encoder_shell: float=0.0,
                 code_max_radius: float=0.0,
                 target_max_recon: float=0.0,
                 embed_init_scale: float=1.0,
                 tangent_proj: bool=False):
        super().__init__()
        self.hop_length = np.prod(ratios)  # 计算乘积
        self.encoder = SEANetEncoder(
            n_filters=n_filters, dimension=D, ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] //
                  (math.ceil(sample_rate / self.hop_length) * 10))
        # --target_max_recon T (c>0 only): derive the per-code radius cap so the
        # worst-case (all codes collinear) Mobius-sum reconstruction lands at
        # fraction T of the ball radius. Rapidities add linearly along a ray, so
        # tanh(n_q * atanh(code_max_radius)) = T  =>  cap = tanh(atanh(T)/n_q).
        # Curvature-independent because code_max_radius is itself a ball fraction.
        # Overrides any explicit --code_max_radius.
        if target_max_recon > 0:
            code_max_radius = math.tanh(math.atanh(target_max_recon) / n_q)
            print(f'[target_max_recon] T={target_max_recon} n_q={n_q} '
                  f'-> code_max_radius={code_max_radius:.6f}')
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 75
        self.bits_per_codebook = int(math.log2(bins))
        self.target_bandwidths = target_bandwidths
        self.exponential_lambda = exponential_lambda
        self.uniform = uniform
        self.n_q = n_q
        self.bins = bins
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.quantizer = ResidualVectorQuantizer(
            dimension=D, codebook_dim=codebook_dim, n_q=n_q, bins=bins, threshold_ema_dead_code=self.threshold_ema_dead_code,
            codebook_weight=codebook_weight, commitment_weight=commitment_weight,
            dot_product_weight=dot_product_weight, entailment_cone_weight=entailment_cone_weight,
            gyration_weight=gyration_weight,
            c=c, ema=ema, decay=decay, kmeans_init=kmeans_init, remove=remove,
            new_method=new_method, approx=approx, hste=hste, hste_riemannian=hste_riemannian,
            hste_clip=hste_clip,
            gyration_only=gyration_only, block_hste=block_hste,
            block_hste_pt=block_hste_pt,
            gradient_correction=gradient_correction, A5=A5, A4_v2=A4_v2,
            a6=a6, a7=a7, a6_1=a6_1, a7_1=a7_1, a8=a8,
            encoder_scale=encoder_scale, encoder_scale_ema=encoder_scale_ema,
            encoder_shell=encoder_shell, code_max_radius=code_max_radius,
            embed_init_scale=embed_init_scale, tangent_proj=tangent_proj)
        self.pre_quant_batchnorm = pre_quant_batchnorm
        self.pre_quant_bn = nn.BatchNorm1d(D) if pre_quant_batchnorm else nn.Identity()
        self.decoder = SEANetDecoder(
            n_filters=n_filters, dimension=D, ratios=ratios)

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def forward(self, x, validation=False):
        e = self.encoder(x)
        e = self.pre_quant_bn(e)
        max_idx = len(self.target_bandwidths) - 1
        if self.exponential_lambda > 0.0:
            idx = min(max_idx, int(random.expovariate(self.exponential_lambda)))
            bw = self.target_bandwidths[idx]
        elif self.uniform:
            bw = self.target_bandwidths[random.randint(0, max_idx)]
        else:
            bw = self.target_bandwidths[-1]
            
        if validation:
            quantized, codes, bandwidth, commit_loss, distance = self.quantizer(
                e, self.frame_rate, bw, validation=validation)
            o = self.decoder(quantized)
            return o, commit_loss, None, codes, distance
        else:
            quantized, codes, bandwidth, commit_loss, distance = self.quantizer(
                e, self.frame_rate, bw, validation=validation)
            o = self.decoder(quantized)
            return o, commit_loss, None, codes, distance

    def encode(self, x, target_bw=None, st=None):
        e = self.encoder(x)
        e = self.pre_quant_bn(e)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        if st is None:
            st = 0
        codes = self.quantizer.encode(e, self.frame_rate, bw, st)
        return codes

    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        o = self.decoder(quantized)
        return o
