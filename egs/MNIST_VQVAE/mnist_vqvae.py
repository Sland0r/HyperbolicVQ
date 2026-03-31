"""VQ-VAE model: 2D convolutional encoder/decoder with residual vector quantization.

Supports MNIST (1×28×28) and CIFAR-100 (3×32×32).
"""
import math
import torch
import torch.nn as nn
from academicodec.quantization import ResidualVectorQuantizer


class Encoder(nn.Module):
    """2D convolutional encoder.

    Architecture:  in_ch→32→64→128 channels via strided convolutions,
    then a 1×1 conv to project to the codebook dimension D.
    Output spatial resolution is always 4×4 for both 28×28 and 32×32 inputs.
    Final output is reshaped to (B, D, N) where N = 4×4 = 16.
    """

    def __init__(self, D: int = 128, in_channels: int = 1, img_size: int = 28):
        super().__init__()
        # The third conv uses k=3 for 28×28 (7→4) and k=4 for 32×32 (8→4)
        k3 = 3 if img_size == 28 else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=k3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, D, kernel_size=1),
        )
        self.spatial_h = 4
        self.spatial_w = 4

    def forward(self, x):
        z = self.net(x)  # (B, D, 4, 4)
        B, D, H, W = z.shape
        z = z.view(B, D, H * W)  # (B, D, N) with N=16
        return z


class Decoder(nn.Module):
    """2D convolutional decoder: mirrors the encoder.

    Takes (B, D, N), reshapes to (B, D, 4, 4), upsamples back to original size.
    """

    def __init__(self, D: int = 128, out_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.spatial_h = 4
        self.spatial_w = 4
        # First transposed conv uses k=3 for 28×28 (4→7) and k=4 for 32×32 (4→8)
        k3 = 3 if img_size == 28 else 4
        self.net = nn.Sequential(
            nn.Conv2d(D, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=k3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        B, D, N = z.shape
        z = z.view(B, D, self.spatial_h, self.spatial_w)
        x_hat = self.net(z)
        return x_hat


class VQVAE2D(nn.Module):
    """Full VQ-VAE for 2D images: Encoder → ResidualVectorQuantizer → Decoder.

    Args:
        D: latent/codebook dimension.
        n_q: number of residual quantization layers.
        bins: codebook size per layer.
        c: curvature for hyperbolic quantization (0 = Euclidean).
        ema: use EMA codebook updates.
        kmeans_init: initialise codebooks with k-means.
        in_channels: number of input image channels (1 for MNIST, 3 for CIFAR).
        img_size: spatial size of input images (28 or 32).
    """

    def __init__(
        self,
        D: int = 128,
        n_q: int = 4,
        bins: int = 256,
        c: float = 0.0,
        ema: bool = True,
        kmeans_init: bool = False,
        in_channels: int = 1,
        img_size: int = 28,
    ):
        super().__init__()
        self.encoder = Encoder(D=D, in_channels=in_channels, img_size=img_size)
        self.quantizer = ResidualVectorQuantizer(
            dimension=D,
            n_q=n_q,
            bins=bins,
            c=c,
            ema=ema,
            kmeans_init=kmeans_init,
        )
        self.decoder = Decoder(D=D, out_channels=in_channels, img_size=img_size)

        self.frame_rate = self.encoder.spatial_h * self.encoder.spatial_w  # 16
        self.target_bandwidths = [
            n_q * math.log2(bins) * self.frame_rate / 1000
        ]

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) images in [0, 1].
        Returns:
            x_hat: reconstructed images.
            commit_loss: scalar commitment loss.
            codes: quantizer indices (n_q, B, N).
        """
        z = self.encoder(x)
        bw = self.target_bandwidths[-1]
        quantized, codes, bandwidth, commit_loss = self.quantizer(
            z, self.frame_rate, bw
        )
        x_hat = self.decoder(quantized)
        return x_hat, commit_loss, codes

    def encode(self, x):
        z = self.encoder(x)
        bw = self.target_bandwidths[-1]
        codes = self.quantizer.encode(z, self.frame_rate, bw)
        return codes

    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        x_hat = self.decoder(quantized)
        return x_hat


# Keep old name as alias for backward compatibility
MNISTVQVAE = VQVAE2D


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for name, ch, sz in [("MNIST", 1, 28), ("CIFAR-100", 3, 32)]:
        model = VQVAE2D(in_channels=ch, img_size=sz)
        total, trainable = _count_params(model)
        print(f"{name}: Total params: {total:,}  Trainable: {trainable:,}")
        x = torch.randn(2, ch, sz, sz).clamp(0, 1)
        x_hat, commit_loss, codes = model(x)
        print(f"  Input: {x.shape}  Output: {x_hat.shape}  Codes: {codes.shape}  Commit: {commit_loss.item():.4f}")
