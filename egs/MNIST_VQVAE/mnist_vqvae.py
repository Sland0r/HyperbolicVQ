"""VQ-VAE model: 2D convolutional encoder/decoder with residual vector quantization.

Supports MNIST (1×28×28) and CIFAR-100 (3×32×32).
"""
import math
import random
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

    def __init__(self, D: int = 128, in_channels: int = 1, img_size: int = 28,
                 size: str = 'small', full_grid: bool = False):
        super().__init__()
        self.size = size
        self.full_grid = full_grid

        if full_grid:
            # Quantize the conv feature map directly at its true spatial
            # resolution (img_size // 4): 32 -> 8×8 = 64 tokens, 28 -> 7×7 = 49.
            # No bottleneck: each spatial position becomes one code token.
            self.spatial_h = img_size // 4
            self.spatial_w = img_size // 4
            self.num_tokens = self.spatial_h * self.spatial_w
            self.bottleneck = None
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # H/2
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # H/4
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),          # H/4
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, D, kernel_size=1),                               # (B, D, H/4, W/4)
            )
            return

        # ── Legacy architecture: bottleneck collapses to N=1/4/16 tokens ──
        # The third conv uses k=3 for 28×28 (7→4) and k=4 for 32×32 (8→4)
        k3 = 3 if img_size == 28 else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=k3, stride=2, padding=1), # (B, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, D, kernel_size=1), # (B, D, 4, 4)
        )

        self.spatial_h = 4
        self.spatial_w = 4
        if size == 'small':
            self.num_tokens = 1
            self.bottleneck = nn.Linear(D * self.spatial_h * self.spatial_w, D)
        elif size == 'medium':
            self.num_tokens = 4
            self.bottleneck = nn.Linear(D * self.spatial_h * self.spatial_w, D * 2 * 2)
        elif size == 'large':
            self.num_tokens = 16
            self.bottleneck = nn.Linear(D * self.spatial_h * self.spatial_w, D * 4 * 4)
        else:
            raise ValueError(f"Unknown size: {size}")

    def forward(self, x):
        if self.full_grid:
            z = self.net(x)                 # (B, D, H, W)
            B, D, H, W = z.shape
            return z.reshape(B, D, H * W)   # (B, D, N) with N = H*W

        z = self.net(x)  # (B, D, 4, 4)
        B, D, H, W = z.shape
        z = z.reshape(B, D * H * W)  # (B, D * N) with N=16
        z = self.bottleneck(z)
        if self.size == 'small':
            z = z.unsqueeze(-1) # (B, D, 1)
        elif self.size == 'medium':
            z = z.reshape(B, D, 4) # (B, D, 4)
        elif self.size == 'large':
            z = z.reshape(B, D, 16) # (B, D, 16)
        return z


class Decoder(nn.Module):
    """2D convolutional decoder: mirrors the encoder.

    Takes (B, D, N), reshapes to (B, D, 4, 4), upsamples back to original size.
    """

    def __init__(self, D: int = 128, out_channels: int = 1, img_size: int = 28,
                 size: str = 'small', full_grid: bool = False):
        super().__init__()
        self.full_grid = full_grid

        if full_grid:
            # Mirror the full-grid encoder: reshape (B, D, N) -> (B, D, H, W) at
            # H = W = img_size // 4, then two stride-2 upsamples back to img_size.
            self.spatial_h = img_size // 4
            self.spatial_w = img_size // 4
            self.bottleneck = None
            self.net = nn.Sequential(
                nn.Conv2d(D, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # H*2
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # H*4
                nn.Sigmoid(),
            )
            return

        # ── Legacy architecture ──
        self.spatial_h = 4
        self.spatial_w = 4
        # First transposed conv uses k=3 for 28×28 (4→7) and k=4 for 32×32 (4→8)
        if size == 'small':
            self.bottleneck = nn.Linear(D, D * self.spatial_h * self.spatial_w)
        elif size == 'medium':
            self.bottleneck = nn.Linear(D * 2 * 2, D * self.spatial_h * self.spatial_w)
        elif size == 'large':
            self.bottleneck = nn.Linear(D * 4 * 4, D * self.spatial_h * self.spatial_w)
        else:
            raise ValueError(f"Unknown size: {size}")
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
        if self.full_grid:
            B, D, N = z.shape
            z = z.reshape(B, D, self.spatial_h, self.spatial_w)
            return self.net(z)

        B, D, N = z.shape
        z = z.reshape(B, D * N)
        z = self.bottleneck(z).reshape(B, D, self.spatial_h, self.spatial_w) # (B, D, 4, 4)
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
        exponential_lambda: float = 0.0,
        uniform: bool = False,
        ema: bool = False,
        kmeans_init: bool = False,
        threshold_ema_dead_code: int=2, 
        codebook_weight: float=1.0,
        commitment_weight: float=0.25,
        dot_product_weight: float=0.0,
        entailment_cone_weight: float=0.0,
        gyration_weight: float=0.0,
        in_channels: int = 1,
        img_size: int = 28,
        size: str = 'small',
        solution: bool = False,
        gyration: bool = False,
        new_method: bool = True,
        approx: bool = False,
        hste: bool = False,
        hste_riemannian: bool = False,
        gradient_correction: bool = False,
        full_grid: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(D=D, in_channels=in_channels, img_size=img_size, size=size, full_grid=full_grid)
        self.quantizer = ResidualVectorQuantizer(
            dimension=D,
            n_q=n_q,
            bins=bins,
            kmeans_init=kmeans_init,
            threshold_ema_dead_code=threshold_ema_dead_code,
            codebook_weight=codebook_weight,
            commitment_weight=commitment_weight,
            dot_product_weight=dot_product_weight,
            entailment_cone_weight=entailment_cone_weight,
            gyration_weight=gyration_weight,
            c=c,
            ema=ema,
            new_method=new_method,
            approx=approx,
            hste=hste,
            hste_riemannian=hste_riemannian,
            gradient_correction=gradient_correction,
        )
        self.decoder = Decoder(D=D, out_channels=in_channels, img_size=img_size, size=size, full_grid=full_grid)
        self.exponential_lambda = exponential_lambda
        self.uniform = uniform

        # Number of code tokens per image (= spatial positions). Used as the
        # "frame_rate" bandwidth label by the quantizer; with full_grid this is
        # the true feature-map resolution (CIFAR 64, MNIST/EMNIST 49).
        self.frame_rate = self.encoder.num_tokens

        self.n_q = n_q
        self.target_bandwidths = [
            n_q * math.log2(bins) * self.frame_rate / 1000
        ]
        print('target_bandwidths', self.target_bandwidths)

    def forward(self, x, validation=False):
        """
        Args:
            x: (B, C, H, W) images in [0, 1].
        Returns:
            x_hat: reconstructed images.
            latent_loss: scalar commitment loss.
            codes: quantizer indices (n_q, B, N).
        """
        z = self.encoder(x) # (B, D, 4, 4)
        bw = self.target_bandwidths[-1]
        if self.exponential_lambda > 0.0:
            nq = min(self.n_q, int(random.expovariate(self.exponential_lambda)))
        elif self.uniform:
            nq = random.randint(1, self.n_q)
        else:
            nq = self.n_q
        
        if validation:
            quantized, codes, bandwidth, latent_loss, distance = self.quantizer(
                z, self.frame_rate, bw, nq = nq
            )
            x_hat = self.decoder(quantized)
            return x_hat, latent_loss, codes, distance
        else:
            quantized, codes, bandwidth, latent_loss, distance = self.quantizer(
                z, self.frame_rate, bw, nq = nq, validation=validation
            )
            x_hat = self.decoder(quantized)
            return x_hat, latent_loss, codes, distance

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


class RQTransformer(nn.Module):
    """RQ-Transformer for autoregressive generation of RQ-VAE codes.

    Follows the two-component architecture from Lee et al. (CVPR 2022):
    - Spatial Transformer: predicts the next spatial position given all previous
    - Depth Transformer: predicts D codes at each position autoregressively over depth

    Args:
        n_positions: T, spatial sequence length (e.g. 1 for 'small', 4 for 'medium', 16 for 'large')
        n_depths: D, number of RQ depths (= n_q)
        codebook_size: K, number of entries per codebook
        embed_dim: hidden dimension for transformer layers
        n_spatial_layers: number of spatial transformer blocks
        n_depth_layers: number of depth transformer blocks
        n_heads: number of attention heads
    """

    def __init__(
        self,
        n_positions: int,
        n_depths: int,
        codebook_size: int,
        embed_dim: int = 256,
        n_spatial_layers: int = 4,
        n_depth_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.n_depths = n_depths
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim

        self.code_embed = nn.Embedding(codebook_size, embed_dim)
        self.spatial_pos_embed = nn.Embedding(n_positions, embed_dim)
        self.depth_pos_embed = nn.Embedding(n_depths, embed_dim)
        self.sos_embed = nn.Parameter(torch.randn(embed_dim))

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        depth_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.depth_transformer = nn.TransformerEncoder(depth_layer, num_layers=n_depth_layers)

        self.head = nn.Linear(embed_dim, codebook_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _build_spatial_input(self, codes: torch.Tensor) -> torch.Tensor:
        """Build spatial transformer input sequence.

        Args:
            codes: (B, T, D) integer code indices.
        Returns:
            (B, T, embed_dim) input embeddings for spatial transformer.
        """
        B, T, D = codes.shape
        device = codes.device

        pos = self.spatial_pos_embed(torch.arange(T, device=device))  # (T, E)

        # u_1 = sos + pos[0]; u_t = pos[t] + sum of code embeds at position t-1
        code_embeds = self.code_embed(codes)  # (B, T, D, E)
        sum_embeds = code_embeds.sum(dim=2)  # (B, T, E)

        # Shift: position t gets sum from t-1
        shifted = torch.zeros(B, T, self.embed_dim, device=device)
        shifted[:, 0] = self.sos_embed
        shifted[:, 1:] = sum_embeds[:, :-1]

        return shifted + pos.unsqueeze(0)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Compute logits for all positions and depths.

        Args:
            codes: (B, T, D) integer code indices (ground truth for teacher forcing).
        Returns:
            logits: (B, T, D, K) predicted logits.
        """
        B, T, D = codes.shape
        device = codes.device

        # Spatial transformer
        spatial_input = self._build_spatial_input(codes)  # (B, T, E)
        spatial_mask = self._causal_mask(T, device)
        h = self.spatial_transformer(spatial_input, mask=spatial_mask)  # (B, T, E)

        # Depth transformer for every spatial position at once. The D depth inputs
        # at position t are independent given h_t, so fold T into the batch dim and
        # run the depth transformer a single time on (B*T, D, E) instead of looping
        # over t. Inputs:  v_{t,0} = depth_pos[0] + h_t
        #                  v_{t,d} = depth_pos[d] + sum of code embeds at depths < d
        depth_pos = self.depth_pos_embed(torch.arange(D, device=device))  # (D, E)
        code_embeds = self.code_embed(codes)  # (B, T, D, E)

        # prev_sum[..., d] = sum of code embeds over depths 0..d-1 (0 at d=0).
        prev_sum = torch.zeros_like(code_embeds)
        prev_sum[:, :, 1:] = code_embeds.cumsum(dim=2)[:, :, :-1]
        depth_input = prev_sum + depth_pos.view(1, 1, D, self.embed_dim)

        # Add the spatial context h_t only at depth 0 (not in-place on the graph).
        spatial_ctx = torch.zeros_like(depth_input)
        spatial_ctx[:, :, 0] = h
        depth_input = depth_input + spatial_ctx

        depth_input = depth_input.reshape(B * T, D, self.embed_dim)
        depth_mask = self._causal_mask(D, device)
        depth_out = self.depth_transformer(depth_input, mask=depth_mask)  # (B*T, D, E)
        logits = self.head(depth_out).view(B, T, D, self.codebook_size)
        return logits

    @torch.no_grad()
    def generate(self, n_samples: int, device: torch.device, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """Autoregressively sample codes.

        Args:
            n_samples: batch size to generate.
            device: torch device.
            temperature: sampling temperature.
            top_k: if > 0, only sample from top-k logits.
        Returns:
            codes: (B, T, D) sampled code indices.
        """
        B, T, D = n_samples, self.n_positions, self.n_depths
        codes = torch.zeros(B, T, D, dtype=torch.long, device=device)

        depth_pos = self.depth_pos_embed(torch.arange(D, device=device))  # (D, E)
        for t in range(T):
            # Rebuild spatial input up to position t. Position 0 is the SOS token;
            # position k (1..t) sums the code embeds of the already-sampled
            # position k-1, added to its spatial positional embedding.
            pos = self.spatial_pos_embed(torch.arange(t + 1, device=device))  # (t+1, E)
            spatial_input = torch.zeros(B, t + 1, self.embed_dim, device=device)
            spatial_input[:, 0] = self.sos_embed + pos[0]
            if t > 0:
                prev_embeds = self.code_embed(codes[:, :t]).sum(dim=2)  # (B, t, E)
                spatial_input[:, 1:] = prev_embeds + pos[1:].unsqueeze(0)

            spatial_mask = self._causal_mask(t + 1, device)
            h = self.spatial_transformer(spatial_input, mask=spatial_mask)
            context = h[:, t]  # (B, E)

            # Depth transformer: predict D codes one by one. At depth step d the
            # codes at depths 0..d-1 are known; v_{k} = depth_pos[k] + sum of code
            # embeds over depths < k, with the spatial context added at depth 0.
            for d in range(D):
                depth_input = depth_pos[:d + 1].unsqueeze(0).expand(B, d + 1, self.embed_dim).clone()
                depth_input[:, 0] = depth_pos[0] + context
                if d > 0:
                    known = self.code_embed(codes[:, t, :d])  # (B, d, E) depths 0..d-1
                    depth_input[:, 1:] = depth_input[:, 1:] + known.cumsum(dim=1)

                depth_mask = self._causal_mask(d + 1, device)
                depth_out = self.depth_transformer(depth_input, mask=depth_mask)
                logits = self.head(depth_out[:, d]) / temperature  # (B, K)

                if top_k > 0:
                    top_vals, _ = logits.topk(top_k, dim=-1)
                    logits[logits < top_vals[:, -1:]] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                codes[:, t, d] = torch.multinomial(probs, 1).squeeze(-1)

        return codes


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
        x_hat, latent_loss, codes = model(x)
        print(f"  Input: {x.shape}  Output: {x_hat.shape}  Codes: {codes.shape}  Latent: {latent_loss.item():.4f}")
