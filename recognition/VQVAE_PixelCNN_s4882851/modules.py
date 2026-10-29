"""Main solution: hierarchical VQ-VAE-2 with a causal Transformer prior.

This is the submitted model for COMP3710 Project 2.4 (synthetic prostate MRI).
It contains the two components that constitute the solution:

  * ``VQVAE2`` — a two-level (coarse 16x8 + fine 64x32) vector-quantised
    autoencoder (Razavi, van den Oord & Vinyals, "Generating Diverse
    High-Fidelity Images with VQ-VAE-2", NeurIPS 2019). The hierarchy lets the
    model represent global anatomy (coarse) and local texture (fine)
    separately, which the single-level VQ-VAE could not.
  * ``TransformerPrior`` — a causal self-attention prior over the fine code
    grid. It replaces the PixelCNN prior because the PixelCNN's limited
    receptive field could not place anatomy globally (see report Section 6.5);
    full causal attention gives every code position access to all preceding
    positions.

The remaining architectures used for comparison and reproducibility — the
single-level VQ-VAE, the masked/gated/conditioned PixelCNN priors and the
DCGAN baseline — live in the ``models`` package and are re-exported at the
bottom of this file, so ``from modules import VQVAE, PixelCNNPrior, ...``
continues to work exactly as before.

All model code is pure PyTorch (no NumPy), as required by the course spec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.primitives import ResidualBlock, VectorQuantizer


# --------------------------------------------------------------------------- #
# MAIN SOLUTION (1/2): VQ-VAE-2 hierarchical vector quantisation
# --------------------------------------------------------------------------- #
class VQVAE2(nn.Module):
    """Hierarchical VQ-VAE with two-level quantization for global + local structure.

    Architecture:
    - Encoder produces TWO latent grids:
      * Coarse (16x8): Captures global anatomy (pelvis position, overall structure)
      * Fine (64x32): Captures local texture and details
    - Decoder combines upsampled coarse with fine for reconstruction
    - Two separate priors:
      * Coarse prior: Learns global anatomical layout
      * Fine prior: Learns local texture conditioned on coarse codes

    This addresses the receptive-field problem: the coarse prior only needs to
    model 16x8=128 positions (easily covered by a PixelCNN), while the fine
    prior has coarse context to guide local detail generation.

    Based on "Generating Diverse High-Fidelity Images with VQ-VAE-2"
    (Razavi, van den Oord & Vinyals, NeurIPS 2019).
    """

    def __init__(self, in_channels=1, width=128, latent_dim=64,
                 num_codes=512, beta=0.25):
        """Build the two-level encoder, both quantisers, and the decoder.

        Args:
            in_channels: input image channels (1 for grayscale MRI).
            width: hidden feature width.
            latent_dim: codebook embedding dimension D for both levels.
            num_codes: codebook size K for both levels.
            beta: commitment-loss weight.
        """
        super().__init__()

        # Encoder path: 256x128 -> 128x64 -> 64x32 -> 32x16 -> 16x8
        # We extract features at two levels: 64x32 (fine) and 16x8 (coarse)

        # Shared encoder backbone to 64x32
        self.enc_conv1 = nn.Conv2d(in_channels, width, 4, stride=2, padding=1)  # 256x128 -> 128x64
        self.enc_conv2 = nn.Conv2d(width, width, 4, stride=2, padding=1)        # 128x64 -> 64x32
        self.enc_res1 = ResidualBlock(width)
        self.enc_res2 = ResidualBlock(width)

        # Fine level: quantize at 64x32
        self.enc_fine = nn.Conv2d(width, latent_dim, 1)
        self.quantizer_fine = VectorQuantizer(num_codes, latent_dim, beta)

        # Continue encoding to coarse level: 64x32 -> 32x16 -> 16x8
        self.enc_conv3 = nn.Conv2d(width, width, 4, stride=2, padding=1)        # 64x32 -> 32x16
        self.enc_conv4 = nn.Conv2d(width, width, 4, stride=2, padding=1)        # 32x16 -> 16x8
        self.enc_res3 = ResidualBlock(width)
        self.enc_res4 = ResidualBlock(width)

        # Coarse level: quantize at 16x8
        self.enc_coarse = nn.Conv2d(width, latent_dim, 1)
        self.quantizer_coarse = VectorQuantizer(num_codes, latent_dim, beta)

        # Decoder path: 16x8 -> 32x16 -> 64x32 -> 128x64 -> 256x128
        # Coarse codes are upsampled and combined with fine codes

        # Decode from coarse: 16x8 -> 32x16 -> 64x32
        self.dec_coarse_in = nn.Conv2d(latent_dim, width, 1)
        self.dec_res1 = ResidualBlock(width)
        self.dec_res2 = ResidualBlock(width)
        self.dec_upsample1 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)  # 16x8 -> 32x16
        self.dec_upsample2 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)  # 32x16 -> 64x32

        # Combine upsampled coarse with fine codes at 64x32
        self.dec_combine = nn.Conv2d(width + latent_dim, width, 1)
        self.dec_res3 = ResidualBlock(width)
        self.dec_res4 = ResidualBlock(width)

        # Final upsampling to image: 64x32 -> 128x64 -> 256x128
        self.dec_upsample3 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)  # 64x32 -> 128x64
        self.dec_upsample4 = nn.ConvTranspose2d(width, in_channels, 4, stride=2, padding=1)  # 128x64 -> 256x128
        self.dec_out = nn.Sigmoid()

    def encode(self, x):
        """Encode image to coarse and fine code indices.

        Args:
            x: image batch [B, C, 256, 128].

        Returns:
            codes_coarse: [B, 16, 8] integer codes.
            codes_fine: [B, 64, 32] integer codes.
        """
        # Shared encoding to 64x32
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = self.enc_res1(h)
        h = self.enc_res2(h)

        # Fine level quantization at 64x32
        z_e_fine = self.enc_fine(h)
        _, _, _, codes_fine = self.quantizer_fine(z_e_fine)

        # Continue to coarse level
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = self.enc_res3(h)
        h = self.enc_res4(h)

        # Coarse level quantization at 16x8
        z_e_coarse = self.enc_coarse(h)
        _, _, _, codes_coarse = self.quantizer_coarse(z_e_coarse)

        return codes_coarse, codes_fine

    def decode(self, codes_coarse, codes_fine):
        """Decode from coarse and fine codes to an image.

        Args:
            codes_coarse: [B, 16, 8] integer codes.
            codes_fine: [B, 64, 32] integer codes.

        Returns:
            x_recon: [B, 1, 256, 128] reconstructed image.
        """
        # Get quantized embeddings
        z_q_coarse = self.quantizer_coarse.codebook(codes_coarse).permute(0, 3, 1, 2).contiguous()
        z_q_fine = self.quantizer_fine.codebook(codes_fine).permute(0, 3, 1, 2).contiguous()

        # Decode coarse: 16x8 -> 64x32
        h = self.dec_coarse_in(z_q_coarse)
        h = self.dec_res1(h)
        h = self.dec_res2(h)
        h = F.relu(self.dec_upsample1(h))  # 16x8 -> 32x16
        h = F.relu(self.dec_upsample2(h))  # 32x16 -> 64x32

        # Combine upsampled coarse with fine codes
        h = torch.cat([h, z_q_fine], dim=1)
        h = self.dec_combine(h)
        h = self.dec_res3(h)
        h = self.dec_res4(h)

        # Upsample to image
        h = F.relu(self.dec_upsample3(h))  # 64x32 -> 128x64
        h = self.dec_upsample4(h)          # 128x64 -> 256x128
        return self.dec_out(h)

    def forward(self, x):
        """Forward pass for training.

        Returns:
            x_recon: Reconstructed image.
            vq_loss: Combined VQ loss from both levels.
            perplexity_coarse: Codebook usage at the coarse level.
            perplexity_fine: Codebook usage at the fine level.
            codes_coarse: Coarse code indices.
            codes_fine: Fine code indices.
        """
        # Encode to both levels
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = self.enc_res1(h)
        h = self.enc_res2(h)

        # Fine level
        z_e_fine = self.enc_fine(h)
        z_q_fine, vq_loss_fine, perplexity_fine, codes_fine = self.quantizer_fine(z_e_fine)

        # Coarse level
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = self.enc_res3(h)
        h = self.enc_res4(h)
        z_e_coarse = self.enc_coarse(h)
        z_q_coarse, vq_loss_coarse, perplexity_coarse, codes_coarse = self.quantizer_coarse(z_e_coarse)

        # Decode
        h = self.dec_coarse_in(z_q_coarse)
        h = self.dec_res1(h)
        h = self.dec_res2(h)
        h = F.relu(self.dec_upsample1(h))
        h = F.relu(self.dec_upsample2(h))
        h = torch.cat([h, z_q_fine], dim=1)
        h = self.dec_combine(h)
        h = self.dec_res3(h)
        h = self.dec_res4(h)
        h = F.relu(self.dec_upsample3(h))
        h = self.dec_upsample4(h)
        x_recon = self.dec_out(h)

        # Combine losses (equal weight for now, could be tuned)
        vq_loss = vq_loss_coarse + vq_loss_fine

        return x_recon, vq_loss, perplexity_coarse, perplexity_fine, codes_coarse, codes_fine


# --------------------------------------------------------------------------- #
# MAIN SOLUTION (2/2): causal Transformer prior over the fine code grid
# --------------------------------------------------------------------------- #
class TransformerPrior(nn.Module):
    """Causal transformer prior over raster-ordered code indices.

    This is the prior that produced anatomically coherent samples in the
    report (Section 6.5.4, Experiment 6). It replaces the PixelCNN because
    masked convolutions cannot see far enough up the 64-row code grid; causal
    self-attention lets every position attend to all preceding positions.

    Same interface as ``PixelCNNPrior``: forward(codes [B,H,W]) -> logits
    [B,K,H,W], loss(codes) -> mean cross-entropy in nats, sample(n, device, ...)
    -> codes.

    Sequence model: a learned start token is prepended to the raster-ordered
    codes, so position p sees only the start token and codes 0..p-1. A strict
    causal mask enforces this. Learned positional embeddings; pre-norm blocks.
    """

    def __init__(self, num_codes=512, width=256, num_layers=6, num_heads=8,
                 grid_hw=(64, 32), dropout=0.0):
        """Build the causal transformer.

        Args:
            num_codes: codebook size K (output vocabulary per position).
            width: embedding/attention width (must divide by num_heads).
            num_layers: number of transformer encoder layers.
            num_heads: number of self-attention heads.
            grid_hw: (height, width) of the code grid; sequence length is H*W.
            dropout: dropout probability inside the transformer blocks.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.num_codes = num_codes
        self.grid_hw = grid_hw
        n = grid_hw[0] * grid_hw[1]
        self.start = num_codes                       # extra token id for <start>
        self.tok = nn.Embedding(num_codes + 1, width)
        self.pos = nn.Parameter(torch.zeros(1, n, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width, num_heads, dim_feedforward=4 * width, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_codes)
        # True above the diagonal = masked (future positions).
        self.register_buffer("causal", torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1))

    def forward(self, codes):
        """codes: [B, H, W] integers -> logits [B, K, H, W]."""
        b = codes.shape[0]
        flat = codes.reshape(b, -1)
        start = torch.full((b, 1), self.start, dtype=flat.dtype, device=codes.device)
        inp = torch.cat([start, flat[:, :-1]], dim=1)   # position p sees codes < p
        n = inp.shape[1]
        x = self.tok(inp) + self.pos[:, :n]
        h = self.encoder(x, mask=self.causal[:n, :n], is_causal=True)
        logits = self.head(self.norm(h))                # [B, N, K]
        return logits.transpose(1, 2).reshape(b, self.num_codes, *self.grid_hw)

    def loss(self, codes):
        """Mean cross-entropy per latent position (in nats)."""
        return F.cross_entropy(self(codes), codes)

    @torch.no_grad()
    def sample(self, n, device, temperature=1.0, progress=None):
        """Sequential raster-order sampling. Recomputes the prefix each step (no KV cache)."""
        if n < 1 or temperature <= 0:
            raise ValueError("n and temperature must be positive")
        self.eval()
        h, w = self.grid_hw
        total = h * w
        seq = torch.full((n, 1), self.start, dtype=torch.long, device=device)
        for t in range(total):
            x = self.tok(seq) + self.pos[:, :t + 1]
            hid = self.encoder(x, mask=self.causal[:t + 1, :t + 1], is_causal=True)
            logits = self.head(self.norm(hid[:, -1])) / temperature
            seq = torch.cat([seq, torch.multinomial(F.softmax(logits, dim=1), 1)], dim=1)
            if progress is not None:
                progress(t + 1, total)
        return seq[:, 1:].reshape(n, h, w)


# --------------------------------------------------------------------------- #
# Re-exports: comparison baselines (kept for reproducibility)
# --------------------------------------------------------------------------- #
# These are not part of the submitted solution. They are imported here so the
# training/evaluation scripts can use a single ``from modules import ...``
# line for both the solution and the baselines it is compared against.
from models import (  # noqa: E402  (import after the solution classes by design)
    CondPixelCNNPrior,
    DCDiscriminator,
    DCGenerator,
    Decoder,
    Encoder,
    GatedMaskedLayer,
    GatedPixelCNNPrior,
    MaskedConv2d,
    PixelCNNPrior,
    VQVAE,
)

__all__ = [
    # Submitted solution
    "VQVAE2", "TransformerPrior",
    # Baselines / reproducibility
    "VQVAE", "Encoder", "Decoder",
    "MaskedConv2d", "PixelCNNPrior", "CondPixelCNNPrior",
    "GatedMaskedLayer", "GatedPixelCNNPrior",
    "DCGenerator", "DCDiscriminator",
    # Shared primitives
    "ResidualBlock", "VectorQuantizer",
]
