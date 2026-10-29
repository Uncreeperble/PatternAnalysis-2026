"""Single-level VQ-VAE (van den Oord et al., 2017) — reproducibility baseline.

This is the original single-level model used for the predeclared campaign and
the DCGAN comparison. The submitted solution extends it to the hierarchical
VQ-VAE-2 in ``modules.py``; this file is kept so the single-level results in
the report can be reproduced exactly.

Design rationale (matching the feasibility review):
  * Encoder downsamples 4x with two 4x4 stride-2 convolutions: 256x128 input
    -> 128x64 -> 64x32 latent grid. Downsampling 8x was considered but 4x
    retains more anatomical detail at a PixelCNN sampling cost of 2048
    tokens per image. An optional third stride-2 stage provides an 8x
    (32x16) grid ablation.
  * Two residual blocks as in the paper (ReLU-Conv-ReLU-Conv plus identity),
    which ease gradient flow in a shallow encoder.
  * Codebook: K=512 entries of dimension D=64, commitment weight beta=0.25,
    straight-through estimator for decoder gradients, Adam at 2e-4.
  * Channel width reduced to 128 (from 256 in the paper) to keep the model
    inside the measured GPU memory and time budget (feasibility smoke test:
    1.18M parameters, 3.39 GB peak, 6.9 s/epoch on an A100-40GB).

All model code is pure PyTorch (no NumPy) as required by the course spec.
"""
import torch
import torch.nn as nn

from .primitives import ResidualBlock, VectorQuantizer


class Encoder(nn.Module):
    """Convolutional encoder mapping a 256x128 slice to a 64x32xD latent grid.

    Architecture per the VQ-VAE paper: two stride-2 convolutions (4x4
    kernels) downsample 4x, followed by two residual blocks and a 1x1
    convolution projecting to the codebook embedding dimension D.
    """

    def __init__(self, in_channels=1, width=128, latent_dim=64, downsample=4):
        """Build the encoder stack.

        Args:
            in_channels: input image channels (1 for grayscale MRI).
            width: hidden feature width (128, reduced from the paper's 256).
            latent_dim: output embedding dimension D.
            downsample: 4 (64x32 grid) or 8 (32x16 grid ablation).
        """
        super().__init__()
        if downsample not in (4, 8):
            raise ValueError("downsample must be 4 or 8")
        extra = [nn.Conv2d(width, width, 4, stride=2, padding=1), nn.ReLU()] if downsample == 8 else []
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=4, stride=2, padding=1),  # 256x128 -> 128x64
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=4, stride=2, padding=1),       # 128x64  -> 64x32
            nn.ReLU(),
            *extra,
            ResidualBlock(width),
            ResidualBlock(width),
            nn.Conv2d(width, latent_dim, kernel_size=1),                       # project to D
        )

    def forward(self, x):
        """Encode an image batch [B, C, 256, 128] to [B, D, H, W]."""
        return self.net(x)


class Decoder(nn.Module):
    """Mirror of the encoder: 64x32xD latent grid back to a 256x128 slice.

    Transposed 4x4 stride-2 convolutions invert the encoder downsampling;
    a final sigmoid bounds outputs to [0, 1] to match the normalised data.
    """

    def __init__(self, out_channels=1, width=128, latent_dim=64, downsample=4):
        """Build the decoder stack.

        Args:
            out_channels: output image channels (1 for grayscale MRI).
            width: hidden feature width.
            latent_dim: input embedding dimension D.
            downsample: 4 or 8, matching the encoder.
        """
        super().__init__()
        extra = [nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.ReLU()] if downsample == 8 else []
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, width, kernel_size=1),                         # lift from D
            ResidualBlock(width),
            ResidualBlock(width),
            *extra,
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),  # 64x32 -> 128x64
            nn.ReLU(),
            nn.ConvTranspose2d(width, out_channels, kernel_size=4, stride=2, padding=1),  # -> 256x128
            nn.Sigmoid(),
        )

    def forward(self, z):
        """Decode quantised features [B, D, H, W] to an image [B, C, 256, 128]."""
        return self.net(z)


class VQVAE(nn.Module):
    """Encoder + vector quantiser + decoder (single-level VQ-VAE).

    forward returns (reconstruction, vq_loss, perplexity, code_indices)
    so the training loop can log utilisation and the PixelCNN prior can
    later be trained on the discrete indices.
    """

    def __init__(self, in_channels=1, width=128, latent_dim=64,
                 num_codes=512, beta=0.25, downsample=4):
        """Assemble encoder, quantiser and decoder.

        Args:
            in_channels: image channels.
            width: hidden feature width.
            latent_dim: codebook embedding dimension D.
            num_codes: codebook size K.
            beta: commitment-loss weight.
            downsample: 4 or 8.
        """
        super().__init__()
        self.encoder = Encoder(in_channels, width, latent_dim, downsample)
        self.quantizer = VectorQuantizer(num_codes, latent_dim, beta)
        self.decoder = Decoder(in_channels, width, latent_dim, downsample)

    def forward(self, x):
        """Reconstruct ``x`` and return (recon, vq_loss, perplexity, codes)."""
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, codes = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, perplexity, codes

    @torch.no_grad()
    def encode_to_codes(self, x):
        """Map images to integer code indices [B, H, W] (for prior training)."""
        return self.quantizer(self.encoder(x))[3]

    @torch.no_grad()
    def decode_from_codes(self, codes):
        """Decode integer code indices [B, H, W] back to images."""
        z_q = self.quantizer.codebook(codes).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)
