"""Shared building blocks used by both the main solution and the baselines.

``ResidualBlock`` and ``VectorQuantizer`` are the two primitives that the
submitted model (VQ-VAE-2 + Transformer prior, in ``modules.py``) and the
reproducibility baselines (single-level VQ-VAE, PixelCNN priors) both need.
They live here so neither side has to import the other, avoiding a circular
import between ``modules.py`` and the ``models`` package.

All code is pure PyTorch (no NumPy), as required by the course specification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block: x + W2 * relu(W1 * relu(x)) as used in the VQ-VAE paper.

    The identity shortcut preserves gradients, which matters because the
    encoder is trained through the straight-through estimator.
    """

    def __init__(self, channels):
        """Build the two-convolution residual branch for ``channels`` features.

        Args:
            channels: number of input and output feature channels (unchanged
                by the block, so the identity shortcut adds cleanly).
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        """Apply the residual branch and add the identity shortcut."""
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    """Nearest-neighbour vector quantisation with a K-entry codebook.

    Loss terms (Eq. 3 of van den Oord et al., 2017):
        L = recon ||x - x_hat||^2                       (handled by caller)
          + ||sg[z_e] - e||^2                            (codebook / VQ loss)
          + beta * ||z_e - sg[e]||^2                     (commitment loss)

    The codebook loss moves the embedding e towards the encoder output
    z_e; the commitment loss (weight beta=0.25 per the paper) makes the
    encoder output commit to the chosen embedding. The straight-through
    estimator copies decoder gradients from e to z_e, bypassing the
    non-differentiable nearest-neighbour argmin.

    Also tracks codebook usage: perplexity = exp(entropy of the average
    one-hot assignment distribution). Perplexity 1 means a single code is
    used for everything (collapse); perplexity K means uniform usage.
    """

    def __init__(self, num_codes=512, code_dim=64, beta=0.25):
        """Create the codebook and the usage-tracking buffer.

        Args:
            num_codes: codebook size K.
            code_dim: dimensionality D of each code vector.
            beta: commitment-loss weight (0.25 in the paper).
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        # Exponential moving average of the per-code assignment frequency,
        # used only for the perplexity diagnostic (not a loss).
        self.register_buffer("ema_counts", torch.zeros(num_codes))

    def forward(self, z_e):
        """Quantise continuous encoder outputs and return the VQ losses.

        Args:
            z_e: encoder output of shape [B, D, H, W].

        Returns:
            quantised_st: straight-through quantised features [B, D, H, W].
            loss: codebook + beta * commitment loss (scalar).
            perplexity: exp(entropy) of the batch-average code distribution.
            indices: chosen code indices of shape [B, H, W].
        """
        # [B*H*W, D] vs [K, D] -> pairwise squared distances [B*H*W, K]
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.code_dim)
        # Chunk the search: a full [B*H*W,K] matrix can consume >1 GB.
        with torch.no_grad():
            weights = self.codebook.weight
            indices = torch.cat([
                (chunk.square().sum(1, keepdim=True)
                 - 2 * chunk @ weights.t()
                 + weights.square().sum(1)).argmin(1)
                for chunk in flat.split(8192)
            ])
        b, _, h, w = z_e.shape
        quantised = self.codebook(indices).view(b, h, w, self.code_dim)
        quantised = quantised.permute(0, 3, 1, 2).contiguous()

        # Losses (sg = stop-gradient).
        codebook_loss = F.mse_loss(quantised, z_e.detach())
        commitment_loss = F.mse_loss(z_e, quantised.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator: decoder sees quantised codes, but
        # gradients flow straight to the encoder output.
        quantised_st = z_e + (quantised - z_e).detach()

        # Usage diagnostics: average assignment distribution over the batch.
        counts = torch.bincount(indices, minlength=self.num_codes).to(flat.dtype)
        avg_probs = counts / counts.sum()
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
        if self.training:
            self.ema_counts.mul_(0.99).add_(counts, alpha=0.01)

        return quantised_st, loss, perplexity, indices.view(z_e.shape[0], *z_e.shape[2:])
