"""Masked PixelCNN priors over VQ-VAE code indices — reproducibility baselines.

Two priors live here:

* ``PixelCNNPrior`` — the original masked-convolution prior used in the
  predeclared campaign. It is the model whose limited receptive field the
  report diagnoses as the generation bottleneck.
* ``CondPixelCNNPrior`` — an exploratory variant conditioned on a downsampled
  segmentation mask (Experiment 2).

The submitted solution replaces the fine prior with the causal Transformer in
``modules.py``; these are retained so the PixelCNN results can be reproduced.

Reference: van den Oord et al., "Conditional Image Generation with PixelCNN
Decoders", NeurIPS 2016.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Module):
    """Convolution with a raster-scan receptive-field mask.

    The VQ-VAE alone cannot sample: it maps images to codes but learns
    nothing about which code grids are plausible. A masked residual
    PixelCNN over the discrete indices supplies that distribution.
    Type-A masking is used for the first layer (excludes the centre pixel),
    type-B for all later layers (includes the centre), which enforces the
    autoregressive factorisation p(z) = prod_ij p(z_ij | z_<ij) over the
    64x32 latent grid.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding):
        """Create the convolution and its fixed (non-learned) mask buffer.

        Args:
            mask_type: "A" (exclude centre) or "B" (include centre).
            in_channels: input feature channels.
            out_channels: output feature channels.
            kernel_size: odd convolution kernel size.
            padding: symmetric padding for the convolution.
        """
        super().__init__()
        assert mask_type in ("A", "B")
        self.mask_type = mask_type
        # Biases allow an unconditional marginal at the first raster position.
        # With every bias disabled, its logits are identically zero forever.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=True)
        # The mask is a fixed buffer (not learned), built once below.
        self.register_buffer("mask", torch.zeros_like(self.conv.weight))

        k = kernel_size
        o = k // 2
        self.mask[:, :, :o, :] = 1.0                     # all rows above centre
        self.mask[:, :, o, :o] = 1.0                     # left of centre
        if mask_type == "B":                             # include centre pixel
            self.mask[:, :, o, o] = 1.0

    def forward(self, x):
        """Apply the masked convolution (mask multiplied in at use time)."""
        # Apply the mask to the weights at use time (not in-place), so the
        # autoregressive property holds while parameters stay clean.
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, self.conv.bias, padding=self.conv.padding[0])


class PixelCNNPrior(nn.Module):
    """Autoregressive prior over the VQ-VAE code indices.

    Input: integer code grids [B, H, W] (K possible values per position).
    Output: logits [B, K, H, W] giving p(code | previous codes in scan order).

    Architecture: a type-A 7x7 masked convolution lifts embedded codes to
    the feature width, followed by residual blocks of two type-B masked
    convolutions, and a 1x1 masked convolution projects back to K logits.
    Residual connections are used because deeper masked stacks otherwise
    train slowly (a known PixelCNN limitation; gated activations are the
    paper's alternative and can be tried if validation NLL plateaus).

    A learned 32-dimensional embedding replaces the K-channel one-hot
    input. The first mask remains type A, so the current code cannot leak.
    Residual kernels default to 3x3 to make sequential inference practical;
    --prior-kernel 7 is a preplanned receptive-field/capacity ablation.
    """

    def __init__(self, num_codes=512, width=128, num_blocks=4, grid_hw=(64, 32),
                 embedding_dim=32, kernel_size=3):
        """Build the masked residual stack.

        Args:
            num_codes: codebook size K (number of output logits per position).
            width: hidden feature width.
            num_blocks: number of residual blocks (each two masked convs).
            grid_hw: (height, width) of the code grid.
            embedding_dim: dimension of the learned code embedding.
            kernel_size: 3 or 7 for the residual masked convolutions.
        """
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("prior kernel must be 3 or 7")
        self.num_codes = num_codes
        self.grid_hw = grid_hw
        self.row_radius = 3 + num_blocks * 2 * (kernel_size // 2)
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.input = MaskedConv2d("A", embedding_dim, width, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                "conv1": MaskedConv2d("B", width, width, kernel_size, kernel_size // 2),
                "conv2": MaskedConv2d("B", width, width, kernel_size, kernel_size // 2),
            }))
        self.out = MaskedConv2d("B", width, num_codes, kernel_size=1, padding=0)

    def forward(self, codes):
        """codes: [B, H, W] integers -> logits [B, K, H, W]."""
        x = self.embedding(codes).permute(0, 3, 1, 2).contiguous()
        x = self.input(x)
        for blk in self.blocks:
            h = F.relu(blk["conv1"](F.relu(x)))
            x = x + blk["conv2"](h)
        return self.out(x)

    def loss(self, codes):
        """Mean cross-entropy per latent position (in nats)."""
        logits = self(codes)
        return F.cross_entropy(logits, codes)

    @torch.no_grad()
    def sample(self, n, device, temperature=1.0, progress=None):
        """Draw n code grids by sequential raster-scan decoding.

        Each position is drawn from a softmax over the K logits (scaled by
        ``temperature``; 1.0 = the learned distribution). Decoding row-major
        guarantees every position is conditioned only on its predecessors,
        matching the masked receptive fields used in training.

        ``progress(done, total)`` is an optional callback so callers can show
        a progress bar/ETA (sequential decoding of 2048 positions is slow
        enough to look frozen without one); the model itself stays free of
        non-PyTorch dependencies.
        """
        if n < 1 or temperature <= 0:
            raise ValueError("n and temperature must be positive")
        self.eval()
        h, w = self.grid_hw
        codes = torch.zeros(n, h, w, dtype=torch.long, device=device)
        total, done = h * w, 0
        for i in range(h):
            for j in range(w):
                # Only preceding rows inside the finite receptive field
                # can affect this position. Regression-tested against the
                # full-grid logits; no change to the sampled distribution.
                top = max(0, i - self.row_radius)
                logits = self(codes[:, top:i + 1])[:, :, i - top, j] / temperature
                probs = F.softmax(logits, dim=1)
                codes[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
                done += 1
                if progress is not None:
                    progress(done, total)
        return codes


class CondPixelCNNPrior(nn.Module):
    """PixelCNN prior conditioned on a downsampled segmentation mask.

    Motivation: the unconditional prior produces MRI-like texture but loses the
    global hip layout, because each position only sees ``row_radius`` rows above
    it. The mask is an input (not a quantity being predicted), so it may be seen
    in full at every position without breaking causality over the codes.

    Conditioning: the mask is resized to the code grid, embedded per class by a
    1x1 convolution, and added as a spatial bias to the features before each
    residual block (as in the conditional PixelCNN of van den Oord et al. 2016).

    Interface matches PixelCNNPrior except that forward/loss/sample take a mask:
      forward(codes [B,H,W], mask [B,C,H,W]) -> logits [B,K,H,W]
      sample(mask, temperature) -> codes, one grid per mask row.
    """

    def __init__(self, num_codes=512, width=128, num_blocks=8, grid_hw=(64, 32),
                 embedding_dim=32, kernel_size=3, mask_classes=6):
        """Build the conditioned masked residual stack.

        Args:
            num_codes: codebook size K.
            width: hidden feature width.
            num_blocks: number of residual blocks.
            grid_hw: (height, width) of the code grid.
            embedding_dim: learned code embedding dimension.
            kernel_size: 3 or 7 for the residual masked convolutions.
            mask_classes: number of segmentation classes to condition on.
        """
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("prior kernel must be 3 or 7")
        self.num_codes = num_codes
        self.grid_hw = grid_hw
        self.mask_classes = mask_classes
        self.row_radius = 3 + num_blocks * 2 * (kernel_size // 2)
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.input = MaskedConv2d("A", embedding_dim, width, kernel_size=7, padding=3)
        self.cond = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # 1x1 conditioning projections are unmasked on purpose: the mask is
            # given, so it carries no information about unseen code positions.
            self.cond.append(nn.Conv2d(mask_classes, width, kernel_size=1))
            self.blocks.append(nn.ModuleDict({
                "conv1": MaskedConv2d("B", width, width, kernel_size, kernel_size // 2),
                "conv2": MaskedConv2d("B", width, width, kernel_size, kernel_size // 2),
            }))
        self.out = MaskedConv2d("B", width, num_codes, kernel_size=1, padding=0)

    def forward(self, codes, mask):
        """codes: [B,H,W] ints; mask: [B,C,h,w] float -> logits [B,K,H,W]."""
        x = self.embedding(codes).permute(0, 3, 1, 2).contiguous()
        x = self.input(x)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        for blk, cnd in zip(self.blocks, self.cond):
            h = F.relu(blk["conv1"](F.relu(x + cnd(mask))))
            x = x + blk["conv2"](h)
        return self.out(x)

    def loss(self, codes, mask):
        """Mean cross-entropy per latent position, conditioned on ``mask``."""
        return F.cross_entropy(self(codes, mask), codes)

    @torch.no_grad()
    def sample(self, mask, temperature=1.0, progress=None):
        """One code grid per mask row, by raster-scan decoding (same as PixelCNNPrior)."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        device = mask.device
        n = mask.shape[0]
        h, w = self.grid_hw
        codes = torch.zeros(n, h, w, dtype=torch.long, device=device)
        small = F.interpolate(mask, size=(h, w), mode="nearest")
        total, done = h * w, 0
        for i in range(h):
            for j in range(w):
                top = max(0, i - self.row_radius)
                logits = self(codes[:, top:i + 1], small[:, :, top:i + 1])[:, :, i - top, j]
                probs = F.softmax(logits / temperature, dim=1)
                codes[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
                done += 1
                if progress is not None:
                    progress(done, total)
        return codes
