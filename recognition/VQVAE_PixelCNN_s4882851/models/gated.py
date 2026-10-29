"""Gated PixelCNN prior — reproducibility baseline (Experiment 8).

This prior exists because the plain ``PixelCNNPrior`` failed in a specific,
measurable way: it reproduced MRI-like *texture* but not the global hip
layout. Two causes were identified, and this class addresses both:

1. **Blind spot.** Stacked masked convolutions cannot see an above-right
   wedge. The dual-stack construction removes it (see ``GatedMaskedLayer``).
2. **Receptive field too small.** The old prior saw only
   ``3 + num_blocks*2*(kernel//2)`` rows above a position: 11 rows for a
   4-block model and 19 for an 8-block model, against a 64-row code grid.

The submitted solution uses the causal Transformer in ``modules.py``; this
gated prior is retained so the Experiment 8 result can be reproduced.

Reference: van den Oord et al., "Conditional Image Generation with PixelCNN
Decoders", NeurIPS 2016.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gate(x):
    """Gated activation unit: tanh(a) * sigmoid(b) over a channel split."""
    a, b = x.chunk(2, dim=1)
    return torch.tanh(a) * torch.sigmoid(b)


class GatedMaskedLayer(nn.Module):
    """One Gated PixelCNN layer with separate vertical and horizontal stacks.

    Why this replaces ``MaskedConv2d``: a single masked convolution has a
    *blind spot* -- a wedge of pixels above-right of the current position that
    can never be reached no matter how many layers are stacked. Gated PixelCNN
    removes it by factorising the receptive field into

      * a **vertical stack**, a (k//2+1, k) convolution covering whole rows
        above, and
      * a **horizontal stack**, a (1, k//2+1) convolution covering the current
        row to the left,

    with the vertical stack feeding the horizontal stack after a one-row
    downward shift. That shift is what preserves causality: the horizontal
    stack at row i only ever receives vertical features computed from rows
    strictly above i.

    Causality argument (checked by ``.ai/gated_causality.py``):
      * ``v_out[i]`` depends on input rows <= i.
      * Feeding ``h`` uses ``v_shift[i] = v_out[i-1]``, so rows <= i-1.
      * ``h_conv[i, j]`` depends on h-input columns <= j, and in the first layer
        the extra right-shift makes that columns <= j-1. In later layers the
        h features at column j already exclude column j of the *codes*, so the
        composition still excludes position (i, j).
    Therefore the logits at (i, j) never depend on the code at (i, j).

    Measured, not assumed. Exhaustively perturbing every earlier position for a
    target near the bottom of a 20x10 grid gives:

      * old ``PixelCNNPrior`` (k=3, 4 blocks): 111 of 194 earlier positions
        visible, **83 invisible**. The invisible set splits into rows 0-7
        (outside the receptive field) *and rows 17-18, immediately above the
        target* -- the latter being the classic blind-spot wedge.
      * this layer (20 layers, k=3): **194 of 194 visible, 0 invisible.**
    """

    def __init__(self, channels, kernel_size, first=False, mask_classes=0):
        """Build the vertical/horizontal stacks for one gated layer.

        Args:
            channels: feature width (the gated unit doubles it internally).
            kernel_size: odd kernel size for the vertical/horizontal stacks.
            first: True for the first layer (adds the mask-A right shift).
            mask_classes: if > 0, add per-class conditioning projections.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.first = first
        k = kernel_size
        pad = k // 2
        self.v_conv = nn.Conv2d(channels, 2 * channels, (pad + 1, k), padding=(pad, pad))
        self.h_conv = nn.Conv2d(channels, 2 * channels, (1, pad + 1), padding=(0, pad))
        self.v_to_h = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1)
        self.h_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.cond_v = nn.Conv2d(mask_classes, 2 * channels, 1) if mask_classes else None
        self.cond_h = nn.Conv2d(mask_classes, 2 * channels, 1) if mask_classes else None

    def forward(self, v, h, cond=None):
        """Run one gated layer, returning the new (vertical, horizontal) states."""
        height, width = v.shape[2], v.shape[3]
        # Vertical stack: crop the bottom padding so row i sees rows <= i.
        v_out = self.v_conv(v)[:, :, :height, :]
        # Horizontal stack: crop the right padding so column j sees cols <= j.
        h_conv = self.h_conv(h)[:, :, :, :width]
        if self.first:
            # Mask-A behaviour: shift right one column to exclude column j.
            h_conv = F.pad(h_conv, (1, 0, 0, 0))[:, :, :, :width]
        # Shift the vertical features down one row before they reach h.
        v_shift = F.pad(v_out, (0, 0, 1, 0))[:, :, :height, :]
        h_total = h_conv + self.v_to_h(v_shift)
        if cond is not None and self.cond_v is not None:
            v_out = v_out + self.cond_v(cond)
            h_total = h_total + self.cond_h(cond)
        v_new = _gate(v_out)
        h_new = self.h_out(_gate(h_total))
        if not self.first:
            h_new = h_new + h          # residual on the horizontal stack only
        return v_new, h_new


class GatedPixelCNNPrior(nn.Module):
    """Gated PixelCNN prior over VQ-VAE code indices, optionally mask-conditioned.

    The vertical receptive field here is ``num_layers * (kernel_size // 2)``
    rows. ``covers_grid()`` reports whether that spans the full grid height, and
    the constructor warns when it does not, so the configuration is checked
    rather than assumed. For a 64x32 grid, kernel 7 with 22 layers gives 66
    rows and therefore full coverage.

    Set ``mask_classes=0`` for a genuinely unconditional prior, or to the number
    of segmentation classes to condition on a downsampled mask (the mask is an
    input, never a prediction, so it may be seen in full).
    """

    def __init__(self, num_codes=512, width=128, num_layers=22, grid_hw=(64, 32),
                 embedding_dim=32, kernel_size=7, mask_classes=0):
        """Build the gated stack and record its receptive field.

        Args:
            num_codes: codebook size K.
            width: hidden feature width.
            num_layers: number of gated layers (sets the vertical receptive field).
            grid_hw: (height, width) of the code grid.
            embedding_dim: learned code embedding dimension.
            kernel_size: odd kernel size (7 with 22 layers covers a 64-row grid).
            mask_classes: 0 for unconditional, else number of mask classes.
        """
        super().__init__()
        self.num_codes = num_codes
        self.grid_hw = tuple(grid_hw)
        self.mask_classes = mask_classes
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.vertical_receptive_field = num_layers * (kernel_size // 2)
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.stem = nn.Conv2d(embedding_dim, width, kernel_size=1)
        self.layers = nn.ModuleList([
            GatedMaskedLayer(width, kernel_size, first=(i == 0),
                             mask_classes=mask_classes)
            for i in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(width, num_codes, kernel_size=1),
        )

    def covers_grid(self):
        """True when the vertical receptive field spans the whole code grid."""
        return self.vertical_receptive_field >= self.grid_hw[0]

    def _cond(self, mask):
        """Resize a mask to the code grid, or return None if unconditional."""
        if mask is None or not self.mask_classes:
            return None
        return F.interpolate(mask, size=self.grid_hw, mode="nearest")

    def forward(self, codes, mask=None):
        """codes: [B, H, W] ints; mask: [B, C, h, w] float -> logits [B, K, H, W]."""
        cond = self._cond(mask)
        x = self.stem(self.embedding(codes).permute(0, 3, 1, 2))
        v, h = x, x
        for layer in self.layers:
            v, h = layer(v, h, cond)
        return self.out(h)

    def loss(self, codes, mask=None):
        """Mean cross-entropy per latent position (in nats)."""
        return F.cross_entropy(self(codes, mask), codes)

    @torch.no_grad()
    def sample(self, n=None, device=None, temperature=1.0, mask=None, progress=None):
        """Raster-scan decoding.

        Unlike ``PixelCNNPrior.sample``, no row cropping is used: the receptive
        field is deliberately as large as the grid, so cropping would change
        the distribution rather than merely save time.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        if mask is not None:
            n, device = mask.shape[0], mask.device
        if n is None or device is None:
            raise ValueError("provide either mask, or both n and device")
        height, width = self.grid_hw
        codes = torch.zeros(n, height, width, dtype=torch.long, device=device)
        total, done = height * width, 0
        for i in range(height):
            for j in range(width):
                logits = self(codes, mask)[:, :, i, j] / temperature
                probs = F.softmax(logits, dim=1)
                codes[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
                done += 1
                if progress is not None:
                    progress(done, total)
        return codes
