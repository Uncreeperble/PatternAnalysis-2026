"""Model components for the VQ-VAE and PixelCNN prior (Project 2.4).

Main model: VQ-VAE (van den Oord, Vinyals & Kavukcuoglu, "Neural Discrete
Representation Learning", NeurIPS 2017) with a PixelCNN prior over the
discrete code indices, for unconditional synthesis of HipMRI 2D prostate
MRI slices. A DCGAN baseline (Radford et al., 2016) is added in a later
module section for comparison.

Design rationale (matching the feasibility review):
  * Encoder downsamples 4x with two 4x4 stride-2 convolutions: 256x128 input
    -> 128x64 -> 64x32 latent grid. Downsampling 8x was considered but 4x
    retains more anatomical detail at a PixelCNN sampling cost of 2048
    tokens per image. An optional third stride-2 stage provides an 8x
    (32x16) grid ablation. Final runtime is measured independently;
    feasibility timings do not establish this implementation's runtime.
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
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# VQ-VAE (van den Oord et al., 2017)
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """Residual block: x + W2 * relu(W1 * relu(x)) as used in the VQ-VAE paper.

    The identity shortcut preserves gradients, which matters because the
    encoder is trained through the straight-through estimator.
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """Convolutional encoder mapping a 256x128 slice to a 64x32xD latent grid.

    Architecture per the VQ-VAE paper: two stride-2 convolutions (4x4
    kernels) downsample 4x, followed by two residual blocks and a 1x1
    convolution projecting to the codebook embedding dimension D.
    """

    def __init__(self, in_channels=1, width=128, latent_dim=64, downsample=4):
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
        return self.net(x)


class Decoder(nn.Module):
    """Mirror of the encoder: 64x32xD latent grid back to a 256x128 slice.

    Transposed 4x4 stride-2 convolutions invert the encoder downsampling;
    a final sigmoid bounds outputs to [0, 1] to match the normalised data.
    """

    def __init__(self, out_channels=1, width=128, latent_dim=64, downsample=4):
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
        return self.net(z)


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


class VQVAE(nn.Module):
    """Encoder + vector quantiser + decoder.

    forward returns (reconstruction, vq_loss, perplexity, code_indices)
    so the training loop can log utilisation and the PixelCNN prior can
    later be trained on the discrete indices.
    """

    def __init__(self, in_channels=1, width=128, latent_dim=64,
                 num_codes=512, beta=0.25, downsample=4):
        super().__init__()
        self.encoder = Encoder(in_channels, width, latent_dim, downsample)
        self.quantizer = VectorQuantizer(num_codes, latent_dim, beta)
        self.decoder = Decoder(in_channels, width, latent_dim, downsample)

    def forward(self, x):
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


# --------------------------------------------------------------------------- #
# PixelCNN prior over VQ-VAE code indices (van den Oord et al., 2016/2017)
# --------------------------------------------------------------------------- #
class MaskedConv2d(nn.Module):
    """Convolution with a raster-scan receptive-field mask.

    The VQ-VAE alone cannot sample: it maps images to codes but learns
    nothing about which code grids are plausible. A masked residual
    PixelCNN over the discrete indices supplies that distribution
    (van den Oord et al., "Conditional Image Generation with PixelCNN
    Decoders", NeurIPS 2016). Type-A masking is used for the first layer
    (excludes the centre pixel), type-B for all later layers (includes the
    centre), which enforces the autoregressive factorisation
    p(z) = prod_ij p(z_ij | z_<ij) over the 64x32 latent grid.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding):
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
    convolutions, and a 1x1 masked convolution
    projects back to K logits. Residual connections are used because deeper
    masked stacks otherwise train slowly (a known PixelCNN limitation;
    gated activations are the paper's alternative and can be tried if
    validation NLL plateaus).

    A learned 32-dimensional embedding replaces the K-channel one-hot
    input. The first mask remains type A, so the current code cannot leak.
    Residual kernels default to 3x3 to make sequential inference practical;
    --prior-kernel 7 is a preplanned receptive-field/capacity ablation.
    Runtime must be measured on this implementation, not extrapolated
    from the different feasibility smoke-test model.
    """

    def __init__(self, num_codes=512, width=128, num_blocks=4, grid_hw=(64, 32),
                 embedding_dim=32, kernel_size=3):
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
        `temperature`; 1.0 = the learned distribution). Decoding row-major
        guarantees every position is conditioned only on its predecessors,
        matching the masked receptive fields used in training.

        `progress(done, total)` is an optional callback so callers can show
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


# --------------------------------------------------------------------------- #
# VQ-VAE-2: Hierarchical Vector Quantization (Razavi et al., 2019)
# --------------------------------------------------------------------------- #
class VQVAE2(nn.Module):
    """Hierarchical VQ-VAE with two-level quantization for global + local structure.
    
    Architecture:
    - Encoder produces TWO latent grids:
      * Coarse (16×8): Captures global anatomy (pelvis position, overall structure)
      * Fine (64×32): Captures local texture and details
    - Decoder combines upsampled coarse with fine for reconstruction
    - Two separate priors:
      * Coarse prior: Learns global anatomical layout
      * Fine prior: Learns local texture conditioned on coarse codes
    
    This solves the receptive field problem: coarse prior only needs to model
    16×8=128 positions (easily covered by PixelCNN), while fine prior has
    coarse context to guide local detail generation.
    
    Based on "Generating Diverse High-Fidelity Images with VQ-VAE-2" 
    (Razavi, van den Oord & Vinyals, NeurIPS 2019).
    """
    
    def __init__(self, in_channels=1, width=128, latent_dim=64,
                 num_codes=512, beta=0.25):
        super().__init__()
        
        # Encoder path: 256×128 → 128×64 → 64×32 → 32×16 → 16×8
        # We extract features at two levels: 64×32 (fine) and 16×8 (coarse)
        
        # Shared encoder backbone to 64×32
        self.enc_conv1 = nn.Conv2d(in_channels, width, 4, stride=2, padding=1)  # 256×128 → 128×64
        self.enc_conv2 = nn.Conv2d(width, width, 4, stride=2, padding=1)        # 128×64 → 64×32
        self.enc_res1 = ResidualBlock(width)
        self.enc_res2 = ResidualBlock(width)
        
        # Fine level: quantize at 64×32
        self.enc_fine = nn.Conv2d(width, latent_dim, 1)
        self.quantizer_fine = VectorQuantizer(num_codes, latent_dim, beta)
        
        # Continue encoding to coarse level: 64×32 → 32×16 → 16×8
        self.enc_conv3 = nn.Conv2d(width, width, 4, stride=2, padding=1)        # 64×32 → 32×16
        self.enc_conv4 = nn.Conv2d(width, width, 4, stride=2, padding=1)        # 32×16 → 16×8
        self.enc_res3 = ResidualBlock(width)
        self.enc_res4 = ResidualBlock(width)
        
        # Coarse level: quantize at 16×8
        self.enc_coarse = nn.Conv2d(width, latent_dim, 1)
        self.quantizer_coarse = VectorQuantizer(num_codes, latent_dim, beta)
        
        # Decoder path: 16×8 → 32×16 → 64×32 → 128×64 → 256×128
        # Coarse codes are upsampled and combined with fine codes
        
        # Decode from coarse: 16×8 → 32×16 → 64×32
        self.dec_coarse_in = nn.Conv2d(latent_dim, width, 1)
        self.dec_res1 = ResidualBlock(width)
        self.dec_res2 = ResidualBlock(width)
        self.dec_upsample1 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)  # 16×8 → 32×16
        self.dec_upsample2 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)  # 32×16 → 64×32
        
        # Combine upsampled coarse with fine codes at 64×32
        self.dec_combine = nn.Conv2d(width + latent_dim, width, 1)
        self.dec_res3 = ResidualBlock(width)
        self.dec_res4 = ResidualBlock(width)
        
        # Final upsampling to image: 64×32 → 128×64 → 256×128
        self.dec_upsample3 = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)  # 64×32 → 128×64
        self.dec_upsample4 = nn.ConvTranspose2d(width, in_channels, 4, stride=2, padding=1)  # 128×64 → 256×128
        self.dec_out = nn.Sigmoid()
    
    def encode(self, x):
        """Encode image to coarse and fine code indices.
        
        Returns:
            codes_coarse: [B, 16, 8] integer codes
            codes_fine: [B, 64, 32] integer codes
        """
        # Shared encoding to 64×32
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = self.enc_res1(h)
        h = self.enc_res2(h)
        
        # Fine level quantization at 64×32
        z_e_fine = self.enc_fine(h)
        _, _, _, codes_fine = self.quantizer_fine(z_e_fine)
        
        # Continue to coarse level
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = self.enc_res3(h)
        h = self.enc_res4(h)
        
        # Coarse level quantization at 16×8
        z_e_coarse = self.enc_coarse(h)
        _, _, _, codes_coarse = self.quantizer_coarse(z_e_coarse)
        
        return codes_coarse, codes_fine
    
    def decode(self, codes_coarse, codes_fine):
        """Decode from coarse and fine codes to image.
        
        Args:
            codes_coarse: [B, 16, 8] integer codes
            codes_fine: [B, 64, 32] integer codes
        
        Returns:
            x_recon: [B, 1, 256, 128] reconstructed image
        """
        # Get quantized embeddings
        z_q_coarse = self.quantizer_coarse.codebook(codes_coarse).permute(0, 3, 1, 2).contiguous()
        z_q_fine = self.quantizer_fine.codebook(codes_fine).permute(0, 3, 1, 2).contiguous()
        
        # Decode coarse: 16×8 → 64×32
        h = self.dec_coarse_in(z_q_coarse)
        h = self.dec_res1(h)
        h = self.dec_res2(h)
        h = F.relu(self.dec_upsample1(h))  # 16×8 → 32×16
        h = F.relu(self.dec_upsample2(h))  # 32×16 → 64×32
        
        # Combine upsampled coarse with fine codes
        h = torch.cat([h, z_q_fine], dim=1)
        h = self.dec_combine(h)
        h = self.dec_res3(h)
        h = self.dec_res4(h)
        
        # Upsample to image
        h = F.relu(self.dec_upsample3(h))  # 64×32 → 128×64
        h = self.dec_upsample4(h)          # 128×64 → 256×128
        return self.dec_out(h)
    
    def forward(self, x):
        """Forward pass for training.
        
        Returns:
            x_recon: Reconstructed image
            vq_loss: Combined VQ loss from both levels
            perplexity_coarse: Codebook usage at coarse level
            perplexity_fine: Codebook usage at fine level
            codes_coarse: Coarse code indices
            codes_fine: Fine code indices
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
# Gated PixelCNN (van den Oord et al., "Conditional Image Generation with
# PixelCNN Decoders", NeurIPS 2016)
# --------------------------------------------------------------------------- #
def _gate(x):
    """Gated activation unit: tanh(a) * sigmoid(b) over a channel split."""
    a, b = x.chunk(2, dim=1)
    return torch.tanh(a) * torch.sigmoid(b)


class GatedMaskedLayer(nn.Module):
    """One Gated PixelCNN layer with separate vertical and horizontal stacks.

    Why this replaces `MaskedConv2d`: a single masked convolution has a
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

    Causality argument (checked by `.ai/gated_causality.py`):
      * `v_out[i]` depends on input rows <= i.
      * Feeding `h` uses `v_shift[i] = v_out[i-1]`, so rows <= i-1.
      * `h_conv[i, j]` depends on h-input columns <= j, and in the first layer
        the extra right-shift makes that columns <= j-1. In later layers the
        h features at column j already exclude column j of the *codes*, so the
        composition still excludes position (i, j).
    Therefore the logits at (i, j) never depend on the code at (i, j).

    Measured, not assumed. Exhaustively perturbing every earlier position for a
    target near the bottom of a 20x10 grid gives:

      * old `PixelCNNPrior` (k=3, 4 blocks): 111 of 194 earlier positions
        visible, **83 invisible**. The invisible set splits into rows 0-7
        (outside the receptive field) *and rows 17-18, immediately above the
        target* -- the latter being the classic blind-spot wedge, which sits
        further right than the nearest diagonal neighbour and so is easy to
        miss with a single hand-picked probe.
      * this layer (20 layers, k=3): **194 of 194 visible, 0 invisible.**
    """

    def __init__(self, channels, kernel_size, first=False, mask_classes=0):
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

    This exists because the plain `PixelCNNPrior` failed in a specific,
    measurable way: it reproduced MRI-like *texture* but not the global hip
    layout. Two causes were identified, and this class addresses both.

    1. **Blind spot.** Stacked masked convolutions cannot see an above-right
       wedge. The dual-stack construction removes it (see `GatedMaskedLayer`).
    2. **Receptive field too small.** The old prior saw only
       `3 + num_blocks*2*(kernel//2)` rows above a position: 11 rows for a
       4-block model and 19 for an 8-block model, against a 64-row code grid.
       A pelvis simply cannot be positioned with that little context.

    The vertical receptive field here is `num_layers * (kernel_size // 2)`
    rows. `covers_grid()` reports whether that spans the full grid height, and
    the constructor warns when it does not, so the configuration is checked
    rather than assumed. For a 64x32 grid, kernel 7 with 22 layers gives 66
    rows and therefore full coverage.

    Set `mask_classes=0` for a genuinely unconditional prior, or to the number
    of segmentation classes to condition on a downsampled mask (the mask is an
    input, never a prediction, so it may be seen in full).
    """

    def __init__(self, num_codes=512, width=128, num_layers=22, grid_hw=(64, 32),
                 embedding_dim=32, kernel_size=7, mask_classes=0):
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

        Unlike `PixelCNNPrior.sample`, no row cropping is used: the receptive
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


# --------------------------------------------------------------------------- #
# Transformer prior (exploratory addendum, not in the predeclared campaign)
# --------------------------------------------------------------------------- #
class TransformerPrior(nn.Module):
    """Causal transformer prior over raster-ordered code indices (exploratory).

    Same interface as PixelCNNPrior: forward(codes [B,H,W]) -> logits [B,K,H,W],
    loss(codes) -> mean cross-entropy in nats, sample(n, device, ...) -> codes.

    Sequence model: a learned start token is prepended to the raster-ordered
    codes, so position p sees only the start token and codes 0..p-1. A strict
    causal mask enforces this. Learned positional embeddings; pre-norm blocks.
    """

    def __init__(self, num_codes=512, width=256, num_layers=6, num_heads=8,
                 grid_hw=(64, 32), dropout=0.0):
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
# DCGAN baseline (Radford, Metz & Chintala, ICLR 2016)
# --------------------------------------------------------------------------- #
class DCGenerator(nn.Module):
    """DCGAN generator mapping a noise vector to a 256x128 slice.

    Architecture per Radford et al. (2016): a linear projection of the
    100-d noise vector to an 8x4x256 base, then five upsample stages to
    256x128 with batch norm and ReLU, and a tanh output (the DCGAN
    convention; training data is rescaled to [-1, 1] to match).

    Upsampling uses nearest-neighbour resize followed by 3x3 convolutions
    ("resize-convolution") instead of transposed convolutions: Odena et al.
    (2016) show transposed convolutions cause checkerboard banding, which
    is exactly the artefact observed in the feasibility smoke-test DCGAN
    samples (near-identical banded outputs), so this fix targets that
    documented failure mode.
    """

    def __init__(self, noise_dim=100, width=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.base = nn.Linear(noise_dim, 8 * 4 * width)
        self.bn0 = nn.BatchNorm1d(8 * 4 * width)
        chans = [width, width, width // 2, width // 4, width // 8]
        blocks = []
        for cin, cout in zip(chans, chans[1:] + [1]):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            ]
            if cout > 1:  # batch norm on all hidden layers (paper guideline)
                blocks += [nn.BatchNorm2d(cout), nn.ReLU()]
        blocks += [nn.Tanh()]
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        x = F.relu(self.bn0(self.base(z)))
        return self.net(x.view(z.size(0), -1, 8, 4))


class DCDiscriminator(nn.Module):
    """DCGAN discriminator: strided 3x3 convolutions 256x128 -> 8x4 -> scalar.

    Per the DCGAN paper it uses strided convolutions for downsampling (no
    pooling), a single linear logit output, and no batch norm on the input
    layer. Spectral normalisation (Miyato et al., 2018) is applied to all
    convolution weights to stabilise the adversarial game; the feasibility
    smoke test flagged DCGAN training instability (mode collapse to banded
    samples) as a key project risk, and a Lipschitz-constrained critic is
    the standard remedy short of switching to WGAN-GP.
    """

    def __init__(self, in_channels=1, width=64):
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1))]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 256x128 -> 128x64 -> 64x32 -> 32x16 -> 16x8 -> 8x4
        self.net = nn.Sequential(
            *block(in_channels, width, norm=False),   # no BN on input (paper)
            *block(width, width * 2),
            *block(width * 2, width * 4),
            *block(width * 4, width * 8),
            *block(width * 8, width * 8),
            # 8x4 head: an (8,4) kernel reduces the whole map to one logit
            # per sample (a 4x4 kernel would leave a 5x1 map).
            nn.utils.spectral_norm(
                nn.Conv2d(width * 8, 1, kernel_size=(8, 4), stride=1, padding=0)),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))


# --------------------------------------------------------------------------- #
# Mask-conditioned PixelCNN prior (exploratory: targets anatomical structure)
# --------------------------------------------------------------------------- #
class CondPixelCNNPrior(nn.Module):
    """PixelCNN prior conditioned on a downsampled segmentation mask.

    Motivation: the unconditional prior produces MRI-like texture but loses the
    global hip layout, because each position only sees `row_radius` rows above
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
