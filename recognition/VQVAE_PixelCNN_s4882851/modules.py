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
