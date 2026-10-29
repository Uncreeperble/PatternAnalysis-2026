"""DCGAN baseline (Radford, Metz & Chintala, ICLR 2016) — reproducibility.

The DCGAN is the simpler generative baseline the report compares against the
VQ-VAE pipeline. It is retained here so the baseline results can be
reproduced; the submitted solution is the VQ-VAE-2 + Transformer prior in
``modules.py``.
"""
import torch.nn as nn
import torch.nn.functional as F


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
        """Build the generator.

        Args:
            noise_dim: dimension of the input Gaussian noise vector.
            width: base channel width (halved at each upsample stage).
        """
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
        """Map noise [B, noise_dim] to images [B, 1, 256, 128] in [-1, 1]."""
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
        """Build the discriminator.

        Args:
            in_channels: input image channels (1 for grayscale MRI).
            width: base channel width (doubled at each downsample stage).
        """
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            """One spectral-normalised conv (+ optional BN) + LeakyReLU block."""
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
        """Return one real/fake logit per input image."""
        return self.net(x).view(x.size(0))
