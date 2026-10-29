"""Reproducibility baselines and shared primitives for Project 2.4.

The **submitted solution** — the hierarchical VQ-VAE-2 and its causal
Transformer prior — lives in ``modules.py`` at the project root, because the
course specification requires the main model to be in ``modules.py``.

This package holds everything else, kept so the report's comparison results
can be reproduced:

    primitives.py   ResidualBlock, VectorQuantizer (shared by both sides)
    vqvae.py        single-level VQ-VAE (van den Oord et al., 2017)
    pixelcnn.py     masked PixelCNN prior + mask-conditioned variant
    gated.py        gated PixelCNN prior (Experiment 8)
    dcgan.py        DCGAN generator/discriminator baseline

Nothing here imports ``modules.py``, so ``modules.py`` can safely import from
this package to re-export the baselines without a circular import.
"""
from .primitives import ResidualBlock, VectorQuantizer
from .vqvae import Decoder, Encoder, VQVAE
from .pixelcnn import CondPixelCNNPrior, MaskedConv2d, PixelCNNPrior
from .gated import GatedMaskedLayer, GatedPixelCNNPrior
from .dcgan import DCDiscriminator, DCGenerator

__all__ = [
    "ResidualBlock", "VectorQuantizer",
    "Encoder", "Decoder", "VQVAE",
    "MaskedConv2d", "PixelCNNPrior", "CondPixelCNNPrior",
    "GatedMaskedLayer", "GatedPixelCNNPrior",
    "DCGenerator", "DCDiscriminator",
]
