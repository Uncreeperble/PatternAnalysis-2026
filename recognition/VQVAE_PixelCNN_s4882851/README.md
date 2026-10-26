# VQ-VAE with PixelCNN Prior for Synthetic Prostate MRI (COMP3710 Project 2.4)

**Student:** Ben Evans (s4882851) &mdash; Hard difficulty &mdash; PyTorch

## Description

Medical imaging research is constrained by patient privacy laws (HIPAA, GDPR)
and limited scan availability, so synthetic cohorts promise anonymous,
shareable data. However a generative model that merely *memorises* training
scans would leak patient data, and one that *hallucinates* impossible anatomy
would poison downstream research. This project builds a **VQ-VAE with a
PixelCNN prior** (van den Oord et al., 2017) that generates synthetic 2D
prostate MRI slices from the HipMRI study, and benchmarks it against a
**DCGAN** baseline (Radford et al., 2016) to investigate this
fidelity&ndash;memorisation&ndash;privacy trilemma.

## How it works

The VQ-VAE encoder maps each 256x128 slice to a 64x32 grid of discrete
latents, each quantised to its nearest entry in a learned 512-entry codebook
(straight-through gradients carry decoder gradients past the quantisation).
The decoder reconstructs the slice from the quantised grid. Because the
autoencoder alone cannot *sample* new grids, a PixelCNN prior is trained on
the discrete code indices with masked convolutions (type-A then type-B,
enforcing the raster-scan autoregressive factorisation). New slices are
synthesised by sampling code grids from the prior and decoding them. The
DCGAN baseline instead maps Gaussian noise directly to pixels through a
resize-convolution generator adversarially trained against a
spectral-normalised discriminator.

![Pipeline overview](figures/overview.png)
*(figure to be added)*

## Repository layout

| File | Purpose |
|------|---------|
| `modules.py` | VQ-VAE, PixelCNN prior and DCGAN model components (pure PyTorch) |
| `dataset.py` | Nifti loading, preprocessing, leakage-free patient-level splits |
| `train.py` | Training/validation/checkpointing for all three models |
| `predict.py` | Evaluation driver: SSIM, KID, memorisation audit, figures |
| `slurm_train.sh` | Slurm batch script for the Rangpur HPC |

## Dependencies and reproducibility

- Python 3.11, PyTorch (CUDA), nibabel, scikit-image, torchmetrics, matplotlib
- Exact versions used for the reported results: *(to be added after final runs)*
- All training runs are seeded (`--seed 0` default); every training script
  re-verifies patient-level split disjointness at startup.

## Usage

```bash
# Train the VQ-VAE (E1)
python train.py --model vqvae --epochs 100 --out checkpoints/vqvae
# Train the PixelCNN prior on the frozen VQ-VAE (E4)
python train.py --model prior --vqvae-checkpoint checkpoints/vqvae/vqvae_best.pt --out checkpoints/prior
# Train the DCGAN baseline (E5)
python train.py --model dcgan --epochs 50 --out checkpoints/dcgan
# Evaluate everything (SSIM, KID, memorisation audit, figures)
python predict.py --vqvae checkpoints/vqvae/vqvae_best.pt \
    --prior checkpoints/prior/prior_best.pt \
    --dcgan checkpoints/dcgan/dcgan_G.pt --out outputs
```

On Rangpur, use `sbatch slurm_train.sh <model> [flags]` (see script header).

## Data and preprocessing

The HipMRI 2D slice data (`keras_slices_data`) contain 12,660 slices from 38
patients: 11,460 train (31 patients), 660 validation (4 patients), 540 test
(3 patients). The splits are provided at patient level and are re-verified
disjoint at runtime in every script (a single leaked patient would invalidate
every privacy claim). Preprocessing: each slice is scaled by its own maximum
to [0, 1] (raw intensities span 0-1212 and vary per scan), and the single
256x144 scan (patient 019, week 1) is centre-cropped to 256x128.

## Feasibility review

*(The one-page feasibility review, as signed off at the practical check-off.)*

### User need, scope and acceptance criteria

Synthetic medical images could support research when real patient data is
limited or cannot be shared due to privacy constraints. However, generated
images must preserve anatomical structure without simply reproducing training
examples. This project investigates these trade-offs using 2D HipMRI slices,
comparing a VQ-VAE with a PixelCNN prior against a DCGAN baseline.

Acceptance criteria:

- Reconstruction fidelity: mean SSIM >= 0.6 between held-out images and their
  VQ-VAE reconstructions.
- Generation: 500 samples from each unconditional model, compared by Kernel
  Inception Distance (KID) against held-out real images.
- Diversity: codebook utilisation and distinct entries used; sample variety
  assessed with uncurated grids and feature-space metrics.
- Memorisation risk: check for duplicates and compare generated-to-training
  similarity against held-out real-to-training similarities (a screening
  test, not a privacy guarantee).
- Computational feasibility: provisional budget of 3 hours training, 500
  images within 15 minutes.

### Model choice and course concepts

The VQ-VAE (van den Oord et al., 2017) encoder maps images to latents
quantised to a learned codebook; the decoder reconstructs from the discrete
grid. A PixelCNN autoregressive prior learns the distribution of code
indices so new grids can be sampled and decoded. A DCGAN (Radford et al.,
2016) provides an unconditional baseline. Course content covered practical
implementations of both VQ-VAE and GAN models.

### Preliminary feasibility evidence (smoke test)

The VQ-VAE followed the paper architecture (two 4x4 stride-2 convolutions
giving a 64x32 latent grid, two residual blocks, 512-entry codebook of
dimension 64, beta=0.25, Adam 2e-4, width 128). A 5-epoch A100 smoke test
reached held-out SSIM 0.624 (78% of slices >= 0.6) at 6.9 s/epoch and
3.39 GB peak, but used only 13/512 codes (perplexity 6.8). A 2.13M-parameter
PixelCNN on the frozen codes cut cross-entropy from 6.24 to 0.63; measured
sampling extrapolates to 2.8 minutes for 500 images. A 3-epoch DCGAN
generated 500 images in 0.12 s but produced banded, near-identical samples
(pixel std 0.067 vs 0.130 real). The pipeline fits the budget; sample quality
was the open risk.

### Risks, budget and fallback

Risks: blurry reconstructions, codebook under-utilisation, poor samples,
DCGAN instability, memorisation, limited compute. Mitigations: loss and
codebook tuning (EMA updates, dead-code resets, K/beta sweeps), resize-
convolution upsampling and spectral norm for the DCGAN, early stopping and
augmentation against memorisation, and reporting reconstruction-only results
if the prior cannot be completed.

## Results

*(Quantitative results to be filled in after the experiment runs: test SSIM,
perplexity, KID table, resource profiling.)*

## Investigation of the open research dilemma

*(To be completed after experiments: baseline comparison, failure autopsies
of 3-5 flawed generations, memorisation audit interpretation, and the
engineering recommendation to the project manager.)*

## Artificial Intelligence Usage Disclosure

*(To be completed: AI tools used, tasks assisted, and how the outputs were
audited and verified.)*

## References

- van den Oord, A., Vinyals, O. &amp; Kavukcuoglu, K. (2017). Neural Discrete
  Representation Learning. NeurIPS.
- van den Oord, A., et al. (2016). Conditional Image Generation with PixelCNN
  Decoders. NeurIPS.
- Radford, A., Metz, L. &amp; Chintala, S. (2016). Unsupervised Representation
  Learning with Deep Convolutional GANs. ICLR.
- Odena, A., Dumoulin, V. &amp; Olah, C. (2016). Deconvolution and Checkerboard
  Artifacts. Distill.
- Miyato, T., et al. (2018). Spectral Normalization for GANs. ICLR.
- Binkowski, M., et al. (2018). Demystifying MMD GANs (KID). ICLR.
