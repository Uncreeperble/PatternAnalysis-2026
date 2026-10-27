# VQ-VAE with PixelCNN Prior for Synthetic Prostate MRI (COMP3710 Project 2.4)

**Student:** Ben Evans (s4882851) &mdash; Hard difficulty &mdash; PyTorch

> **Status: work in progress.** This is a draft of the report. Sections marked
> *[TODO]* are pending figures, failure-case review, or the student's own
> annotation, and will be completed before submission.

## 1. Problem and description

Medical imaging research is limited by patient privacy rules (HIPAA, GDPR) and
by how few clinical scans are available. Synthetic scans could help, but a
generator that memorises training scans leaks patient data, and one that
produces anatomically impossible structures would mislead downstream models.
This project builds a **VQ-VAE with a PixelCNN prior** (van den Oord et al.,
2017) that generates synthetic 2D prostate MRI slices from the HipMRI study,
and compares it with a **DCGAN** baseline (Radford et al., 2016). The aim is to
investigate the trade-off between fidelity, memorisation and privacy.

## 2. How the algorithm works

A VQ-VAE encodes each slice into a grid of discrete codes and decodes codes back
into an image. The encoder maps a 256×128 slice to a 64×32 grid of latent
vectors. Each vector is replaced by its nearest entry in a learned 512-entry
codebook, and the decoder reconstructs the slice from the quantised grid.
Gradients pass through the quantisation with a straight-through estimator.

A VQ-VAE alone cannot sample new images, because it does not model which code
grids are plausible. A **PixelCNN prior** is therefore trained on the code
indices. It uses masked convolutions so that each code depends only on codes
earlier in raster order. To generate, the prior samples a 64×32 code grid one
position at a time, and the decoder turns that grid into a slice.

The DCGAN baseline maps Gaussian noise directly to images through a generator,
trained adversarially against a discriminator.

> **[Diagram placeholder: VQ-VAE architecture and PixelCNN sampling flow.
> To be replaced with a proper diagram, produced with the student in a later
> session.]**

## 3. Feasibility review

*Summary of the one-page feasibility review written before the experiments
(source: `feasibility_review.tex`).*

**User need.** Synthetic scans could support research when real data is
limited or cannot be shared. Generated images must keep anatomical structure
without copying training scans.

**Acceptance criteria (as planned).**
- Reconstruction: mean SSIM ≥ 0.6 between held-out slices and their
  reconstructions.
- Generation: 500 samples per unconditional model, compared with held-out real
  slices by Kernel Inception Distance (KID).
- Diversity: codebook use and distinct entries; uncurated grids.
- Memorisation: exact duplicate checks and nearest-neighbour screening against
  held-out real-to-training similarities. This is a screen, not a privacy
  guarantee.
- Compute: 3 hours of training, and 500 samples within 15 minutes.

**Preliminary evidence.** A 5-epoch smoke test gave held-out SSIM 0.624 but
used only 13 of 512 codes (perplexity 6.8). A PixelCNN on those codes cut
cross-entropy from 6.24 to 0.63. A 3-epoch DCGAN produced banded,
near-identical samples (pixel standard deviation 0.067 against 0.130 for real
slices). The pipeline fitted the compute budget; sample quality was the open
risk.

**Planned mitigations.** Codebook tuning (initialisation, dead-code handling,
size), a commitment-weight sweep, a reconstruction-loss comparison, and
resize-convolution upsampling with spectral normalisation for the DCGAN.

## 4. Data

### 4.1 Source and layout

The HipMRI prostate study provides 2D slices in `keras_slices_data`, split into
train, validation and test folders. The dataset is not redistributed in this
repository.

### 4.2 Audit findings

| Property | Finding | Consequence |
|---|---|---|
| Slices | 12,660 in total; 11,460 train (31 patients, 191 scans), 660 validation (4 patients), 540 test (3 patients) | Splits are about 90.5 / 5.2 / 4.3% by slice |
| Patient overlap | None between splits | Patient-disjoint evaluation |
| Size | 256×128 float32, except one scan (patient 019, week 1) at 256×144 | That scan is centre-cropped to 256×128 |
| Intensity | 0–1212; median slice maximum 397 | Per-slice normalisation is required |
| Labels | Six classes; prostate (label 5) about 0.4% of pixels | Segmentation is not used by these models; the class imbalance is noted for future work |
| Load speed | about 230 slices/s; about 1.7 GB in RAM | Full set kept in memory |

The audit figures are in `data_exploration/` in the project folder and will be
added to this section in a later commit.

### 4.3 Preprocessing

Each slice is divided by its own maximum to map it to [0, 1]. Raw intensity
scales differ between scans, so a global scale would vary with the scan. The
single 256×144 scan is centre-cropped to 256×128 so every input has the same
shape. The grayscale value is kept as a single channel.

### 4.4 Splits

The splits come from the dataset provider and are kept patient-disjoint. Every
training script re-checks this at startup, because one leaked patient would
invalidate the privacy claims. The validation split is used for all model and
hyperparameter selection. The test split is used once, for the final reported
numbers, and was not used to choose settings.

## 5. Experiments

Experiments were run in order. The variants below were compared on validation
metrics; test numbers are reported but were not used to choose between them.

### 5.1 Baseline VQ-VAE and the codebook collapse

The first model followed van den Oord et al. (2017): two 4×4 stride-2
convolutions, two residual blocks, a 512-entry codebook of dimension 64,
β = 0.25, Adam at 2×10⁻⁴, and width 128. In the 5-epoch smoke test it reached
validation SSIM 0.624, but only 13 codes were used (perplexity 6.8). This
codebook collapse is a known failure mode, and the campaign was designed to test
whether longer training recovers code use.

### 5.2 Longer training and reconstruction loss

Training for 100 epochs raised code use to 199–338 of 512 on the test set,
depending on seed and configuration. The L1 reconstruction variant was selected
on validation SSIM (0.93164), against 0.93111 for the main configuration with
seed 0. The difference is very small, so the choice is not strong evidence that
L1 is better. It was made by the predeclared validation rule, not by test
performance.

### 5.3 Codebook and downsampling variants

| Variant | Validation SSIM | Codes used (test) | Decision |
|---|---:|---:|---|
| Baseline (L1, 4× downsampling, 512 codes) | 0.9316 | 199 | **Selected** |
| Data-based codebook initialisation | 0.9287 | 236 | Not selected |
| 128 codes | 0.9184 | 128 | Not selected; lower perplexity (66) |
| 8× downsampling (32×16 grid) | 0.7850 | 187 | Not selected; large loss of detail |

Eight-fold downsampling makes sampling about four times cheaper, but the
reconstruction quality drops sharply, so it was rejected. Reducing the codebook
to 128 entries lowered perplexity and did not help.

### 5.4 Prior variants

The PixelCNN prior was trained on the frozen selected VQ-VAE. The selected prior
reached a validation NLL of 3.455 nats per code. Other prior variants ranged from
2.924 to 3.602 nats; these are reported in `RESULTS.md`. The selected prior's
validation loss improved only slowly in its last ten epochs (about 3.468 at
epoch 30 and 3.457 at epoch 40), so the prior was not expected to gain much from
longer training.

### 5.5 DCGAN baseline

The first DCGAN produced banded samples, a known artefact of transposed
convolutions (Odena et al., 2016). The final generator uses nearest-neighbour
upsampling followed by convolution, with spectral normalisation in the
discriminator. The baseline was trained for 50 epochs with two seeds.

### 5.6 What did not work

- 8× downsampling: reconstructions lost too much detail (see 5.3).
- 128 codes: lower perplexity and no gain in quality.
- Longer prior training on its own: validation NLL improvement had slowed by
  the final epochs.
- The first DCGAN: banded samples. The upsampling change in 5.5 was made to
  address this, but the fix has not yet been checked on a visual comparison.

## 6. Results

All numbers come from the frozen campaign. Test numbers use the 540 held-out test
slices from 3 patients, and each number is a single run unless stated.

### 6.1 Quantitative results

**Reconstruction (test, SSIM).** The selected VQ-VAE reaches mean SSIM
**0.9425**, with 540/540 reconstructions at or above 0.6. Seeds 0 and 1 of the
main configuration give 0.9414 and 0.9432. The 0.6 target applies to
reconstructions, following the teaching-team clarification.

**Generation (unpaired SSIM, supplementary).** Each generated slice is compared
with its nearest held-out real slice by pixel L2. The same protocol on real
scans gives mean 0.2504, so this measure cannot reach 0.6 even for real data.

| Cohort | Mean SSIM to nearest reference | Fraction ≥ 0.6 |
|---|---:|---:|
| Real test scans (control) | 0.2504 | 0.000 |
| VQ-VAE + PixelCNN prior (selected) | 0.1827 | 0.000 |
| DCGAN (seed 0) | 0.2154 | 0.000 |

**Distribution fidelity (KID, 1,500 samples per model).**

| Run | VQ-VAE + prior | DCGAN |
|---|---:|---:|
| Seed 0 | 0.4218 (selected) / 0.3734 (main) | 0.4890 |
| Seed 1 | 0.4308 (main) | 0.2830 |

Lower is better. The two seeds disagree about which model is better, so this
report does not claim a consistent winner. The K128 variant scored 0.3478 on the
test set, but it was not chosen on test results.

**Resources (A100-PCIE-40GB).**

| Model | Epoch time (s) | Peak training VRAM (GB) | Time for 500 samples |
|---|---:|---:|---:|
| VQ-VAE (selected) | 8.21 | 1.03 | - |
| PixelCNN prior (selected) | 7.85 | 0.68 | about 40 s |
| DCGAN | 8.65 | 1.01 | about 0.08 s |

The campaign did not record parameter counts. Preliminary parameter counts from
the smoke tests were 1.18 M (VQ-VAE), 2.13 M (PixelCNN prior), and 3.2 M / 2.8 M
(DCGAN generator / discriminator). Total GPU allocation was 3.03 hours, of which
2.53 hours was training.

### 6.2 Qualitative results

![Test reconstructions](figures/reconstruction.png)

*Test-set reconstructions from the selected VQ-VAE, uncurated.*

![VQ-VAE + prior samples, seed 0](figures/vqvae_prior_seed0.png)

*Uncurated samples from the VQ-VAE + PixelCNN prior, fixed seed 0.*

![DCGAN samples, seed 0](figures/dcgan_seed0.png)

*Uncurated samples from the DCGAN baseline, fixed seed 0.*

![Nearest-neighbour distances](figures/nn_distance_distributions.png)

*Distances from generated and real slices to their nearest training slices.*

**[TODO] Side-by-side comparison.** Each model's samples next to real slices,
with written observations on structure, tissue boundaries and diversity.

**[TODO] Closest training pairs.** Generated samples next to their nearest
training slices, to show whether they are distinct.

### 6.3 Memorisation screen

No generated slice exactly matches a training slice after uint8 conversion, and
all sampled slices are unique. For the selected prior, 3.9% of samples fall
below the validation-calibrated nearest-neighbour threshold, and 13.2% do at
lower sampling temperature. This is evidence against verbatim copying in these
runs. It is not a privacy guarantee, and it cannot rule out near-copies.

### 6.4 Failure autopsies

**[TODO] Three to five failure cases**, selected from the failure-candidate
figures and reviewed by the student. Each should name the artefact observed,
the likely trigger, and whether it is a reconstruction or a sampling failure.

## 7. Limitations

- Unpaired generated-slice SSIM cannot reach 0.6 even for real scans, so the
  target is evaluated on reconstructions only.
- Patient numbers are small (3 test patients, 4 validation patients), so
  per-patient variation is large.
- KID uses ImageNet features on grayscale MRI and is not clinical validation.
- Nearest-neighbour screening does not prove the absence of memorisation.
- No formal differential-privacy bound was applied.
- Parameter counts were not recorded.

## 8. Recommendation for the project manager

For reconstruction-based downstream use, the VQ-VAE meets the fidelity target.
For unconditional synthetic cohorts, neither model is validated: the prior is
about 500 times slower per sample than the DCGAN and is not clearly better on
KID. Before any synthetic data is shared, we recommend expert review of samples,
treating the memorisation screen as a screen only, and evaluating FID and a
larger held-out cohort.

## 9. Dependencies and reproducibility

- Python 3.11, torch 2.13.0+cu130, torchvision 0.28.0+cu130,
  torchmetrics 1.9.0, torch-fidelity 0.4.0, nibabel 5.4.2,
  scikit-image 0.26.0, matplotlib 3.11.2, tqdm 4.70.0, NumPy.
- KID requires torch-fidelity and Inception weights, cached outside Git.
- Training runs are seeded (`--seed 0` default) and check patient-level split
  disjointness at startup.
- Operational instructions for running experiments are in `RUNNING.md`.

Local tests: `python -m unittest discover -p 'test_*.py' -v`.

## 10. Artificial Intelligence Usage Disclosure

**Tools used:** DeepSeek (agentic coding assistant) and Claude (analysis and
report drafting). The design decisions, experiment selection and final
judgements are the author's own.

**Tasks assisted:**
- DeepSeek: scaffolding and completing the model, training, evaluation and
  campaign code; generating the figures and plotting scripts; drafting the
  README and maintaining the project's status record.
- Claude: reviewing the experimental results and helping draft the analysis
  and conclusions in this report.

**Verification:** The author checked AI-generated code by running the
regression tests and by running the pipeline on the real HipMRI split. Reported
numbers were cross-checked against the raw `campaign_results.json` output
rather than taken from generated text. The real-to-real SSIM control was run
to test the metric's interpretation. [Author to confirm which code was read
line by line and which parts were tested only.]

## 11. References

- van den Oord, A., Vinyals, O. & Kavukcuoglu, K. (2017). Neural Discrete
  Representation Learning. NeurIPS.
- van den Oord, A., et al. (2016). Conditional Image Generation with PixelCNN
  Decoders. NeurIPS.
- Radford, A., Metz, L. & Chintala, S. (2016). Unsupervised Representation
  Learning with Deep Convolutional GANs. ICLR.
- Odena, A., Dumoulin, V. & Olah, C. (2016). Deconvolution and Checkerboard
  Artifacts. Distill.
- Miyato, T., et al. (2018). Spectral Normalization for GANs. ICLR.
- Binkowski, M., et al. (2018). Demystifying MMD GANs (KID). ICLR.
- Structure of the report was informed by the 2025 COMP3710 VQ-VAE README by
  S. Subramaniam (PatternAnalysis-2025, s4858817_VQVAE). No text was copied.
