# Predeclared experiment campaign — 26 October 2026

## Purpose and scope

Compare unconditional hard VQ-VAE + masked PixelCNN synthesis against a
stabilized DCGAN on patient-disjoint HipMRI slices. Reconstruction fidelity,
generation fidelity, diversity, memorisation screens and computational cost
are separate outcomes. No outcome here establishes clinical usefulness,
anonymisation or differential privacy.

The campaign is deliberately submitted in full before inspecting its test
results. Backups are run even if the baseline succeeds, to avoid needing an
additional manual submission between Monday and Wednesday.

## Fixed settings and controlled changes

VQ defaults: width 128, D=64, K=512, beta=.25, Adam 2e-4, batch 32,
100 epochs, MSE reconstruction, 4× downsampling, uniform codebook init.
Best weights are selected by **validation SSIM**, not test scores.

| VQ run | Seed | Single change relative to the relevant comparator |
|---|---:|---|
| `vq_main_s0` | 0 | Paper-based main configuration |
| `vq_main_s1` | 1 | Training-seed repeat of main |
| `vq_init_s0` | 0 | Initialize codes from training encoder vectors |
| `vq_k128_s0` | 0 | K=128, compared with `vq_init_s0` |
| `vq_l1_s0` | 0 | L1 reconstruction, compared with `vq_init_s0` |
| `vq_grid8_s0` | 0 | 8× downsampling (512 tokens), compared with `vq_init_s0` |

The initialization is sampled encoder vectors, **not** k-means or EMA
codebook updates. The 8× architecture adds a third stride-2 encoder/decoder
stage. Masks/labels are not used for training.

Each VQ run has its own frozen-code PixelCNN: width 128, learned embedding
32, four two-convolution residual blocks, 7×7 initial type-A mask and 3×3
type-B residual masks; Adam 2e-4, batch 16, 40 epochs. Prior selection uses
sample-weighted validation NLL in nats/code.

`prior_wide7` is a coupled capacity/receptive-field alternative on the
unchanged `vq_main_s0`: width 64, two residual blocks and 7×7 residual
kernels. Report it as a **coupled** ablation, not an isolated kernel effect.

DCGAN: seeds 0 and 1, 50 epochs, batch 32, Adam 2e-4, betas (.5,.999),
100-dimensional Gaussian noise, nearest-resize convolution generator,
spectral-normalised discriminator, real-label smoothing .9.
Use the **predeclared final epoch**, with no test-KID checkpoint selection.

## Evaluation, randomness and selection

- Seven predeclared test evaluations: all six VQ/prior variants plus wide7.
- Each evaluation uses **500 generated images per model per sampling seed**
  (seeds 0,1,2: 1500 images/model). The matching DCGAN training seed is used.
- Repeated sampling seeds are **not** independent training runs. Only the
  main architecture has a second training-seed repeat; do not claim all
  ablations have multi-training-seed uncertainty.
- KID uses repeated grayscale channels, torch-fidelity Inception features,
  50 subsets of 100 images, fixed subset RNG 2026. Subset std is not a
  patient-level confidence interval. ImageNet features have a domain gap.
- Report reconstruction SSIM on all 540 test slices, per-patient means,
  fraction >=.6, test code usage/perplexity, and fixed uncurated grids.
- Generated images have no paired ground truth. Nearest-held-out-by-pixel-L2
  SSIM is an explicitly **biased similarity screen**, not reconstruction
  accuracy or clinical fidelity.
- Nearest-training audits use **all 11,460 training slices**, with chunked
  exact pixel-L2 search, compared against held-out-real-to-train distances.
  The 5th-percentile RMSE flag threshold is calibrated on validation.
  Inspect suspicious pairs manually; absence of flags proves no privacy.
- Save across-sample pixel std, distinct quantized images, interpolation,
  and four distance-selected failure candidates per model.
- Main test temperature stays at 1.0. A separate **validation-only**
  sensitivity task samples temperatures .7,1.0,1.3 with 500 images each,
  seed 0. Do not select a test temperature retrospectively.
- Engineering selection among seed-0 VQ runs uses highest validation SSIM.
  Report all test ablations, including failures, without cherry-picking.
  A lower-KID test variant is not automatically the selected architecture.

## Safety, resources and scheduling

The 25-task DAG consists of one CUDA gate, six VQ trainings, two GAN
trainings, seven prior trainings, seven test evaluations, one temperature
task, and one CPU collector. Training/evaluation jobs require successful
parents (`afterok`); the collector waits for terminal parents (`afterany`).
Invalid dependencies are cancelled rather than waiting silently forever.

The GPU gate exercises actual model widths at batch 32 on 64 real slices,
including finite optimizer steps, all model types, alternative codebook/
prior settings, 8×/L1, sampling, KID and audit/interpolation. Full jobs cannot
start until it passes. It is not evidence of final quality or final runtime.

Model jobs request one GPU, four CPUs, permitted normal QoS, and a 90-minute
walltime; the gate has 30 minutes and collector 15 minutes. No physical
memory request is used because this Slurm installation advertises synthetic
memory resources. One GPU does not necessarily imply exclusive physical GPU
use; preserve Slurm allocation records.

The original **three-hour training budget was provisional**. Running backups,
repeats and full audits expands total campaign cost. Record actual summed
GPU-job elapsed hours as well as per-run times; neither test gates nor old
feasibility timings prove compliance with that total budget. The measured
500-image generation target is <=900s, with 3600s as the review's upper target.
Queue waiting is separate from compute time; Wednesday completion cannot be
guaranteed by submission.

Atomic best-weight and epoch-boundary resume checkpoints retain optimizer
and RNG state. Re-running interrupted training requires its exact config
and `--resume`; no batch-size/architecture changes are silently made.
CPU uninterrupted/resumed trajectories were verified bit-for-bit.
CUDA reproducibility is seeded but not promised bit-for-bit across devices.

Every job verifies source SHA256 against the submission manifest. Config,
checkpoint provenance, CSV/history plots, timestamps, 30-second heartbeats
and batch/position progress remain in ignored campaign/log directories.
A heartbeat means the launcher is alive, **not** proof that a model advances.

## Wednesday artifacts and human work

`campaigns/<campaign>/report_artifacts.tar.gz` contains JSON/CSV/Markdown,
figures and logs, but no model files or Nifti data. It is created automatically
by the final collector. Check `campaign_results.json` has `complete: true`,
no missing evaluations and no failed/unverified tasks before reporting a
completed experiment. Preserve checksums, per-patient scores, and raw records.

The archive supports results tables, training curves, uncurated grids,
reconstruction comparison, runtime/memory discussion, nearest-neighbour
analysis, interpolation and initial failure candidates.

Human review is still required: annotate 3–5 genuine failure autopsies with
anatomical/reliability interpretation, evaluate the ethics trade-off, write
the manager-facing recommendation, verify references and required AI
disclosure, and satisfy report/submission requirements. Automated
distance-selected candidates are not clinical diagnoses.