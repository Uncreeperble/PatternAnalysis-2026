# Running the experiments

This file holds the operational instructions that were removed from the report
(README.md). It covers how to reproduce the training and evaluation pipeline.

## Local checks

```bash
python -m unittest discover -p 'test_*.py' -v
```

## Single runs

```bash
# VQ-VAE
python train.py --model vqvae --epochs 100 --out checkpoints/vqvae
# PixelCNN prior on the frozen VQ-VAE
python train.py --model prior --vqvae-checkpoint checkpoints/vqvae/vqvae_best.pt --out checkpoints/prior
# DCGAN baseline
python train.py --model dcgan --epochs 50 --out checkpoints/dcgan
# Evaluation (SSIM, KID, memorisation audit, figures)
python predict.py --vqvae checkpoints/vqvae/vqvae_best.pt \
    --prior checkpoints/prior/prior_best.pt \
    --dcgan checkpoints/dcgan/dcgan_G.pt --out outputs
```

## Full experiment campaign (Slurm)

`campaign.py` defines the complete predeclared set of 25 jobs (training,
priors, evaluations, temperature comparisons and collection) with success
dependencies.

```bash
python campaign.py plan --root <campaign-dir>
python campaign.py submit --manifest <campaign-dir>/manifest.json --test-only
python campaign.py submit --manifest <campaign-dir>/manifest.json
python campaign.py monitor --manifest <campaign-dir>/manifest.json --watch
```

- Do not submit a second copy of an existing campaign. Job IDs are saved after
  every submission.
- `--continue` fills only missing submissions after an interruption. It does not
  retry failed experiments.
- Batch scripts omit `--mem`, because this cluster's memory accounting does not
  reflect physical RAM. Log directories must exist before Slurm launches.
- Training writes per-epoch CSV and plots, best weights, `latest.pt` (with
  optimiser and RNG state), and status JSON. To resume, use exactly the same
  flags plus `--resume`.
- Existing run directories are refused to prevent accidental overwrites.

## Retrieving results

The campaign collector writes `report_artifacts.tar.gz` with figures, JSON, CSV
and logs. It excludes dataset files and model weights. Check
`campaign_results.json` for `complete: true`, no missing evaluations and no
failed or unverified tasks before reporting results.

## Flags

`--limit` and `--skip-kid` are for debugging only. Official evaluations always
compute KID.
