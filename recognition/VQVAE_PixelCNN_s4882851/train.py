"""Training, validation and checkpointing for the VQ-VAE (Project 2.4).

Trains the VQ-VAE on the HipMRI train split, validates on the held-out
validation patients, and saves the checkpoint with the best validation
SSIM (model selection on validation only; the test split is untouched
here and evaluated once at reporting time).

Choices and their justification:
  * Optimiser: Adam at lr=2e-4 as prescribed by van den Oord et al. (2017).
  * Reconstruction loss: MSE on [0,1]-normalised slices (matches the
    Gaussian likelihood assumed by the paper).
  * Model selection: mean validation SSIM (structural similarity is the
    acceptance metric, so checkpointing on it directly targets the
    criterion; MSE alone rewards over-smoothed reconstructions).
  * Codebook utilisation (perplexity) is logged every epoch because the
    feasibility smoke test observed early codebook collapse (13/512 codes);
    the first official experiment (E1) tests whether longer training
    recovers utilisation, as reported by prior work on this dataset.
  * Plots of losses, validation SSIM and perplexity are written every
    epoch (course spec requires plotting losses and metrics during
    training) so runs can be monitored from Slurm output directories.

Usage (on a Rangpur GPU node or any CUDA machine):
    python train.py --model vqvae --epochs 100 --out checkpoints/vqvae
    python train.py --model dcgan --epochs 50 --out checkpoints/dcgan
    python train.py --model prior \
        --vqvae-checkpoint checkpoints/vqvae/vqvae_best.pt --out checkpoints/prior
    python train.py --limit 256 --epochs 2        # quick pipeline check

The 'prior' mode freezes the trained VQ-VAE, encodes every slice to its
discrete code grid, and fits the PixelCNN to those grids (the second stage
of the VQ-VAE generative pipeline). Checkpointing is on validation
negative log-likelihood per code position.
"""
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # headless (HPC) plotting
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from dataset import HipMRISlices, verify_patient_splits
from modules import VQVAE, PixelCNNPrior, DCGenerator, DCDiscriminator


def parse_args():
    p = argparse.ArgumentParser(description="Train the VQ-VAE on HipMRI 2D slices")
    p.add_argument("--data-root", default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-codes", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="debug: cap slices per split (0 = use all)")
    p.add_argument("--out", default="checkpoints/vqvae", help="output directory")
    p.add_argument("--model", choices=("vqvae", "prior", "dcgan"), default="vqvae",
                   help="'vqvae': train the autoencoder; 'prior': train the "
                        "PixelCNN on codes from a frozen VQ-VAE checkpoint")
    p.add_argument("--vqvae-checkpoint",
                   default="checkpoints/vqvae/vqvae_best.pt",
                   help="frozen VQ-VAE weights used by --model prior")
    p.add_argument("--prior-width", type=int, default=128)
    p.add_argument("--prior-blocks", type=int, default=4)
    p.add_argument("--noise-dim", type=int, default=100,
                   help="generator noise dimension for --model dcgan")
    return p.parse_args()


@torch.no_grad()
def validation_ssim(model, val, batch_size):
    """Mean SSIM between held-out slices and their reconstructions.

    SSIM is computed per slice with scikit-image (data-range 1.0 because
    slices are normalised to [0,1]) and averaged over the split.
    """
    model.eval()
    total, n = 0.0, 0
    for i in tqdm(range(0, len(val), batch_size), desc="validating",
                   leave=False, unit="batch"):
        x = val._cache[i:i + batch_size] if val._cache is not None else \
            torch.stack([val[j] for j in range(i, min(i + batch_size, len(val)))])
        x_hat = model(x)[0]
        for a, b in zip(x, x_hat):
            total += ssim(a[0].numpy(), b[0].numpy(), data_range=1.0)
            n += 1
    return total / n


def save_plots(history, out_dir):
    """Loss/metric curves for monitoring training from the Slurm output dir."""
    epochs = range(1, len(history["train_recon"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_recon"], label="recon MSE")
    axes[0].plot(epochs, history["train_vq"], label="VQ + commitment")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses"); axes[0].legend()

    axes[1].plot(epochs, history["val_ssim"], color="tab:green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mean SSIM")
    axes[1].set_title("Validation SSIM (checkpoint criterion)")

    axes[2].plot(epochs, history["perplexity"], color="tab:red")
    axes[2].axhline(512, ls="--", c="grey", lw=0.8)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("codebook perplexity")
    axes[2].set_title("Codebook utilisation (512 = uniform)")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


@torch.no_grad()
def encode_split(vqvae, data, batch_size, device):
    """Encode every slice of a split to its integer code grid [N, H, W].

    The VQ-VAE is frozen, so this is a one-off precomputation that lets the
    prior train on compact code grids instead of re-running the encoder.
    """
    codes = []
    for i in tqdm(range(0, len(data), batch_size),
                  desc="encoding", leave=False, unit="batch"):
        if data._cache is not None:
            x = data._cache[i:i + batch_size]
        else:
            x = torch.stack([data[j] for j in range(i, min(i + batch_size, len(data)))])
        codes.append(vqvae.encode_to_codes(x.to(device)).cpu())
    return torch.cat(codes)


def train_prior(args):
    """Train the PixelCNN prior on code grids from a frozen VQ-VAE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    patients = verify_patient_splits(args.data_root)
    print("patient-disjoint split check passed:", patients, flush=True)

    vqvae = VQVAE(width=args.width, latent_dim=args.latent_dim,
                  num_codes=args.num_codes, beta=args.beta).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
    vqvae.eval()  # frozen: provides codes only, receives no gradients

    train = HipMRISlices(args.data_root, "train", preload=True)
    val = HipMRISlices(args.data_root, "validate", preload=True)
    if args.limit:
        train._cache, train.paths = train._cache[:args.limit], train.paths[:args.limit]
        val._cache, val.paths = val._cache[:max(args.limit // 4, 1)], val.paths[:max(args.limit // 4, 1)]
    print("encoding splits to code grids...", flush=True)
    train_codes = encode_split(vqvae, train, args.batch_size, device)
    val_codes = encode_split(vqvae, val, args.batch_size, device)
    print(f"train codes {tuple(train_codes.shape)}, val codes {tuple(val_codes.shape)}",
          flush=True)

    prior = PixelCNNPrior(num_codes=args.num_codes, width=args.prior_width,
                           num_blocks=args.prior_blocks,
                           grid_hw=tuple(train_codes.shape[1:])).to(device)
    n_params = sum(p.numel() for p in prior.parameters())
    opt = torch.optim.Adam(prior.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(train_codes, batch_size=args.batch_size,
                                         shuffle=True)

    history = {"train_nll": [], "val_nll": [], "epoch_seconds": []}
    best_nll = float("inf")
    peak_vram = 0.0

    for epoch in range(1, args.epochs + 1):
        prior.train()
        t0 = time.time()
        nll_sum = batches = 0
        for codes in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}",
                          leave=False, unit="batch"):
            codes = codes.to(device)
            nll = prior.loss(codes)
            opt.zero_grad(set_to_none=True)
            nll.backward()
            opt.step()
            nll_sum += nll.item()
            batches += 1
        epoch_s = time.time() - t0
        if device.type == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)

        # Validation NLL in bits per code for interpretability (log2).
        prior.eval()
        with torch.no_grad():
            val_nll = sum(prior.loss(c.to(device)).item()
                          for c in tqdm(torch.split(val_codes, args.batch_size),
                                        desc="validating", leave=False))
            val_nll /= len(torch.split(val_codes, args.batch_size))
        history["train_nll"].append(nll_sum / batches)
        history["val_nll"].append(val_nll)
        history["epoch_seconds"].append(epoch_s)
        print(f"epoch {epoch:3d} | train NLL {history['train_nll'][-1]:.4f} "
              f"| val NLL {val_nll:.4f} ({val_nll / torch.log(torch.tensor(2.0)):.4f} "
              f"bits/code) | {epoch_s:.1f}s", flush=True)

        if val_nll < best_nll:
            best_nll = val_nll
            torch.save(prior.state_dict(), os.path.join(args.out, "prior_best.pt"))

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ep = range(1, epoch + 1)
        ax[0].plot(ep, history["train_nll"]); ax[0].set_title("Train NLL (nats/code)")
        ax[1].plot(ep, history["val_nll"], color="tab:orange")
        ax[1].set_title("Validation NLL (nats/code)")
        for a in ax:
            a.set_xlabel("epoch"); a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "prior_training_curves.png"), dpi=150)
        plt.close(fig)

    summary = {
        "model": "pixelcnn_prior", "params": n_params, "epochs": args.epochs,
        "prior_width": args.prior_width, "prior_blocks": args.prior_blocks,
        "best_val_nll_nats": best_nll, "seed": args.seed,
        "epoch_seconds_avg": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        "peak_vram_gb": peak_vram,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(history.keys())
        w.writerows(zip(*history.values()))
    print("best val NLL (nats/code):", best_nll, flush=True)


def train_dcgan(args):
    """Train the DCGAN baseline (Radford et al., 2016) on the train split.

    Choices and their justification:
      * Data is rescaled to [-1, 1] to match the generator's tanh output
        (the DCGAN convention).
      * Adam with beta1=0.5 as prescribed by the DCGAN paper; the default
        0.9 momentum is known to cause oscillation in GAN training.
      * One-sided label smoothing (real targets 0.9) reduces the gradient
        the discriminator can pass on easily-real examples, a documented
        remedy for overconfident discriminators destabilising training.
      * There is no reliable held-out metric for GAN model selection
        (the validation split cannot be scored against ground truth), so
        checkpoints are saved every epoch and fixed-noise sample grids are
        written throughout for qualitative monitoring; final selection is
        made at evaluation time (E6) via KID against held-out real images.
      * Sample pixel std is logged as a cheap diversity proxy: the
        feasibility smoke test showed near-identical samples (pixel std
        0.067 vs 0.130 real), so this number diagnoses mode collapse
        without waiting for the full evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    patients = verify_patient_splits(args.data_root)
    print("patient-disjoint split check passed:", patients, flush=True)

    train = HipMRISlices(args.data_root, "train", preload=True)
    if args.limit:
        train._cache, train.paths = train._cache[:args.limit], train.paths[:args.limit]
    print(f"train {len(train)} slices", flush=True)

    netG = DCGenerator(noise_dim=args.noise_dim).to(device)
    netD = DCDiscriminator().to(device)
    bce = nn.BCEWithLogitsLoss()
    # beta1=0.5 per the DCGAN paper (default 0.9 oscillates).
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.noise_dim, device=device)  # fixed monitoring grid
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                         shuffle=True)
    history = {"d_real": [], "d_fake": [], "g_loss": [], "sample_std": [],
               "epoch_seconds": []}
    peak_vram = 0.0

    for epoch in range(1, args.epochs + 1):
        netG.train(); netD.train()
        t0 = time.time()
        d_real_sum = d_fake_sum = g_sum = batches = 0
        for x in loader:
            x = x.to(device) * 2.0 - 1.0  # [0,1] -> [-1,1] (tanh convention)
            b = x.size(0)

            # --- discriminator step on real then fake ---
            netD.zero_grad(set_to_none=True)
            d_real = bce(netD(x), torch.full((b,), 0.9, device=device))  # smoothed real
            fake = netG(torch.randn(b, args.noise_dim, device=device))
            d_fake = bce(netD(fake.detach()), torch.zeros(b, device=device))
            (d_real + d_fake).backward()
            optD.step()

            # --- generator step (wants D to call fakes real) ---
            netG.zero_grad(set_to_none=True)
            g_loss = bce(netD(fake), torch.ones(b, device=device))
            g_loss.backward()
            optG.step()

            d_real_sum += d_real.item(); d_fake_sum += d_fake.item()
            g_sum += g_loss.item(); batches += 1
        epoch_s = time.time() - t0
        if device.type == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)

        # Uncurated fixed-noise monitoring grid + diversity proxy.
        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()
        sample_std = samples.std().item()
        history["d_real"].append(d_real_sum / batches)
        history["d_fake"].append(d_fake_sum / batches)
        history["g_loss"].append(g_sum / batches)
        history["sample_std"].append(sample_std)
        history["epoch_seconds"].append(epoch_s)
        print(f"epoch {epoch:3d} | D(real) {history['d_real'][-1]:.3f} "
              f"| D(fake) {history['d_fake'][-1]:.3f} | G {g_sum / batches:.3f} "
              f"| sample std {sample_std:.3f} | {epoch_s:.1f}s", flush=True)

        torch.save(netG.state_dict(), os.path.join(args.out, "dcgan_G.pt"))
        torch.save(netD.state_dict(), os.path.join(args.out, "dcgan_D.pt"))
        grid = samples[:64] * 0.5 + 0.5  # back to [0,1] for display
        fig, axs = plt.subplots(4, 8, figsize=(16, 8))
        for ax, img in zip(axs.flat, grid):
            ax.imshow(img[0].numpy(), cmap="gray"); ax.axis("off")
        fig.suptitle(f"DCGAN fixed-noise samples, epoch {epoch}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "dcgan_samples.png"), dpi=120)
        plt.close(fig)

        ep = range(1, epoch + 1)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(ep, history["d_real"], label="D loss (real)")
        ax[0].plot(ep, history["d_fake"], label="D loss (fake)")
        ax[0].plot(ep, history["g_loss"], label="G loss")
        ax[0].set_title("Adversarial losses"); ax[0].legend()
        ax[1].plot(ep, history["sample_std"], color="tab:red")
        ax[1].set_title("Fixed-noise sample pixel std (diversity proxy)")
        ax[2].plot(ep, history["epoch_seconds"], color="tab:gray")
        ax[2].set_title("Seconds per epoch")
        for a in ax:
            a.set_xlabel("epoch"); a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "dcgan_training_curves.png"), dpi=150)
        plt.close(fig)

    summary = {
        "model": "dcgan",
        "params_G": sum(q.numel() for q in netG.parameters()),
        "params_D": sum(q.numel() for q in netD.parameters()),
        "epochs": args.epochs, "lr": args.lr, "noise_dim": args.noise_dim,
        "seed": args.seed, "final_sample_std": history["sample_std"][-1],
        "epoch_seconds_avg": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        "peak_vram_gb": peak_vram,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(history.keys())
        w.writerows(zip(*history.values()))
    print("DCGAN training complete; see dcgan_samples.png for the final grid",
          flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)  # reproducible weight init / shuffling

    if args.model == "prior":
        train_prior(args)
        return
    if args.model == "dcgan":
        train_dcgan(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Guard against the most damaging mistake in medical ML: patient leakage.
    patients = verify_patient_splits(args.data_root)
    print("patient-disjoint split check passed:", patients, flush=True)

    train = HipMRISlices(args.data_root, "train", preload=True)
    val = HipMRISlices(args.data_root, "validate", preload=True)
    if args.limit:  # debug mode: keep pipelines quick
        train._cache, train.paths = train._cache[:args.limit], train.paths[:args.limit]
        val._cache, val.paths = val._cache[:max(args.limit // 4, 1)], val.paths[:max(args.limit // 4, 1)]
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
    print(f"train {len(train)} slices, val {len(val)} slices", flush=True)

    model = VQVAE(width=args.width, latent_dim=args.latent_dim,
                  num_codes=args.num_codes, beta=args.beta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_recon": [], "train_vq": [], "val_ssim": [],
               "perplexity": [], "epoch_seconds": []}
    best_ssim = -1.0
    peak_vram = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        recon_sum = vq_sum = 0.0
        perp_sum = batches = 0
        for x in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}",
                      leave=False, unit="batch"):
            x = x.to(device, non_blocking=True)
            x_hat, vq_loss, perplexity, _ = model(x)
            recon = F.mse_loss(x_hat, x)
            loss = recon + vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            recon_sum += recon.item() * len(x)
            vq_sum += vq_loss.item() * len(x)
            perp_sum += perplexity.item()
            batches += 1
        epoch_s = time.time() - t0
        if device.type == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)

        val_ssim = validation_ssim(model, val, args.batch_size)
        history["train_recon"].append(recon_sum / len(train))
        history["train_vq"].append(vq_sum / len(train))
        history["val_ssim"].append(val_ssim)
        history["perplexity"].append(perp_sum / batches)
        history["epoch_seconds"].append(epoch_s)

        print(f"epoch {epoch:3d} | recon {history['train_recon'][-1]:.4f} "
              f"| vq {history['train_vq'][-1]:.4f} | val SSIM {val_ssim:.4f} "
              f"| perplexity {history['perplexity'][-1]:.1f} | {epoch_s:.1f}s", flush=True)

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(args.out, "vqvae_best.pt"))
        save_plots(history, args.out)

    # Machine-readable summary for the experiment log / report tables.
    summary = {
        "params": n_params, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "seed": args.seed,
        "best_val_ssim": best_ssim,
        "final_perplexity": history["perplexity"][-1],
        "epoch_seconds_avg": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        "peak_vram_gb": peak_vram,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(history.keys())
        w.writerows(zip(*history.values()))
    print("best val SSIM:", best_ssim, flush=True)


if __name__ == "__main__":
    main()