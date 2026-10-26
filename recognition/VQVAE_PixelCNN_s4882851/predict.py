"""Inference and evaluation driver (Project 2.4).

Loads the trained VQ-VAE + PixelCNN prior and the DCGAN baseline, then
produces every quantitative and qualitative result used in the report:

  1. Test-set reconstruction SSIM for the VQ-VAE (the acceptance metric).
  2. 500 unconditional samples from each generative model, timed against
     the 15-minute budget, with uncurated fixed-seed sample grids.
  3. Codebook utilisation on the test split (distinct codes, perplexity).
  4. KID (Kernel Inception Distance) of each sample set against held-out
     real test images (lower = closer feature distributions).
  5. Memorisation audit: nearest-training-neighbour distance of generated
     images, compared against the same statistic for held-out real
     images. Generated images much closer to training slices than real
     held-out slices would indicate memorisation; this is a screening
     test, not a privacy guarantee.
  6. results.json with all numbers for the report tables.

NumPy is used here for visualisation and evaluation only, which the
course spec permits in the prediction script; the models themselves
remain pure PyTorch.

Usage (GPU node):
    python predict.py --vqvae checkpoints/vqvae/vqvae_best.pt \
        --prior checkpoints/prior/prior_best.pt \
        --dcgan checkpoints/dcgan/dcgan_G.pt --out outputs
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.kid import KernelInceptionDistance

from dataset import HipMRISlices, verify_patient_splits
from modules import VQVAE, PixelCNNPrior, DCGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained generative models")
    p.add_argument("--data-root",
                   default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    p.add_argument("--vqvae", default="checkpoints/vqvae/vqvae_best.pt")
    p.add_argument("--prior", default="checkpoints/prior/prior_best.pt")
    p.add_argument("--dcgan", default="checkpoints/dcgan/dcgan_G.pt")
    p.add_argument("--num-codes", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--prior-width", type=int, default=128)
    p.add_argument("--prior-blocks", type=int, default=4)
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--audit-train", type=int, default=2000,
                   help="training slices subsampled for the nearest-neighbour audit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs")
    return p.parse_args()


def to_rgb_uint8(images, device):
    """[N,1,H,W] float [0,1] -> [N,3,H,W] uint8 on device.

    InceptionV3 expects 3-channel input, so the single MRI channel is
    repeated; torchmetrics KID requires uint8 images.
    """
    x = (images.clamp(0, 1) * 255).to(torch.uint8)
    return x.repeat(1, 3, 1, 1).to(device)


def save_grid(images, path, title, cols=8):
    """Uncurated sample grid (fixed seeds upstream, no cherry-picking)."""
    n = min(len(images), 64)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2.4 * rows))
    for ax in axs.flat:
        ax.axis("off")
    for ax, img in zip(axs.flat, images[:n]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def recon_ssim(model, test, batch_size, device):
    """Mean test SSIM and the fraction of slices meeting the 0.6 criterion."""
    model.eval()
    total, n, passed = 0.0, 0, 0
    for i in tqdm(range(0, len(test), batch_size), desc="test SSIM",
                   leave=False, unit="batch"):
        x = test._cache[i:i + batch_size].to(device)
        x_hat = model(x)[0].cpu()
        for a, b in zip(x.cpu(), x_hat):
            s = ssim(a[0].numpy(), b[0].numpy(), data_range=1.0)
            total += s
            passed += s >= 0.6
            n += 1
    return total / n, passed / n


@torch.no_grad()
def nearest_train_matches(images, train, device):
    """For each image, its most similar training slice (screening audit).

    Distance = squared L2 on flattened pixels. Brute force is adequate:
    a few hundred queries x ~2000 training slices of 32k pixels is a
    trivial GPU matmul. Returns (best index into train, best distance).
    """
    flat = images.reshape(len(images), -1)
    tr = train.reshape(len(train), -1).to(device)
    best_idx, best_dist = [], []
    for i in tqdm(range(0, len(flat), 100), desc="NN audit",
                  leave=False, unit="chunk"):
        d = torch.cdist(flat[i:i + 100].to(device), tr)  # [chunk, n_train]
        v, j = d.min(dim=1)
        best_idx.append(j.cpu())
        best_dist.append(v.cpu())
    return torch.cat(best_idx), torch.cat(best_dist)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}

    patients = verify_patient_splits(args.data_root)
    print("patient-disjoint split check passed:", patients, flush=True)

    test = HipMRISlices(args.data_root, "test", preload=True)
    train = HipMRISlices(args.data_root, "train", preload=True)
    print(f"test {len(test)} slices, train {len(train)} slices", flush=True)

    # ------------------------------------------------------------------ VQ-VAE
    vqvae = VQVAE(width=args.width, latent_dim=args.latent_dim,
                  num_codes=args.num_codes, beta=args.beta).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae, map_location=device))

    mean_ssim, frac = recon_ssim(vqvae, test, 64, device)
    results["vqvae_test_ssim_mean"] = mean_ssim
    results["vqvae_test_ssim_frac_ge_0.6"] = frac
    print(f"VQ-VAE test SSIM: mean {mean_ssim:.4f}, {frac:.1%} of slices >= 0.6",
          flush=True)

    # Reconstruction grid (input | reconstruction pairs) for the report.
    with torch.no_grad():
        xs = test._cache[:8].to(device)
        recons = vqvae(xs)[0].cpu()
    fig, axs = plt.subplots(4, 4, figsize=(8, 10))
    for r in range(8):
        axs[r % 4, 2 * (r // 4)].imshow(xs[r][0].cpu(), cmap="gray")
        axs[r % 4, 2 * (r // 4) + 1].imshow(recons[r][0], cmap="gray")
    for ax in axs.flat:
        ax.axis("off")
    fig.suptitle("Test slices (left of each pair) and VQ-VAE reconstructions")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "recon_grid.png"), dpi=130)
    plt.close(fig)

    # Codebook utilisation on the test split.
    with torch.no_grad():
        codes = vqvae.encode_to_codes(test._cache.to(device)).cpu()
    used = torch.unique(codes)
    hist = torch.bincount(codes.flatten(), minlength=args.num_codes).float()
    p = hist / hist.sum()
    results["codes_used"] = int(len(used))
    results["codebook_perplexity_test"] = float(
        torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    print(f"codebook: {len(used)}/{args.num_codes} codes used "
          f"(perplexity {results['codebook_perplexity_test']:.1f})", flush=True)

    # ------------------------------------------------- PixelCNN prior sampling
    prior = PixelCNNPrior(num_codes=args.num_codes, width=args.prior_width,
                           num_blocks=args.prior_blocks,
                           grid_hw=tuple(codes.shape[1:])).to(device)
    prior.load_state_dict(torch.load(args.prior, map_location=device))
    prior.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        bar = tqdm(total=64 * 32, desc="sampling codes", unit="pos")
        sampled_codes = prior.sample(args.n_samples, device, temperature=1.0,
                                     progress=lambda d, t: bar.update(1))
        bar.close()
        vq_samples = vqvae.decode_from_codes(sampled_codes).cpu()
    results["vqvae_gen_seconds"] = time.time() - t0
    results["vqvae_gen_peak_vram_gb"] = (
        torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0)
    print(f"prior+decoder generated {args.n_samples} images in "
          f"{results['vqvae_gen_seconds']:.1f}s", flush=True)
    save_grid([s[0].numpy() for s in vq_samples],
              os.path.join(args.out, "vqvae_samples.png"),
              "VQ-VAE + PixelCNN samples (uncurated, fixed seed)")

    # ------------------------------------------------------------------- DCGAN
    netG = DCGenerator(noise_dim=args.noise_dim).to(device)
    netG.load_state_dict(torch.load(args.dcgan, map_location=device))
    netG.eval()
    t0 = time.time()
    with torch.no_grad():
        gan_samples = netG(torch.randn(args.n_samples, args.noise_dim, device=device))
    gan_samples = gan_samples.mul(0.5).add(0.5).clamp(0, 1).cpu()  # tanh -> [0,1]
    results["dcgan_gen_seconds"] = time.time() - t0
    print(f"DCGAN generated {args.n_samples} images in "
          f"{results['dcgan_gen_seconds']:.1f}s", flush=True)
    save_grid([s[0].numpy() for s in gan_samples],
              os.path.join(args.out, "dcgan_samples.png"),
              "DCGAN samples (uncurated, fixed seed)")

    # ------------------------------------------------------------------- KID
    # KID compares Inception feature distributions of samples vs held-out
    # real test images; lower is better. It is chosen over FID because its
    # estimator is unbiased at small sample sizes (Binkowski et al., 2018),
    # which matters with only 540 test slices and 500 samples.
    kid = KernelInceptionDistance(subsets=8, subset_size=32).to(device)
    real_uint8 = to_rgb_uint8(test._cache, device)
    for name, samp in (("vqvae_prior", vq_samples), ("dcgan", gan_samples)):
        fake_uint8 = to_rgb_uint8(samp, device)
        kid.update(real_uint8, real=True)
        kid.update(fake_uint8, real=False)
        mean, std = kid.compute()
        results[f"kid_{name}_mean"] = float(mean)
        results[f"kid_{name}_std"] = float(std)
        print(f"KID {name}: {float(mean):.4f} +/- {float(std):.4f}", flush=True)
        kid.reset()

    # ------------------------------------------------- Memorisation audit
    # Compare how close generated images are to their nearest training
    # slice versus how close held-out real test images are to theirs.
    # Distances much below the real-held-out baseline suggest copying.
    g = torch.Generator().manual_seed(args.seed)
    audit_idx = torch.randperm(len(train), generator=g)[:args.audit_train]
    audit_train = train._cache[audit_idx]

    for name, samp in (("vqvae_prior", vq_samples), ("dcgan", gan_samples)):
        _, dist = nearest_train_matches(samp, audit_train, device)
        results[f"{name}_nn_dist_mean"] = float(dist.mean())
        results[f"{name}_nn_dist_min"] = float(dist.min())
        print(f"{name}: nearest-train L2 mean {dist.mean():.2f}, "
              f"min {dist.min():.2f}", flush=True)

    _, real_dist = nearest_train_matches(test._cache, audit_train, device)
    results["real_test_nn_dist_mean"] = float(real_dist.mean())
    results["real_test_nn_dist_min"] = float(real_dist.min())
    print(f"real held-out: nearest-train L2 mean {real_dist.mean():.2f}, "
          f"min {real_dist.min():.2f}", flush=True)

    # Autopsy figure: the 4 most-suspicious VQ-VAE samples (smallest
    # nearest-train distance) shown beside their closest training slice.
    idx, dist = nearest_train_matches(vq_samples, audit_train, device)
    order = torch.argsort(dist)[:4]
    fig, axs = plt.subplots(4, 2, figsize=(4, 10))
    for r, i in enumerate(order):
        axs[r, 0].imshow(vq_samples[i][0].numpy(), cmap="gray")
        axs[r, 1].imshow(audit_train[int(idx[int(i)])][0].numpy(), cmap="gray")
        axs[r, 0].set_ylabel(f"L2={dist[int(i)]:.1f}", fontsize=8)
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_title("generated"); axs[0, 1].set_title("nearest train")
    fig.suptitle("Memorisation audit: closest generated/training pairs")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "memorisation_audit.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("results written to", os.path.join(args.out, "results.json"), flush=True)


if __name__ == "__main__":
    main()