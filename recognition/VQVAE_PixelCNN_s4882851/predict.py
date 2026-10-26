"""Predeclared held-out evaluation of VQ-VAE/PixelCNN and DCGAN.

Reconstruction SSIM is reported separately from unconditional generation SSIM.
Unconditional samples have no paired ground truth: generated-to-reference SSIM
uses the nearest held-out slice by pixel L2 (explicitly a biased similarity
screen, not paired reconstruction accuracy). KID is a distribution metric;
ImageNet features on repeated grayscale MRI are not clinical validation.

Reports fixed-seed grids, runtime for 500 samples, per-patient reconstruction
scores, full-train nearest neighbours, diversity, latent interpolations and
distance-selected failure candidates. None establishes differential privacy.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from dataset import DEFAULT_ROOT, HipMRISlices, patient_id, verify_patient_splits
from runtime import (atomic_json, atomic_save, device_for, load_model, log,
                     progress, seed_all, sha256)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--vqvae", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--dcgan", default="", help="optional baseline checkpoint")
    p.add_argument("--split", choices=("validate", "test"), default="test")
    p.add_argument("--n-samples", type=int, default=500, help="samples PER seed")
    p.add_argument("--sample-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--audit-train", type=int, default=0, help="0=ALL training slices")
    p.add_argument("--limit", type=int, default=0, help="debug only, cap data loading")
    p.add_argument("--skip-kid", action="store_true", help="debug only; requires --limit")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="outputs/main")
    args = p.parse_args()
    if min(args.n_samples, args.batch_size, args.threads) < 1 or args.temperature <= 0:
        p.error("sample count, batch size, threads, temperature must be positive")
    if args.skip_kid and not args.limit:
        p.error("--skip-kid is only allowed with --limit (non-official smoke test)")
    if len(set(args.sample_seeds)) != len(args.sample_seeds):
        p.error("sample seeds must be distinct")
    return args


def to_rgb_uint8(images, device):
    return (images.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1).to(device)


def save_grid(images, path, title, cols=8):
    n = min(len(images), 64)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.3 * rows), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, img in zip(axs.flat, images[:n]):
        if getattr(img, "ndim", 0) == 3:
            img = img[0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


@torch.no_grad()
def recon_scores(model, data, batch_size, device):
    scores, hist, grid_codes, grid_recons = [], torch.zeros(model.quantizer.num_codes), None, None
    for x in progress(torch.split(data._cache, batch_size), "reconstruction SSIM and codes"):
        recon, _, _, codes = model(x.to(device))
        recon, codes = recon.cpu(), codes.cpu()
        if grid_codes is None:
            grid_codes, grid_recons = codes[:8], recon[:8]
        hist += torch.bincount(codes.flatten(), minlength=len(hist))
        scores.extend(float(ssim(a[0].numpy(), b[0].numpy(), data_range=1.0))
                      for a, b in zip(x, recon))
    return np.array(scores), hist, grid_codes, grid_recons


@torch.no_grad()
def nearest_train_matches(images, train, device, chunk=32, reference_chunk=512):
    """Exact pixel Euclidean NN, chunked on both axes to bound GPU memory."""
    refs = train.flatten(1)
    indices, distances = [], []
    for queries in progress(torch.split(images.flatten(1), chunk), "nearest-neighbour audit"):
        q = queries.to(device)
        best = torch.full((len(q),), float("inf"), device=device)
        idx = torch.zeros(len(q), dtype=torch.long, device=device)
        for start in range(0, len(refs), reference_chunk):
            r = refs[start:start + reference_chunk].to(device)
            squared = (q.square().sum(1, keepdim=True) + r.square().sum(1)
                       - 2 * q @ r.t()).clamp_min(0)
            values, positions = squared.min(1)
            mask = values < best
            idx[mask] = positions[mask] + start
            best = torch.minimum(best, values)
        indices.append(idx.cpu())
        distances.append(best.sqrt().cpu())
    return torch.cat(indices), torch.cat(distances)


def distance_summary(dist, pixels):
    rmse = dist / pixels ** 0.5
    return {"l2_mean": float(dist.mean()), "l2_min": float(dist.min()),
            "rmse_mean": float(rmse.mean()),
            "rmse_quantiles_0_5_50_95_100": [float(v) for v in
                torch.quantile(rmse, torch.tensor([0., .05, .5, .95, 1.]))]}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate(args, model, decoder, name, seed, device, out):
    cache = out / f"{name}_seed{seed}.pt"
    timing = out / f"{name}_seed{seed}_timing.json"
    if cache.exists() and timing.exists():
        log(f"reusing verified-config generation cache: {cache}")
        return torch.load(cache, map_location="cpu", weights_only=True), json.loads(timing.read_text())
    seed_all(seed)
    batches = []
    noises = []
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    for start in range(0, args.n_samples, args.batch_size):
        n = min(args.batch_size, args.n_samples - start)
        if name == "vqvae_prior":
            last = time.monotonic()
            def callback(done, total):
                nonlocal last
                now = time.monotonic()
                if now - last >= 30 or done == total:
                    log(f"seed={seed} images={start}/{args.n_samples} "
                        f"positions={done}/{total} sampling_elapsed={now-t0:.1f}s")
                    last = now
            codes = model.sample(n, device, args.temperature, callback)
            batches.append(decoder.decode_from_codes(codes).cpu())
        else:
            z = torch.randn(n, model.noise_dim, device=device)
            noises.append(z.cpu())
            batches.append(((model(z) + 1) / 2).clamp(0, 1).cpu())
        log(f"{name} seed={seed}: generated {start+n}/{args.n_samples}")
    sync(device)
    seconds = time.monotonic() - t0
    samples = torch.cat(batches)
    if not torch.isfinite(samples).all():
        raise FloatingPointError("non-finite generated images")
    info = {"seconds": seconds, "n_samples": args.n_samples,
            "meets_500_in_900s": seconds <= 900 if args.n_samples == 500 else None,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0}
    atomic_save(samples, cache)
    atomic_json(info, timing)
    if noises:
        atomic_save(torch.cat(noises), out / f"noise_seed{seed}.pt")
    save_grid(samples, out / f"{name}_seed{seed}.png",
              f"{name}: first 64, seed={seed}, temperature={args.temperature:g}")
    return samples, info


def kid_scores(real, fake, device, seed, batch_size):
    # Import/download is deliberately preflighted before expensive GPU jobs.
    from torchmetrics.image.kid import KernelInceptionDistance
    subset = min(100, len(real), len(fake))
    if subset < 2:
        raise ValueError("KID needs at least two real and two generated images")
    seed_all(seed)
    metric = KernelInceptionDistance(subsets=50, subset_size=subset).to(device)
    with torch.no_grad():
        for x in progress(torch.split(real, batch_size), "KID real features"):
            metric.update(to_rgb_uint8(x, device), real=True)
        for x in progress(torch.split(fake, batch_size), "KID generated features"):
            metric.update(to_rgb_uint8(x, device), real=False)
    mean, std = metric.compute()
    return {"mean": float(mean), "subset_std": float(std), "subset_size": subset, "subsets": 50}


@torch.no_grad()
def interpolation_figure(vqvae, prior_samples, nearest_images, gan, noise, device, out):
    fig, axs = plt.subplots(2 + int(gan is not None), 5, figsize=(10, 6), squeeze=False)
    for row in range(2):
        source = prior_samples[row:row+1].to(device)
        target = nearest_images[row:row+1].to(device)
        a = vqvae.quantizer.codebook(vqvae.encode_to_codes(source)).permute(0, 3, 1, 2)
        b = vqvae.quantizer.codebook(vqvae.encode_to_codes(target)).permute(0, 3, 1, 2)
        for col, alpha in enumerate(torch.linspace(0, 1, 5)):
            z = (1 - float(alpha)) * a + float(alpha) * b
            quantized = vqvae.quantizer(z)[0]
            img = vqvae.decoder(quantized)[0, 0].cpu()
            axs[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axs[row, col].set_title(f"alpha={float(alpha):.2f}")
        axs[row, 0].set_ylabel("VQ synth -> nearest train")
    if gan is not None:
        for col, alpha in enumerate(torch.linspace(0, 1, 5)):
            z = ((1-float(alpha))*noise[0:1] + float(alpha)*noise[1:2]).to(device)
            img = ((gan(z)[0, 0].cpu() + 1) / 2).clamp(0, 1)
            axs[2, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axs[2, 0].set_ylabel("GAN noise interpolation")
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Latent interpolation (screening only; no privacy guarantee)")
    fig.tight_layout(); fig.savefig(out / "interpolation.png", dpi=140); plt.close(fig)


@torch.no_grad()
def evaluate(args, out, device):
    log(f"patient-disjoint split check: {verify_patient_splits(args.data_root)}")
    data = HipMRISlices(args.data_root, args.split, limit=args.limit)
    train = HipMRISlices(args.data_root, "train", limit=args.limit)
    val = HipMRISlices(args.data_root, "validate", limit=args.limit)
    if args.audit_train:
        idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:args.audit_train]
        train._cache = train._cache[idx]
        train.paths = [train.paths[int(i)] for i in idx]
    vqvae, _ = load_model(args.vqvae, "vqvae", device)
    prior, prior_config = load_model(args.prior, "prior", device)
    if prior_config["vqvae_sha256"] != sha256(args.vqvae):
        raise ValueError("prior was trained on a different VQ-VAE checkpoint")
    gan = load_model(args.dcgan, "dcgan", device)[0] if args.dcgan else None
    results = {"split": args.split, "debug_limit": args.limit, "reference_slices": len(data),
               "audit_training_slices": len(train), "full_train_audit": not args.audit_train and not args.limit,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
               "checkpoint_sha256": {k: sha256(v) for k, v in
                    (("vqvae", args.vqvae), ("prior", args.prior), ("dcgan", args.dcgan)) if v},
               "sample_seeds": args.sample_seeds, "temperature": args.temperature}
    scores, hist, codes, recons = recon_scores(vqvae, data, args.batch_size, device)
    prior.grid_hw = tuple(codes.shape[1:])
    pairs = torch.stack([data._cache[:len(recons)], recons], dim=1).flatten(0, 1)
    save_grid(pairs, out / "reconstruction.png", "Reference | reconstruction (fixed first slices)")
    probs = hist / hist.sum()
    per_patient = {}
    for pid in sorted({patient_id(p) for p in data.paths}):
        values = [s for p, s in zip(data.paths, scores) if patient_id(p) == pid]
        per_patient[str(pid)] = {"n": len(values), "mean_ssim": float(np.mean(values))}
    results["reconstruction"] = {"mean_ssim": float(scores.mean()),
        "fraction_ge_0.6": float((scores >= .6).mean()), "per_patient": per_patient,
        "codes_used": int((hist > 0).sum()),
        "perplexity": float(torch.exp(-(probs[probs > 0] * probs[probs > 0].log()).sum()))}
    with open(out / "reconstruction_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "patient", "ssim"])
        w.writerows((Path(p).name, patient_id(p), float(s)) for p, s in zip(data.paths, scores))
    worst = np.argsort(scores)[:4]
    save_grid(torch.stack([data._cache[worst], vqvae(data._cache[worst].to(device))[0].detach().cpu()],
                         dim=1).flatten(0, 1),
              out / "worst_reconstructions.png", "Four lowest-SSIM references | reconstructions")
    atomic_json(results, out / "partial_results.json")
    _, real_dist = nearest_train_matches(data._cache, train._cache, device)
    _, val_dist = nearest_train_matches(val._cache, train._cache, device)
    pixels = data._cache[0].numel()
    # A predeclared screening threshold calibrated on VALIDATION, not test.
    threshold = float(torch.quantile(val_dist / pixels ** .5, .05))
    results["reference_nn"] = distance_summary(real_dist, pixels)
    results["validation_calibrated_rmse_threshold_5pct"] = threshold
    cohorts, nearest = {}, {}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist((real_dist / pixels ** .5).numpy(), bins=30, density=True, alpha=.4, label="real held-out")
    for name, model in (("vqvae_prior", prior), ("dcgan", gan)):
        if model is None:
            continue
        sets, times = [], {}
        for seed in args.sample_seeds:
            samples, info = generate(args, model, vqvae, name, seed, device, out)
            sets.append(samples); times[str(seed)] = info
        samples = torch.cat(sets)
        cohorts[name] = samples
        idx, dist = nearest_train_matches(samples, train._cache, device)
        nearest[name] = idx
        ref_idx, _ = nearest_train_matches(samples, data._cache, device)
        generated_ssim = [float(ssim(x[0].numpy(), data._cache[int(j)][0].numpy(), data_range=1.0))
                          for x, j in zip(samples, ref_idx)]
        item = {"n": len(samples), "generation_per_seed": times,
                "nn_train": distance_summary(dist, pixels),
                "fraction_below_validation_screen": float((dist / pixels ** .5 < threshold).float().mean()),
                "nearest_reference_ssim_mean": float(np.mean(generated_ssim)),
                "nearest_reference_ssim_fraction_ge_0.6": float(np.mean(np.array(generated_ssim) >= .6)),
                "exact_uint8_nearest_training_matches": int(
                    (((samples * 255).to(torch.uint8) ==
                      (train._cache[idx] * 255).to(torch.uint8)).flatten(1).all(1)).sum()),
                "across_sample_pixel_std": float(samples.std(dim=0).mean()),
                "unique_images_after_uint8": int(torch.unique((samples*255).to(torch.uint8).flatten(1), dim=0).size(0))}
        if not args.skip_kid:
            item["kid"] = kid_scores(data._cache, samples, device, 2026, min(args.batch_size, 16))
        results[name] = item
        with open(out / f"{name}_audit.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sample", "train_file", "train_l2", "train_rmse", "nearest_reference_ssim"])
            w.writerows((i, Path(train.paths[int(j)]).name, float(d), float(d/pixels**.5), s)
                        for i, (j, d, s) in enumerate(zip(idx, dist, generated_ssim)))
        order = dist.argsort()[:4]
        pairs = torch.stack([samples[order], train._cache[idx[order]]], dim=1).flatten(0, 1)
        save_grid(pairs, out / f"{name}_closest_training_pairs.png",
                  "Four closest generated | training pairs (distance-selected)")
        order = torch.tensor(np.argsort(generated_ssim)[:4].copy())
        pairs = torch.stack([samples[order], data._cache[ref_idx[order]]], dim=1).flatten(0, 1)
        save_grid(pairs, out / f"{name}_failure_candidates.png",
                  "Four lowest nearest-reference SSIM: generated | reference (review required)")
        ax.hist((dist / pixels ** .5).numpy(), bins=30, density=True, alpha=.4, label=name)
        atomic_json(results, out / "partial_results.json")
    ax.axvline(threshold, ls="--", color="black", label="validation 5th percentile")
    ax.set(xlabel="Nearest-training pixel RMSE", ylabel="density")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "nn_distance_distributions.png", dpi=150)
    plt.close(fig)
    noise = torch.load(out / f"noise_seed{args.sample_seeds[0]}.pt", weights_only=True) if gan else None
    interpolation_figure(vqvae, cohorts["vqvae_prior"], train._cache[nearest["vqvae_prior"]],
                         gan, noise, device, out)
    results["complete"] = True
    atomic_json(results, out / "results.json")
    log(f"COMPLETE evaluation: {out / 'results.json'}")


def main():
    args = parse_args()
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["weights_sha256"] = {p: sha256(p) for p in (args.vqvae, args.prior, args.dcgan) if p}
    config_path = out / "evaluation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("evaluation output has a different config; use a new --out")
    atomic_json(config, config_path)
    if (out / "results.json").exists():
        log(f"already completed: {out / 'results.json'}"); return
    atomic_json({"state": "running", "updated": time.time()}, out / "status.json")
    try:
        evaluate(args, out, device)
        atomic_json({"state": "completed", "updated": time.time()}, out / "status.json")
    except Exception as exc:
        atomic_json({"state": "failed", "error": repr(exc), "updated": time.time()}, out / "status.json")
        raise


if __name__ == "__main__":
    main()