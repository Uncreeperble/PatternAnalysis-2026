"""Train VQ-VAE, its masked PixelCNN prior, or the stabilized DCGAN baseline.

Selection is validation SSIM for VQ-VAE, validation NLL for the prior,
and a predeclared final epoch for DCGAN (never selection on test KID).
All runs save configuration, per-epoch CSV/plots, atomic best weights,
and an epoch-boundary resume checkpoint including optimizers and RNG.

References: van den Oord et al. (2017), Radford et al. (2016),
Salimans et al. (2016; label smoothing), Miyato et al. (2018).
"""
import argparse
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from dataset import (DEFAULT_ROOT, MASK_CLASSES, HipMRISlices, list_mask_split,
                     load_mask, verify_image_mask_alignment, verify_patient_splits)
from modules import (VQVAE, CondPixelCNNPrior, GatedPixelCNNPrior, PixelCNNPrior,
                     TransformerPrior, DCGenerator, DCDiscriminator)
from runtime import (atomic_json, atomic_save, checkpoint_epoch, device_for,
                     finite_loss, load_model, log, progress, restore, sha256, start_run)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("vqvae", "prior", "dcgan"), default="vqvae")
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-codes", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--codebook-init", choices=("uniform", "data"), default="uniform")
    p.add_argument("--downsample", type=int, choices=(4, 8), default=4)
    p.add_argument("--recon-loss", choices=("mse", "l1"), default="mse")
    p.add_argument("--prior-width", type=int, default=128)
    p.add_argument("--prior-blocks", type=int, default=4)
    p.add_argument("--prior-kernel", type=int, choices=(3, 7), default=3)
    p.add_argument("--prior-embedding", type=int, default=32)
    p.add_argument("--prior-type",
                   choices=("pixelcnn", "transformer", "cond", "gated", "gated-cond"),
                   default="pixelcnn")
    p.add_argument("--prior-layers", type=int, default=22,
                   help="gated prior only: number of gated layers. The vertical "
                        "receptive field is layers*(kernel//2) rows, which must "
                        "reach the code-grid height or the prior cannot place "
                        "anatomy globally.")
    p.add_argument("--prior-heads", type=int, default=8, help="transformer prior only")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Adam L2 weight decay for the prior (0 = unchanged campaign behaviour)")
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--gan-width", type=int, default=256)
    p.add_argument("--disc-width", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="debug cap BEFORE loading, not an official run")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--vqvae-checkpoint", default="checkpoints/vqvae/vqvae_best.pt")
    p.add_argument("--out", default="checkpoints/vqvae")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.limit < 0:
        p.error("epochs, batch-size and threads must be positive; limit must be nonnegative")
    if args.model == "dcgan" and args.batch_size < 2:
        p.error("DCGAN training needs batch-size >=2 for BatchNorm")
    return args


@torch.no_grad()
def validation_ssim(model, val, batch_size):
    """Patient-disjoint validation only; transfer input/output explicitly."""
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for x in progress(torch.split(val._cache, batch_size), "validation SSIM"):
        recon = model(x.to(device))[0].cpu()
        scores.extend(float(ssim(a[0].numpy(), b[0].numpy(), data_range=1.0))
                      for a, b in zip(x, recon))
    return sum(scores) / len(scores)


@torch.no_grad()
def encode_split(vqvae, data, batch_size, device):
    vqvae.eval()
    return torch.cat([vqvae.encode_to_codes(x.to(device)).cpu()
                      for x in progress(torch.split(data._cache, batch_size), "encoding")])


def code_grid_masks(root, split, grid_hw, limit=0):
    """Label masks downsampled to the code grid, as uint8 [N, h, w].

    Nearest-neighbour by exact striding (256/64 = 4, 128/32 = 4), which keeps
    label values intact; averaging would invent classes that do not exist.

    Each mask is strided as it is read rather than preloading the split through
    HipMRIMasks: the full-resolution int64 train cache would be ~3 GB purely to
    be downsampled immediately. The result here is ~22 MB as uint8, expanded to
    one-hot per batch.
    """
    paths = list_mask_split(root, split)
    if limit:
        paths = paths[:limit]
    h, w = grid_hw
    out = torch.empty(len(paths), h, w, dtype=torch.uint8)
    for i, path in enumerate(progress(paths, f"loading {split} masks")):
        arr = load_mask(path)
        sh, sw = arr.shape[0] // h, arr.shape[1] // w
        if sh * h != arr.shape[0] or sw * w != arr.shape[1]:
            raise ValueError(f"mask shape {arr.shape} not divisible by grid {grid_hw}")
        out[i] = torch.from_numpy(arr[::sh, ::sw]).clamp(0, MASK_CLASSES - 1).to(torch.uint8)
    return out


def one_hot_masks(batch, device):
    """uint8 [B, h, w] label grid -> float one-hot [B, MASK_CLASSES, h, w]."""
    return F.one_hot(batch.to(device).long(), MASK_CLASSES).permute(0, 3, 1, 2).float()


def save_plots(history, out_dir):
    keys = [k for k in history if k != "epoch_seconds"]
    fig, axs = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5), squeeze=False)
    for ax, key in zip(axs.flat, keys):
        ax.plot(range(1, len(history[key]) + 1), history[key])
        ax.set(title=key, xlabel="epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "training_curves.png", dpi=130)
    plt.close(fig)


def finish(args, models, history, best, device, peak, extras=None):
    summary = {
        "model": args.model, "epochs": len(history["epoch_seconds"]),
        "seed": args.seed, "debug_limit": args.limit, "batch_size": args.batch_size,
        "lr": args.lr, "best_selection_metric": best,
        "parameters": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
        "epoch_seconds_avg": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        "peak_vram_gb": peak,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        **(extras or {}),
    }
    atomic_json(summary, Path(args.out) / "summary.json")
    atomic_json({"state": "completed", "epoch": summary["epochs"],
                 "updated": time.time()}, Path(args.out) / "status.json")
    log(f"COMPLETE {args.out}: {summary}")


def train_vqvae(args, device):
    train = HipMRISlices(args.data_root, "train", limit=args.limit)
    val = HipMRISlices(args.data_root, "validate", limit=max(1, args.limit // 4) if args.limit else 0)
    model = VQVAE(width=args.width, latent_dim=args.latent_dim,
                  num_codes=args.num_codes, beta=args.beta, downsample=args.downsample).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    models, opts = {"vqvae": model}, {"vqvae": opt}
    epoch0, history, best = restore(args, models, opts, device)
    history = history or {k: [] for k in ("train_recon", "train_vq", "val_ssim",
                                          "perplexity", "epoch_seconds")}
    if args.codebook_init == "data" and not args.resume:
        # Preplanned collapse fallback: initialize from TRAIN encoder outputs,
        # never from validation/test. No architecture or loss changes.
        with torch.no_grad():
            z = model.encoder(train._cache[:min(32, len(train))].to(device))
            flat = z.permute(0, 2, 3, 1).reshape(-1, args.latent_dim)
            idx = torch.randint(len(flat), (args.num_codes,), device=device)
            model.quantizer.codebook.weight.copy_(flat[idx])
        log("codebook initialized from training encoder vectors")
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    log(f"train={len(train)} validation={len(val)} parameters="
        f"{sum(p.numel() for p in model.parameters())}")
    for epoch in range(epoch0 + 1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        recon_sum = vq_sum = 0.0
        counts = torch.zeros(args.num_codes)
        for x in progress(loader, f"VQ epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            recon, vq, _, codes = model(x)
            mse = F.mse_loss(recon, x)
            objective = mse if args.recon_loss == "mse" else F.l1_loss(recon, x)
            finite_loss(objective + vq)
            opt.zero_grad(set_to_none=True)
            (objective + vq).backward()
            opt.step()
            recon_sum += mse.item() * len(x)
            vq_sum += vq.item() * len(x)
            counts += torch.bincount(codes.detach().flatten().cpu(), minlength=args.num_codes)
        score = validation_ssim(model, val, args.batch_size)
        probs = counts / counts.sum()
        perplexity = float(torch.exp(-(probs[probs > 0] * probs[probs > 0].log()).sum()))
        for key, value in (("train_recon", recon_sum / len(train)),
                           ("train_vq", vq_sum / len(train)), ("val_ssim", score),
                           ("perplexity", perplexity),
                           ("epoch_seconds", time.monotonic() - start)):
            history[key].append(value)
        if score > best:
            best = score
            atomic_save(model.state_dict(), Path(args.out) / "vqvae_best.pt")
            atomic_json({"epoch": epoch, "val_ssim": score}, Path(args.out) / "best.json")
        checkpoint_epoch(args, epoch, history, best, models, opts)
        save_plots(history, args.out)
        log(f"epoch={epoch} val_SSIM={score:.5f} perplexity={perplexity:.1f} "
            f"epoch_seconds={history['epoch_seconds'][-1]:.1f}")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
    finish(args, models, history, best, device, peak,
           {"final_perplexity": history["perplexity"][-1], "best_val_ssim": best})


def train_prior(args, device, vqvae):
    train = HipMRISlices(args.data_root, "train", limit=args.limit)
    val = HipMRISlices(args.data_root, "validate", limit=max(1, args.limit // 4) if args.limit else 0)
    train_codes = encode_split(vqvae, train, args.batch_size, device)
    val_codes = encode_split(vqvae, val, args.batch_size, device)
    del train, val, vqvae
    grid_hw = tuple(train_codes.shape[1:])
    train_masks = val_masks = None
    conditional = args.prior_type in ("cond", "gated-cond")
    if conditional:
        # Conditioning on the segmentation labels targets anatomical layout,
        # which the unconditional prior loses: each code position only sees
        # row_radius rows above it, so it cannot place the hip globally.
        log(f"image/mask alignment: {verify_image_mask_alignment(args.data_root, 'train')} "
            f"{verify_image_mask_alignment(args.data_root, 'validate')}")
        train_masks = code_grid_masks(args.data_root, "train", grid_hw, args.limit)
        val_masks = code_grid_masks(args.data_root, "validate", grid_hw,
                                    max(1, args.limit // 4) if args.limit else 0)
        if len(train_masks) != len(train_codes) or len(val_masks) != len(val_codes):
            raise ValueError("code/mask counts differ; refusing to train on misaligned pairs")
    if args.prior_type in ("gated", "gated-cond"):
        model = GatedPixelCNNPrior(num_codes=args.num_codes, width=args.prior_width,
                                   num_layers=args.prior_layers,
                                   embedding_dim=args.prior_embedding,
                                   kernel_size=args.prior_kernel,
                                   mask_classes=MASK_CLASSES if conditional else 0,
                                   grid_hw=grid_hw).to(device)
        # Refuse to train a prior that structurally cannot see the whole grid:
        # that is the defect this model exists to fix, so it must be checked.
        if not model.covers_grid():
            raise ValueError(
                f"gated prior receptive field {model.vertical_receptive_field} rows "
                f"< grid height {grid_hw[0]}; increase --prior-layers or --prior-kernel")
        log(f"gated prior: {args.prior_layers} layers, kernel {args.prior_kernel}, "
            f"vertical receptive field {model.vertical_receptive_field} rows "
            f">= grid height {grid_hw[0]}")
    elif args.prior_type == "cond":
        model = CondPixelCNNPrior(num_codes=args.num_codes, width=args.prior_width,
                                  num_blocks=args.prior_blocks,
                                  embedding_dim=args.prior_embedding,
                                  kernel_size=args.prior_kernel,
                                  mask_classes=MASK_CLASSES, grid_hw=grid_hw).to(device)
    elif args.prior_type == "transformer":
        model = TransformerPrior(num_codes=args.num_codes, width=args.prior_width,
                                 num_layers=args.prior_blocks, num_heads=args.prior_heads,
                                 grid_hw=grid_hw).to(device)
    else:
        model = PixelCNNPrior(num_codes=args.num_codes, width=args.prior_width,
                          num_blocks=args.prior_blocks, embedding_dim=args.prior_embedding,
                          kernel_size=args.prior_kernel,
                          grid_hw=grid_hw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    models, opts = {"prior": model}, {"prior": opt}
    epoch0, history, best = restore(args, models, opts, device)
    history = history or {k: [] for k in ("train_nll", "val_nll", "epoch_seconds")}
    # One dataset of (codes, mask) pairs, so shuffling can never separate them.
    train_set = (torch.utils.data.TensorDataset(train_codes, train_masks)
                 if train_masks is not None else train_codes)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_batches = (list(zip(torch.split(val_codes, args.batch_size),
                            torch.split(val_masks, args.batch_size)))
                   if val_masks is not None else torch.split(val_codes, args.batch_size))

    def batch_loss(batch):
        if train_masks is None:
            return model.loss(batch.to(device)), len(batch)
        codes, mask = batch
        return model.loss(codes.to(device), one_hot_masks(mask, device)), len(codes)

    for epoch in range(epoch0 + 1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        total = 0.0
        for batch in progress(loader, f"prior epoch {epoch}/{args.epochs}"):
            loss, n = batch_loss(batch)
            finite_loss(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * n
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in progress(val_batches, "validation NLL"):
                loss, n = batch_loss(batch)
                finite_loss(loss)
                val_total += loss.item() * n
        score = val_total / len(val_codes)  # weighted, including the short last batch
        history["train_nll"].append(total / len(train_codes))
        history["val_nll"].append(score)
        history["epoch_seconds"].append(time.monotonic() - start)
        if score < best:
            best = score
            atomic_save(model.state_dict(), Path(args.out) / "prior_best.pt")
            atomic_json({"epoch": epoch, "val_nll": score}, Path(args.out) / "best.json")
        checkpoint_epoch(args, epoch, history, best, models, opts)
        save_plots(history, args.out)
        log(f"epoch={epoch} val_NLL={score:.5f} nats ({score / math.log(2):.5f} bits/code)")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
    finish(args, models, history, best, device, peak, {"best_val_nll_nats": best})


def train_dcgan(args, device):
    train = HipMRISlices(args.data_root, "train", limit=args.limit)
    netG = DCGenerator(noise_dim=args.noise_dim, width=args.gan_width).to(device)
    netD = DCDiscriminator(width=args.disc_width).to(device)
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    models, opts = {"G": netG, "D": netD}, {"G": optG, "D": optD}
    # Local RNG prevents monitoring samples altering the training trajectory.
    fixed = torch.randn(32, args.noise_dim,
                        generator=torch.Generator().manual_seed(args.seed + 10000)).to(device)
    epoch0, history, _ = restore(args, models, opts, device)
    history = history or {k: [] for k in ("d_real", "d_fake", "g_loss",
                                          "sample_std", "epoch_seconds")}
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        drop_last=(len(train) % args.batch_size == 1))
    for epoch in range(epoch0 + 1, args.epochs + 1):
        start = time.monotonic()
        netG.train(); netD.train()
        totals, n = [0.0] * 3, 0
        for x in progress(loader, f"GAN epoch {epoch}/{args.epochs}"):
            x = x.to(device) * 2 - 1
            b = len(x)
            optD.zero_grad(set_to_none=True)
            fake = netG(torch.randn(b, args.noise_dim, device=device))
            d_real = F.binary_cross_entropy_with_logits(netD(x), torch.full((b,), 0.9, device=device))
            d_fake = F.binary_cross_entropy_with_logits(netD(fake.detach()), torch.zeros(b, device=device))
            finite_loss(d_real + d_fake)
            (d_real + d_fake).backward()
            optD.step()
            netD.requires_grad_(False)
            optG.zero_grad(set_to_none=True)
            g_loss = F.binary_cross_entropy_with_logits(netD(fake), torch.ones(b, device=device))
            finite_loss(g_loss)
            g_loss.backward()
            optG.step()
            netD.requires_grad_(True)
            totals = [a + loss.item() * b for a, loss in zip(totals, (d_real, d_fake, g_loss))]
            n += b
        netG.eval()
        with torch.no_grad():
            samples = (netG(fixed).cpu() + 1) / 2
        # Across-sample variability, not pooled brightness/contrast variance.
        diversity = samples.std(dim=0).mean().item()
        for key, value in zip(("d_real", "d_fake", "g_loss"), totals):
            history[key].append(value / n)
        history["sample_std"].append(diversity)
        history["epoch_seconds"].append(time.monotonic() - start)
        atomic_save(netG.state_dict(), Path(args.out) / "dcgan_G.pt")
        atomic_save(netD.state_dict(), Path(args.out) / "dcgan_D.pt")
        checkpoint_epoch(args, epoch, history, 0.0, models, opts)
        save_plots(history, args.out)
        fig, axs = plt.subplots(4, 8, figsize=(12, 7))
        for ax, img in zip(axs.flat, samples):
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1); ax.axis("off")
        fig.suptitle(f"Fixed-noise DCGAN, epoch {epoch} (uncurated)")
        fig.tight_layout()
        fig.savefig(Path(args.out) / "dcgan_samples.png", dpi=120)
        plt.close(fig)
        log(f"epoch={epoch} D_real={totals[0]/n:.4f} D_fake={totals[1]/n:.4f} "
            f"G={totals[2]/n:.4f} across_sample_std={diversity:.5f}")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
    finish(args, models, history, 0.0, device, peak, {"selection": "predeclared final epoch"})


def main():
    args = parse_args()
    device = device_for(args.device)
    vqvae = None
    if args.model == "prior":
        vqvae, config = load_model(args.vqvae_checkpoint, "vqvae", device)
        for key in ("num_codes", "latent_dim", "width", "beta", "downsample"):
            setattr(args, key, config[key])
        args.vqvae_sha256 = sha256(args.vqvae_checkpoint)
    start_run(args)
    log(f"patient-disjoint split check: {verify_patient_splits(args.data_root)}")
    try:
        if args.model == "vqvae":
            train_vqvae(args, device)
        elif args.model == "prior":
            train_prior(args, device, vqvae)
        else:
            train_dcgan(args, device)
    except Exception as exc:
        atomic_json({"state": "failed", "error": repr(exc), "updated": time.time()},
                    Path(args.out) / "status.json")
        raise


if __name__ == "__main__":
    main()