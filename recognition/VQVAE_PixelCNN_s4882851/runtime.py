"""Reproducible execution, atomic artifacts, and timestamped batch progress."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from modules import (VQVAE, CondPixelCNNPrior, GatedPixelCNNPrior, PixelCNNPrior,
                     TransformerPrior, DCGenerator)


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def progress(items, desc, every=30):
    """Tqdm on interactive terminals, plus durable log lines every 30 seconds."""
    start = last = time.monotonic()
    total = len(items)
    log(f"{desc}: starting {total} batches")
    for i, value in enumerate(tqdm(items, desc=desc, mininterval=10, leave=False), 1):
        yield value
        now = time.monotonic()
        if now - last >= every or i == total:
            elapsed = now - start
            eta = elapsed / i * (total - i)
            log(f"{desc}: {i}/{total} batches; elapsed={elapsed:.1f}s ETA={eta:.1f}s")
            last = now


def finite_loss(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"non-finite loss: {loss.detach().cpu()}")


def start_run(args):
    """Refuse accidental overwrites and record the complete run configuration."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        ignored = {"epochs", "resume", "device", "out"}
        mismatches = {k for k in old if k not in ignored and old[k] != config.get(k)}
        if mismatches:
            raise ValueError(f"resume configuration changed: {sorted(mismatches)}")
        if not args.resume:
            raise FileExistsError(f"{out} already contains a run; use --resume or a new --out")
    elif args.resume:
        raise FileNotFoundError(f"cannot resume without {path}")
    atomic_json(config, path)
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    log(f"START model={args.model} device={args.device} seed={args.seed} out={args.out}")
    atomic_json({"state": "running", "epoch": 0, "updated": time.time()}, out / "status.json")


def restore(args, models, optimizers, device):
    if not args.resume:
        return 0, {}, -math.inf if args.model == "vqvae" else math.inf
    path = Path(args.out) / "latest.pt"
    # Only locally generated, trusted checkpoints may use weights_only=False.
    saved = torch.load(path, map_location=device, weights_only=False)
    for key, model in models.items():
        model.load_state_dict(saved["models"][key])
    for key, opt in optimizers.items():
        opt.load_state_dict(saved["optimizers"][key])
    torch.set_rng_state(saved["torch_rng"].cpu())
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    if device.type == "cuda" and saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    log(f"RESUME epoch={saved['epoch']} target={args.epochs}")
    return saved["epoch"], saved["history"], saved["best"]


def checkpoint_epoch(args, epoch, history, best, models, optimizers, extra=None):
    """Epoch-boundary resume saves weights, optimiser state, history and RNG."""
    atomic_save({
        "epoch": epoch, "history": history, "best": best,
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, Path(args.out) / "latest.pt")
    atomic_json(history, Path(args.out) / "history.json")
    import csv
    with open(Path(args.out) / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    atomic_json({
        "state": "running", "epoch": epoch, "target_epochs": args.epochs,
        "updated": time.time(), "best_selection_metric": best, **(extra or {}),
    }, Path(args.out) / "status.json")


def load_model(path, kind, device):
    """Instantiate from saved config, preventing ablation shape mismatches."""
    config = json.loads((Path(path).parent / "config.json").read_text())
    if kind == "vqvae":
        model = VQVAE(width=config["width"], latent_dim=config["latent_dim"],
                      num_codes=config["num_codes"], beta=config["beta"],
                      downsample=config.get("downsample", 4))
    elif kind == "prior" and config.get("prior_type", "pixelcnn") in ("gated", "gated-cond"):
        mask_classes = 0
        if config["prior_type"] == "gated-cond":
            from dataset import MASK_CLASSES
            mask_classes = MASK_CLASSES
        model = GatedPixelCNNPrior(num_codes=config["num_codes"], width=config["prior_width"],
                                   num_layers=config["prior_layers"],
                                   kernel_size=config["prior_kernel"],
                                   embedding_dim=config["prior_embedding"],
                                   mask_classes=mask_classes)
    elif kind == "prior" and config.get("prior_type", "pixelcnn") == "cond":
        from dataset import MASK_CLASSES
        model = CondPixelCNNPrior(num_codes=config["num_codes"], width=config["prior_width"],
                                  num_blocks=config["prior_blocks"],
                                  kernel_size=config["prior_kernel"],
                                  embedding_dim=config["prior_embedding"],
                                  mask_classes=MASK_CLASSES)
    elif kind == "prior" and config.get("prior_type", "pixelcnn") == "transformer":
        model = TransformerPrior(num_codes=config["num_codes"], width=config["prior_width"],
                                 num_layers=config["prior_blocks"], num_heads=config["prior_heads"])
    elif kind == "prior":
        model = PixelCNNPrior(num_codes=config["num_codes"], width=config["prior_width"],
                              num_blocks=config["prior_blocks"],
                              kernel_size=config["prior_kernel"],
                              embedding_dim=config["prior_embedding"])
    elif kind == "dcgan":
        model = DCGenerator(noise_dim=config["noise_dim"], width=config["gan_width"])
    else:
        raise ValueError(kind)
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, config