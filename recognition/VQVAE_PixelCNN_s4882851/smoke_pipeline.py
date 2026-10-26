"""Small real-data CLI checks, including deterministic epoch-boundary resumes.

Not final experimental evidence. Outputs must go outside Git or in ignored
checkpoints/. Use --device cuda for the dependency-gating Slurm preflight.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from runtime import atomic_json, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--skip-kid", action="store_true")
    p.add_argument("--quick", action="store_true", help="GPU gate: omit CPU bitwise-resume comparisons")
    args = p.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parent
    data = str(Path(args.data_root).resolve())
    runs = []

    def run(script, flags):
        cmd = [sys.executable, "-u", str(project / script), *map(str, flags)]
        log("SMOKE COMMAND " + shlex.join(cmd))
        start = time.monotonic()
        subprocess.run(cmd, check=True)
        runs.append({"command": cmd, "seconds": time.monotonic() - start, "exit_code": 0})

    limit = 64 if args.device == "cuda" else 8
    batch = 32 if args.device == "cuda" else 2
    base = ["--data-root", data, "--limit", limit, "--batch-size", batch,
            "--threads", 2, "--device", args.device]
    for model in ("vqvae", "dcgan", "prior"):
        extra = ["--vqvae-checkpoint", root / "vqvae" / "vqvae_best.pt"] if model == "prior" else []
        epochs = 1 if args.quick else 2
        run("train.py", [*base, "--model", model, "--epochs", epochs,
                          "--out", root / model, *extra])
        summary = json.loads((root / model / "summary.json").read_text())
        assert summary["epochs"] == epochs and summary["debug_limit"] == limit
        assert (root / model / "training_curves.png").is_file()
        if not args.quick:
            # Same data, initialization and RNG; only interruption differs.
            resumed = root / f"{model}_resumed"
            run("train.py", [*base, "--model", model, "--epochs", 1, "--out", resumed, *extra])
            run("train.py", [*base, "--model", model, "--epochs", 2, "--out", resumed, "--resume", *extra])
            a = torch.load(root / model / "latest.pt", weights_only=False, map_location="cpu")
            b = torch.load(resumed / "latest.pt", weights_only=False, map_location="cpu")
            if args.device == "cpu":
                for key in a["models"]:
                    for param in a["models"][key]:
                        torch.testing.assert_close(a["models"][key][param], b["models"][key][param],
                                                   atol=0, rtol=0)
                for key in a["history"]:
                    if key != "epoch_seconds":
                        assert a["history"][key] == b["history"][key], key
                log(f"PASS bitwise identical uninterrupted/resumed {model}")
        # Verify accidental overwrites are rejected before expensive loading.
        cmd = [sys.executable, str(project / "train.py"), *map(str, [
            *base, "--model", model, "--epochs", epochs, "--out", root / model, *extra])]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode != 0 and "already contains a run" in result.stdout

    # Backup initialization and K ablation must work before entering the queue.
    run("train.py", [*base, "--model", "vqvae", "--epochs", 1, "--num-codes", 128,
                     "--codebook-init", "data", "--out", root / "fallback"])
    run("train.py", [*base, "--model", "prior", "--epochs", 1,
                     "--prior-width", 64, "--prior-blocks", 2, "--prior-kernel", 7,
                     "--vqvae-checkpoint", root / "fallback" / "vqvae_best.pt",
                     "--out", root / "fallback_prior"])
    run("train.py", [*base, "--model", "vqvae", "--epochs", 1,
                     "--downsample", 8, "--recon-loss", "l1", "--codebook-init", "data",
                     "--out", root / "coarse"])
    run("train.py", [*base, "--model", "prior", "--epochs", 1,
                     "--vqvae-checkpoint", root / "coarse" / "vqvae_best.pt",
                     "--out", root / "coarse_prior"])
    eval_base = ["--data-root", data, "--limit", limit, "--n-samples",
                 32 if args.device == "cuda" else 4, "--sample-seeds", 0,
                 "--batch-size", 32 if args.device == "cuda" else 4,
                 "--threads", 2, "--device", args.device]
    if args.skip_kid:
        eval_base.append("--skip-kid")
    run("predict.py", [*eval_base, "--vqvae", root / "vqvae" / "vqvae_best.pt",
                       "--prior", root / "prior" / "prior_best.pt",
                       "--dcgan", root / "dcgan" / "dcgan_G.pt", "--out", root / "evaluation"])
    run("predict.py", [*eval_base, "--vqvae", root / "fallback" / "vqvae_best.pt",
                       "--prior", root / "fallback_prior" / "prior_best.pt",
                       "--temperature", .7, "--out", root / "fallback_evaluation"])
    run("predict.py", [*eval_base, "--vqvae", root / "coarse" / "vqvae_best.pt",
                       "--prior", root / "coarse_prior" / "prior_best.pt",
                       "--temperature", 1.3, "--out", root / "coarse_evaluation"])
    for folder in ("evaluation", "fallback_evaluation", "coarse_evaluation"):
        results = json.loads((root / folder / "results.json").read_text())
        assert results["complete"] and results["debug_limit"] == limit
        assert (root / folder / "interpolation.png").is_file()
        if not args.skip_kid:
            assert "kid" in results["vqvae_prior"]
    atomic_json({"complete": True, "device": args.device, "kid_tested": not args.skip_kid,
                 "resume_bitwise_tested": not args.quick and args.device == "cpu",
                 "runs": runs}, root / "smoke_summary.json")
    log(f"ALL PIPELINE CHECKS PASSED: {root / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()