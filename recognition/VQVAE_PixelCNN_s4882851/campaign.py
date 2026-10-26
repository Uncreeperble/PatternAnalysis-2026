"""Bounded, predeclared Slurm campaign; standard library only.

Examples (from this project directory on Rangpur):
  python campaign.py plan --root campaigns/oct26
  python campaign.py submit --manifest campaigns/oct26/manifest.json --test-only
  python campaign.py submit --manifest campaigns/oct26/manifest.json
  python campaign.py monitor --manifest campaigns/oct26/manifest.json --watch

Remote monitoring works without Conda:
  python campaign.py monitor --remote USER@HOST --manifest /absolute/remote/manifest.json --watch

Submission records every accepted job ID immediately. Never resubmit an
existing campaign blindly. --continue only fills unsubmitted tasks after a
submission interruption. GPU jobs use afterok; the CPU collector uses afterany
and records missing/failed outputs rather than fabricating success.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time


PROJECT = Path(__file__).resolve().parent
SOURCES = ("campaign.py", "slurm_job.sh", "runtime.py", "dataset.py",
           "modules.py", "train.py", "predict.py", "smoke_pipeline.py",
           "test_models.py", "test_campaign.py")


def log(text):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}", flush=True)


def save_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, **kwargs):
    return subprocess.run(list(map(str, argv)), check=True, text=True, **kwargs)


def make_plan(root, data_root):
    root = Path(root).resolve()
    ckpt, out = root / "checkpoints", root / "outputs"
    tasks = []

    def add(name, script, flags, deps=(), hours="01:30:00", gpu=True, kind="train"):
        tasks.append({"name": name, "script": script, "flags": list(map(str, flags)),
                      "deps": list(deps), "time": hours, "gpu": gpu, "kind": kind,
                      "dependency_type": "afterok" if gpu else "afterany"})

    common = ["--data-root", data_root, "--device", "cuda", "--threads", 4]
    add("gpu_gate", "smoke_pipeline.py",
        ["--data-root", data_root, "--device", "cuda", "--quick", "--out", root / "gpu_smoke"],
        hours="00:30:00", kind="gate")
    variants = [
        ("main_s0", 0, []),
        ("main_s1", 1, []),
        ("init_s0", 0, ["--codebook-init", "data"]),
        ("k128_s0", 0, ["--codebook-init", "data", "--num-codes", 128]),
        ("l1_s0", 0, ["--codebook-init", "data", "--recon-loss", "l1"]),
        ("grid8_s0", 0, ["--codebook-init", "data", "--downsample", 8]),
    ]
    for name, seed, extra in variants:
        add(f"vq_{name}", "train.py",
            [*common, "--model", "vqvae", "--epochs", 100, "--batch-size", 32,
             "--seed", seed, "--out", ckpt / f"vq_{name}", *extra], ["gpu_gate"])
    for seed in (0, 1):
        add(f"gan_s{seed}", "train.py",
            [*common, "--model", "dcgan", "--epochs", 50, "--batch-size", 32,
             "--seed", seed, "--out", ckpt / f"gan_s{seed}"], ["gpu_gate"])
    for name, seed, _ in variants:
        add(f"prior_{name}", "train.py",
            [*common, "--model", "prior", "--epochs", 40, "--batch-size", 16,
             "--seed", seed, "--vqvae-checkpoint", ckpt / f"vq_{name}" / "vqvae_best.pt",
             "--out", ckpt / f"prior_{name}"], [f"vq_{name}"])
    for name, seed, _ in variants:
        add(f"eval_{name}", "predict.py",
            [*common, "--vqvae", ckpt / f"vq_{name}" / "vqvae_best.pt",
             "--prior", ckpt / f"prior_{name}" / "prior_best.pt",
             "--dcgan", ckpt / f"gan_s{seed}" / "dcgan_G.pt",
             "--split", "test", "--n-samples", 500, "--sample-seeds", 0, 1, 2,
             "--batch-size", 32, "--temperature", 1, "--out", out / f"eval_{name}"],
            [f"prior_{name}", f"gan_s{seed}"], kind="evaluate")
    add("prior_wide7", "train.py",
        [*common, "--model", "prior", "--epochs", 40, "--batch-size", 16,
         "--prior-width", 64, "--prior-blocks", 2, "--prior-kernel", 7,
         "--vqvae-checkpoint", ckpt / "vq_main_s0" / "vqvae_best.pt",
         "--out", ckpt / "prior_wide7"], ["vq_main_s0"])
    add("eval_wide7", "predict.py",
        [*common, "--vqvae", ckpt / "vq_main_s0" / "vqvae_best.pt",
         "--prior", ckpt / "prior_wide7" / "prior_best.pt",
         "--dcgan", ckpt / "gan_s0" / "dcgan_G.pt",
         "--n-samples", 500, "--sample-seeds", 0, 1, 2, "--batch-size", 32,
         "--out", out / "eval_wide7"], ["prior_wide7", "gan_s0"], kind="evaluate")
    add("temperature_validation", "predict.py",
        [*common, "--vqvae", ckpt / "vq_main_s0" / "vqvae_best.pt",
         "--prior", ckpt / "prior_main_s0" / "prior_best.pt",
         "--split", "validate", "--n-samples", 500, "--sample-seeds", 0,
         "--batch-size", 32, "--out", out / "temperature"],
        ["prior_main_s0"], kind="temperature")
    add("collect", "campaign.py", [], [t["name"] for t in tasks],
        hours="00:15:00", gpu=False, kind="collect")
    validate_plan(tasks)
    revision = command(["git", "-C", PROJECT, "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    return {"schema": 1, "created": time.time(), "revision": revision,
            "source_sha256": {f: digest(PROJECT / f) for f in SOURCES},
            "root": str(root), "project": str(PROJECT), "data_root": str(data_root),
            "tasks": tasks, "selection_rule":
                "Highest validation SSIM among seed-0 VQ runs; test results are ablations, never selectors.",
            "budget_note": "3 h original training target is provisional, not a total-campaign guarantee. "
                           "Backups/repeats expand total cost; record actual GPU-hours and queue time."}


def validate_plan(tasks):
    seen = set()
    for task in tasks:
        name = task["name"]
        if name in seen or not set(task["deps"]) <= seen:
            raise ValueError(f"duplicate/cyclic/out-of-order task: {name}")
        if task["kind"] in ("train", "evaluate") and "--out" not in task["flags"]:
            raise ValueError(f"missing isolated output: {name}")
        if "--limit" in task["flags"] or "--skip-kid" in task["flags"]:
            raise ValueError(f"debug flags forbidden in official task: {name}")
        seen.add(name)


def batch_command(manifest_path, task, jobs, test_only=False):
    deps = [jobs[d]["job_id"] for d in task["deps"]] if not test_only else []
    cmd = ["sbatch", "--parsable", "--job-name", "hip_" + task["name"],
           "--chdir", str(PROJECT), "--partition", "comp3710" if task["gpu"] else "cpu",
           "--account", "comp3710", "--qos", "normal",
           "--cpus-per-task", "4" if task["gpu"] else "1",
           "--gres", "gpu:1" if task["gpu"] else "none", "--time", task["time"],
           "--kill-on-invalid-dep=yes"]
    if deps:
        cmd += ["--dependency", task["dependency_type"] + ":" + ":".join(deps)]
    if test_only:
        cmd.append("--test-only")
    cmd += [str(PROJECT / "slurm_job.sh"), "--manifest", str(Path(manifest_path).resolve()),
            "--task", task["name"]]
    return cmd


def submit(args):
    plan = read_json(args.manifest)
    validate_plan(plan["tasks"])
    root = Path(plan["root"]); root.mkdir(parents=True, exist_ok=True)
    (PROJECT / "logs").mkdir(exist_ok=True)
    jobs_path = root / "jobs.json"
    if jobs_path.exists() and not args.continue_submit and not args.test_only:
        raise FileExistsError("jobs.json exists; inspect IDs before using --continue")
    jobs = read_json(jobs_path) if jobs_path.exists() else {}
    for task in plan["tasks"]:
        name = task["name"]
        if name in jobs and not args.test_only:
            log(f"already submitted {name}={jobs[name]['job_id']}"); continue
        cmd = batch_command(args.manifest, task, jobs, args.test_only)
        log(shlex.join(cmd))
        result = command(cmd, capture_output=True, cwd=PROJECT)
        if args.test_only:
            log(result.stderr.strip() or result.stdout.strip() or "Slurm test-only accepted")
            continue
        raw = result.stdout.strip()
        job = raw.split(";")[0]
        if not job.isdecimal():
            raise RuntimeError(f"Unverified sbatch response {raw!r}; inspect Slurm, DO NOT blindly retry")
        jobs[name] = {"job_id": job, "submitted": time.time(), "deps": task["deps"],
                      "dependency_type": task["dependency_type"], "command": cmd}
        save_json(jobs, jobs_path)
        log(f"SUBMITTED {name}={job}")
    log(f"{'TEST-ONLY ACCEPTED' if args.test_only else 'ALL SUBMITTED'}: {len(plan['tasks'])} tasks")


def run_task(args):
    plan = read_json(args.manifest)
    changed = [f for f, sha in plan["source_sha256"].items() if digest(PROJECT / f) != sha]
    if changed:
        raise RuntimeError(f"code changed since submission: {changed}; refusing a mixed experiment")
    task = next(t for t in plan["tasks"] if t["name"] == args.task)
    status_path = Path(plan["root"]) / "task_status" / f"{args.task}.json"
    save_json({"state": "running", "started": time.time(), "job_id": os.environ.get("SLURM_JOB_ID")},
              status_path)
    log(f"TASK {args.task} revision={plan['revision']} flags={shlex.join(task['flags'])}")
    try:
        if task["kind"] == "collect":
            collect(plan)
        else:
            if task["kind"] == "gate":
                log("Running model regression tests within the compute allocation")
                command([sys.executable, "-m", "unittest", "-v", "test_models"], cwd=PROJECT)
            flags = task["flags"].copy()
            if task["kind"] == "train":
                out = Path(flags[flags.index("--out") + 1])
                if (out / "latest.pt").exists():
                    flags.append("--resume")
            if task["kind"] == "temperature":
                for temperature in (.7, 1., 1.3):
                    f = flags.copy()
                    i = f.index("--out") + 1
                    f[i] += f"_t{temperature:g}"
                    command([sys.executable, "-u", PROJECT / task["script"], *f,
                             "--temperature", str(temperature)])
            else:
                command([sys.executable, "-u", PROJECT / task["script"], *flags])
        save_json({"state": "completed", "updated": time.time()}, status_path)
        log(f"TASK COMPLETE {args.task}")
    except Exception as exc:
        save_json({"state": "failed", "updated": time.time(), "error": repr(exc)}, status_path)
        raise


def snapshot(plan):
    root = Path(plan["root"])
    jobs = read_json(root / "jobs.json")
    ids = ",".join(j["job_id"] for j in jobs.values())
    account = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ids, "--format=JobID,State,Elapsed,Start,ExitCode"],
        capture_output=True, text=True)
    states = {}
    if account.returncode == 0:
        for line in account.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 5:
                states[fields[0]] = fields[1:5]
    queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "s4882851"),
                            "-o", "%i|%T|%R"], capture_output=True, text=True)
    pending = {}
    for line in queue.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            pending[parts[0]] = parts[1:]
    records = []
    for task in plan["tasks"]:
        name = task["name"]
        job_id = jobs.get(name, {}).get("job_id", "")
        state, elapsed, start, exit_code = states.get(job_id, ["UNVERIFIED", "", "", ""])
        queue_state, reason = pending.get(job_id, ["", ""])
        if queue_state:
            state = queue_state
        path = root / "task_status" / f"{name}.json"
        status = read_json(path) if path.exists() else {}
        log_file = PROJECT / "logs" / f"hip_{name}-{job_id}.log"
        last = ""
        age = None
        if log_file.exists():
            age = time.time() - log_file.stat().st_mtime
            with log_file.open("rb") as f:
                f.seek(max(0, log_file.stat().st_size - 16000))
                lines = f.read().decode(errors="replace").splitlines()
            # Prefer actual model progress; heartbeat is liveness, not progress.
            lines = [line for line in lines if line.startswith("[") and "HEARTBEAT" not in line]
            last = lines[-1] if lines else ""
        records.append({"task": name, "job_id": job_id, "slurm_state": state,
                        "reason": reason, "elapsed": elapsed, "start": start, "exit_code": exit_code,
                        "task_state": status.get("state", "not-started"),
                        "log_age_seconds": age, "last_progress": last})
    return records


def monitor(args):
    while True:
        if args.remote:
            remote_cmd = "source /etc/profile; python3 " + shlex.quote(str(args.remote_script)) + \
                " monitor --manifest " + shlex.quote(args.manifest)
            try:
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                         args.remote, remote_cmd], text=True, timeout=35)
                if result.returncode:
                    log(f"ALERT remote monitor exit={result.returncode}")
            except (subprocess.TimeoutExpired, OSError) as exc:
                log(f"ALERT monitoring connection failed: {exc}")
        else:
            plan = read_json(args.manifest)
            records = snapshot(plan)
            log("CAMPAIGN SNAPSHOT (heartbeat != training progress)")
            for r in records:
                log(f"{r['task']:24} {r['job_id']:8} {r['slurm_state']:20} "
                    f"elapsed={r['elapsed']} {r['reason']}")
                if r["last_progress"]:
                    print("    " + r["last_progress"], flush=True)
                if r["slurm_state"] in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL") \
                        or "DependencyNeverSatisfied" in r["reason"]:
                    log(f"ALERT {r['task']} needs diagnosis; not successful")
                if r["slurm_state"] == "RUNNING" and r["log_age_seconds"] is not None \
                        and r["log_age_seconds"] > 180:
                    log(f"ALERT {r['task']} has no log activity for >180s")
            save_json(records, Path(plan["root"]) / "monitor_snapshot.json")
        if not args.watch:
            break
        log(f"Next snapshot in {args.interval}s")
        time.sleep(args.interval)


def collect(plan):
    root = Path(plan["root"])
    rows = []
    for path in sorted((root / "checkpoints").glob("*/summary.json")):
        item = read_json(path)
        rows.append({"run": path.parent.name, **item})
    evaluations = {p.parent.name: read_json(p) for p in sorted((root / "outputs").glob("*/results.json"))}
    expected = [t["name"] for t in plan["tasks"] if t["kind"] == "evaluate"]
    missing = [name for name in expected if name not in evaluations]
    status = snapshot(plan)
    failed = [r["task"] for r in status if r["task"] != "collect" and
              (r["slurm_state"] not in ("COMPLETED",) or r["task_state"] != "completed")]
    candidate = [r for r in rows if r["model"] == "vqvae" and r["seed"] == 0]
    selected = max(candidate, key=lambda r: r["best_val_ssim"])["run"] if candidate else None
    report = {"revision": plan["revision"], "training": rows, "evaluations": evaluations,
              "selected_on_validation": selected, "missing_evaluations": missing,
              "failed_or_unverified_tasks": failed, "complete": not missing and not failed,
              "slurm_accounting": status, "human_review_required": [
                  "Clinical/anatomical failure annotations by a human (automated candidates are not diagnoses)",
                  "Ethics judgment: empirical NN screening is not differential privacy",
                  "Confirm task-sheet reporting, AI disclosure, references and submission requirements"]}
    save_json(report, root / "campaign_results.json")
    with open(root / "resource_table.csv", "w", newline="") as f:
        keys = ["run", "model", "seed", "epochs", "epoch_seconds_avg", "peak_vram_gb", "best_selection_metric"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    text = ["# Campaign results", "", f"Complete: **{report['complete']}**",
            f"Validation-only chosen VQ run: `{selected}`", "",
            "| Run | Epochs | Seconds/epoch | Peak GB | Validation metric |",
            "|---|---:|---:|---:|---:|"]
    text += [f"| {r['run']} | {r['epochs']} | {r['epoch_seconds_avg']:.2f} | "
             f"{r['peak_vram_gb']:.2f} | {r['best_selection_metric']:.5f} |" for r in rows]
    text += ["", "## Held-out results",
             "| Run | Reconstruction SSIM | VQ+prior KID | GAN KID |",
             "|---|---:|---:|---:|"]
    for name, result in evaluations.items():
        kid_vq = result["vqvae_prior"].get("kid", {}).get("mean")
        kid_gan = result.get("dcgan", {}).get("kid", {}).get("mean")
        text.append(f"| {name} | {result['reconstruction']['mean_ssim']:.5f} | {kid_vq} | {kid_gan} |")
    text += ["", f"Missing evaluations: {missing}", f"Failed/unverified tasks: {failed}",
             "", "Human annotations and ethical/reliability interpretation remain mandatory."]
    (root / "RESULTS.md").write_text("\n".join(text) + "\n")
    with tarfile.open(root / "report_artifacts.tar.gz", "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv", ".png", ".md", ".txt"):
                tar.add(path, arcname=path.relative_to(root))
        jobs = read_json(root / "jobs.json")
        for name, record in jobs.items():
            path = PROJECT / "logs" / f"hip_{name}-{record['job_id']}.log"
            if path.exists():
                tar.add(path, arcname="logs/" + path.name)
    log(f"COLLECTED {root / 'report_artifacts.tar.gz'} complete={report['complete']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    plan_p = sub.add_parser("plan")
    plan_p.add_argument("--root", required=True)
    plan_p.add_argument("--data-root", default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    submit_p = sub.add_parser("submit")
    submit_p.add_argument("--manifest", required=True)
    submit_p.add_argument("--test-only", action="store_true")
    submit_p.add_argument("--continue", dest="continue_submit", action="store_true")
    run_p = sub.add_parser("run")
    run_p.add_argument("--manifest", required=True)
    run_p.add_argument("--task", required=True)
    monitor_p = sub.add_parser("monitor")
    monitor_p.add_argument("--manifest", required=True)
    monitor_p.add_argument("--watch", action="store_true")
    monitor_p.add_argument("--interval", type=int, default=30)
    monitor_p.add_argument("--remote", default="")
    monitor_p.add_argument("--remote-script", default="/home/Student/s4882851/PatternAnalysis-2026/"
                           "recognition/VQVAE_PixelCNN_s4882851/campaign.py")
    args = p.parse_args()
    if args.action == "plan":
        root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.json"
        if manifest.exists():
            raise FileExistsError("manifest already exists; do not overwrite an experiment plan")
        plan = make_plan(root, args.data_root)
        save_json(plan, manifest)
        log(f"PLAN SAVED {manifest}: {len(plan['tasks'])} tasks")
    elif args.action == "submit":
        submit(args)
    elif args.action == "run":
        run_task(args)
    else:
        monitor(args)


if __name__ == "__main__":
    main()