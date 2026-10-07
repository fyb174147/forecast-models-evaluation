#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_conda_executable() -> str:
    """Return the conda executable to use for environment execution."""
    candidates = [
        shutil.which("conda"),
        shutil.which("micromamba"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate

    win_candidates = [
        Path.home() / "miniconda3" / "condabin" / "conda.bat",
        Path.home() / "anaconda3" / "condabin" / "conda.bat",
        Path("C:/Users/thefu/miniconda3/condabin/conda.bat"),
    ]
    for candidate in win_candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Conda was not found on PATH. Install Miniconda/Anaconda or add conda to PATH."
    )


def build_model_specs(dataset: str, sampling_rates: str, runs: int, gpu: int, results_dir: str):
    sampling_parts = sampling_rates.split() if isinstance(sampling_rates, str) else [str(sampling_rates)]
    specs = [
        {
            "name": "Naive Drift",
            "env": "naivedrift",
            "script": "models/model_naivedrift.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "Mamba",
            "env": "mamba",
            "script": "models/model_mamba.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "StatsML",
            "env": "statsml",
            "script": "models/models_statsml.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "GluonTS",
            "env": "gluonts",
            "script": "models/models_gluonts.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "NeuralForecast",
            "env": "neuralforecast",
            "script": "models/models_neuralforecast.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "TimeGPT",
            "env": "timegpt",
            "script": "models/fmodel_timegpt.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--model", "timegpt-1-long-horizon",
                "--confidence_level", "95",
                "--finetune_steps", "0",
                "--finetune_depth", "1",
                "--gpu", str(gpu),
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "TimesFM",
            "env": "timesfm",
            "script": "models/fmodel_timesfm.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--model_id", "google/timesfm-1.0-200m-pytorch",
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "MOIRAI",
            "env": "moirai",
            "script": "models/fmodel_moirai.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--model", "moirai",
                "--size", "small",
                "--patch_size", "auto",
                "--batch_size", "32",
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "Chronos",
            "env": "chronos",
            "script": "models/fmodel_chronos.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--model_id", "amazon/chronos-bolt-small",
                "--results_dir", results_dir,
            ],
        },
        {
            "name": "Timer-XL",
            "env": "timerxl",
            "script": "models/fmodel_timerxl.py",
            "args": [
                "--dataset", dataset,
                "--sampling_rates", *sampling_parts,
                "--runs", str(runs),
                "--gpu", str(gpu),
                "--model_id", "thuml/timer-base-84m",
                "--save_folder", "TimerXL",
                "--results_dir", results_dir,
            ],
        },
    ]
    return specs


def run_model(spec, repo_root: Path, conda_executable: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec['name'].lower().replace(' ', '_').replace('-', '_')}.log"
    command = [conda_executable, "run", "-n", spec["env"], "python", spec["script"], *spec["args"]]

    print(f"\n===== Running: {spec['name']} ({spec['env']}) =====")
    print("Command:")
    print(" ".join(command))
    print(f"Log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(command, cwd=str(repo_root), stdout=log_file, stderr=subprocess.STDOUT)

    if process.returncode == 0:
        print(f"SUCCESS: {spec['name']}")
    else:
        print(f"FAILED: {spec['name']} with exit code {process.returncode}")
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all forecasting models in this repo sequentially.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium")
    parser.add_argument("--sampling_rates", default="100", help="Sampling percentages as a single string, e.g. '100' or '25 50 100'.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index for supported local models; use -1 for CPU.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--skip", nargs="*", default=[], help="Model names to skip, e.g. --skip TimeGPT Chronos")
    parser.add_argument("--list", action="store_true", help="List available models and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with the next model even if one fails.")
    args = parser.parse_args()

    specs = build_model_specs(args.dataset, args.sampling_rates, args.runs, args.gpu, args.results_dir)
    if args.list:
        print("Available models:")
        for spec in specs:
            print(f"- {spec['name']} ({spec['env']})")
        return 0

    normalize = lambda s: s.strip().lower()
    skip_set = {normalize(name) for name in args.skip}
    conda_executable = find_conda_executable()
    log_dir = ROOT / "logs" / "all_models"

    failed = []
    for spec in specs:
        model_key = normalize(spec["name"])
        if model_key in skip_set:
            print(f"Skipping: {spec['name']}")
            continue

        if args.dry_run:
            command = [conda_executable, "run", "-n", spec["env"], "python", spec["script"], *spec["args"]]
            print(f"\n{spec['name']} -> {' '.join(command)}")
            continue

        result = run_model(spec, ROOT, conda_executable, log_dir)
        if result != 0:
            failed.append(spec["name"])
            if not args.continue_on_error:
                print(f"Stopping because {spec['name']} failed.")
                return result

    if failed:
        print("\nSummary: some models failed:")
        for name in failed:
            print(f"- {name}")
        return 1

    print("\nAll enabled models finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
