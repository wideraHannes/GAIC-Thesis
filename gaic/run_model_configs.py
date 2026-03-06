#!/usr/bin/env python3
"""Run experiment configs for models."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
V1_CONFIG_DIR = PROJECT_ROOT / "config" / "experiments" / "v1"


def get_all_models() -> list[str]:
    """Get all model directories under v1 config."""
    return sorted([d.name for d in V1_CONFIG_DIR.iterdir() if d.is_dir()])


def run_config(model: str, config_name: str) -> bool:
    """Run a single config. Returns True on success."""
    config_path = V1_CONFIG_DIR / model / f"{config_name}.toml"
    experiment_script = PROJECT_ROOT / "gaic" / "unified_experiment.py"

    if not config_path.exists():
        print(f"Warning: Config not found, skipping: {config_path}")
        return True

    print(f"\n{'='*60}")
    print(f"Running {config_name} for {model}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        ["uv", "run", str(experiment_script), str(config_path)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"Error: {config_name} for {model} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment configs for models"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model directory name (e.g., mistral_small_24b). Required unless --all-models is used.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run across all models in v1 config directory",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["c0", "c1", "c2", "c3"],
        help="Configs to run (default: c0 c1 c2 c3)",
    )
    args = parser.parse_args()

    if not args.all_models and not args.model:
        parser.error("Either specify a model or use --all-models")

    if args.all_models:
        models = get_all_models()
        print(f"Running configs {args.configs} for all models: {models}")
    else:
        if not (V1_CONFIG_DIR / args.model).exists():
            print(f"Error: Model directory not found: {V1_CONFIG_DIR / args.model}")
            sys.exit(1)
        models = [args.model]

    for model in models:
        for config_name in args.configs:
            if not run_config(model, config_name):
                sys.exit(1)

    print(f"\n{'='*60}")
    print(f"All runs completed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
