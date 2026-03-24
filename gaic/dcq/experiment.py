"""
DCQ (Data Contamination Quiz) Experiment
Based on Golchin & Surdeanu (TACL 2025)

Three-phase contamination detection:
1. Generate synonym-based perturbations (Phase 1)
2. Bias Detector Quiz - find position bias (Phase 2)
3. Bias Compensator Quiz - detect contamination (Phase 3)

Usage:
    # Phase 1: Generate perturbations (run once)
    uv run gaic/dcq/experiment.py generate config/experiments/dcq/perturbator.toml

    # Phase 2: Run BDQ for specific model
    uv run gaic/dcq/experiment.py bdq config/experiments/dcq/perturbator.toml config/experiments/dcq/gpt52.toml

    # Phase 3: Run BCQ for specific model
    uv run gaic/dcq/experiment.py bcq config/experiments/dcq/perturbator.toml config/experiments/dcq/gpt52.toml
"""

import json
import math
import os
import random
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL
from tqdm import tqdm

from config.paths import GAIC_DATA_DIR
from gaic.dcq.schemas import PerturbationSet

load_dotenv()

# ============================================================================
# Prompts (from Golchin & Surdeanu TACL 2025)
# ============================================================================

PERTURBATION_PROMPT = """Your task is to create a four-choice quiz by replacing the words in the provided INPUT TEXT with their contextually relevant synonyms.
The meaning and sentence structure of the four options MUST EXACTLY match every detail in the INPUT TEXT.
You MUST NOT include the provided INPUT TEXT as an option.

You MUST make sure that:
(1) You generate DISTINCT options based on the provided INPUT TEXT;
(2) The ONLY difference between options is WORD-LEVEL PERTURBATIONS.
(3) Options are ORDERED;
(4) There is NOT any extra explanation;
(5) You follow the following FORMAT to generate options;
(6) You comply with every specific symbol and letter detail in the given INPUT TEXT; and
(7) All options retain the EXACT LABEL from the INPUT TEXT, if there is one.
---
INPUT TEXT:
{original_sentence}
---
FORMAT:

A)
First word-level perturbation goes here

B)
Second word-level perturbation goes here

C)
Third word-level perturbation goes here

D)
Fourth word-level perturbation goes here
---
"""

BDQ_PROMPT = """You are provided with a five-choice quiz. Your task is to CORRECTLY select the option that EXACTLY corresponds to an instance from the {split} split of the {dataset_name} dataset.

When selecting the option, you MUST ensure that you follow the following rules:
(1) You MUST ensure that you ONLY generate a single option letter as your answer.
(2) If you do not know the dataset or the correct answer, you MUST select option "E) None of the provided options."

Hint: While all the following options seem similar in terms of meaning, there is ONLY one option that reflects the EXACT WORDING with respect to the original instance.
---
A)
{option_a}

B)
{option_b}

C)
{option_c}

D)
{option_d}

E)
None of the provided options.
---
ANSWER:
"""

BCQ_PROMPT = """You are provided with a five-choice quiz. Your task is to CORRECTLY select the option that EXACTLY corresponds to an instance from the {split} split of the {dataset_name} dataset.

When selecting the option, you MUST ensure that you follow the following rules:
(1) You MUST ensure that you ONLY generate a single option letter as your answer.
(2) If you do not know the dataset or the correct answer, you MUST select option "E) None of the provided options."

Hint: While all the following options seem similar in terms of meaning, there is ONLY one option that reflects the EXACT WORDING with respect to the original instance.
---
A)
{option_a}

B)
{option_b}

C)
{option_c}

D)
{option_d}

E)
None of the provided options.
---
ANSWER:
"""


# ============================================================================
# Utilities
# ============================================================================


def load_config(path: Path) -> dict:
    """Load TOML configuration file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def merge_configs(base_cfg: dict, model_cfg: dict) -> dict:
    """Merge base config with model-specific config."""
    merged = base_cfg.copy()
    merged["model"] = model_cfg["model"]
    return merged


def make_client(cfg: dict) -> OpenAI:
    """Create OpenAI client for given provider."""
    provider = cfg["provider"]
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    if provider == "groq":
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    if provider == "portkey":
        return OpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key=os.environ["PORTKEY_API_KEY"],
        )
    if provider == "together_ai":
        return OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )
    if provider == "mistral":
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
        )
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


def load_dev_data(datasets: list[str]) -> dict:
    """Load dev.jsonl + dev_labels.jsonl and filter by enabled datasets."""
    # Load labels first
    labels = {}
    with open(GAIC_DATA_DIR / "dev_labels.jsonl") as f:
        for line in f:
            label_entry = json.loads(line)
            labels[label_entry["id"]] = label_entry["label"]

    # Load data and join with labels
    data = {}
    with open(GAIC_DATA_DIR / "dev.jsonl") as f:
        for line in f:
            sample = json.loads(line)
            sample_id = sample["id"]
            # Extract dataset from id (e.g., "ABSTRCT-dev-1" -> "ABSTRCT")
            dataset = sample_id.split("-")[0]

            if dataset in datasets:
                if dataset not in data:
                    data[dataset] = []
                # Normalize to expected format
                data[dataset].append({
                    "ID": sample_id,
                    "Dataset": dataset,
                    "Sentence": sample["sentence"],
                    "Label": labels.get(sample_id, "Unknown"),
                })
    return data


def sample_balanced(samples: list[dict], n: int) -> list[dict]:
    """Sample n samples with balanced Argument/No-Argument split."""
    args = [s for s in samples if s["Label"] == "Argument"]
    no_args = [s for s in samples if s["Label"] == "No-Argument"]

    per_class = n // 2
    random.seed(42)
    selected = random.sample(args, min(per_class, len(args))) + random.sample(
        no_args, min(per_class, len(no_args))
    )
    return selected


# ============================================================================
# Phase 1: Generate Perturbations
# ============================================================================


def generate_perturbations(base_cfg: dict):
    """Phase 1: Generate synonym-based perturbations for all samples."""
    print("=" * 80)
    print("DCQ Phase 1: Perturbation Generation")
    print("=" * 80)
    print(f"Generator: {base_cfg['phase1']['model']}")
    print(f"Samples per dataset: {base_cfg['sampling']['samples_per_dataset']}")
    print("-" * 80)

    output_dir = Path(base_cfg["output"]["base_dir"]) / "phase1_perturbations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets = base_cfg["datasets"]["enabled"]
    data = load_dev_data(datasets)

    # Setup client
    client = make_client(base_cfg["phase1"])
    max_retries = base_cfg["phase1"]["max_retries"]

    # Generate perturbations per dataset
    target_count = base_cfg["sampling"]["samples_per_dataset"]

    for dataset in datasets:
        if dataset not in data:
            logger.warning(f"No data found for {dataset}, skipping")
            continue

        output_file = output_dir / f"{dataset}.jsonl"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"Skipping {dataset} - {output_file} already exists")
            continue

        # Get all samples, balanced by class
        all_samples = data[dataset]
        args = [s for s in all_samples if s["Label"] == "Argument"]
        no_args = [s for s in all_samples if s["Label"] == "No-Argument"]
        random.seed(42)
        random.shuffle(args)
        random.shuffle(no_args)

        logger.info(f"Generating perturbations for {dataset}: target {target_count} samples")

        results = []
        arg_idx, no_arg_idx = 0, 0
        per_class = target_count // 2

        with tqdm(total=target_count, desc=f"  {dataset}") as pbar:
            while len(results) < target_count:
                # Pick next sample, alternating classes to maintain balance
                arg_count = sum(1 for r in results if r["label"] == "Argument")
                no_arg_count = len(results) - arg_count

                if arg_count < per_class and arg_idx < len(args):
                    sample = args[arg_idx]
                    arg_idx += 1
                elif no_arg_count < per_class and no_arg_idx < len(no_args):
                    sample = no_args[no_arg_idx]
                    no_arg_idx += 1
                elif arg_idx < len(args):
                    sample = args[arg_idx]
                    arg_idx += 1
                elif no_arg_idx < len(no_args):
                    sample = no_args[no_arg_idx]
                    no_arg_idx += 1
                else:
                    logger.warning(f"Exhausted samples for {dataset}, got {len(results)}/{target_count}")
                    break

                sentence = sample["Sentence"]
                success = False

                # Retry loop with validation
                for attempt in range(max_retries):
                    try:
                        resp = client.beta.chat.completions.parse(
                            model=base_cfg["phase1"]["model"],
                            messages=[
                                {"role": "user", "content": PERTURBATION_PROMPT.format(original_sentence=sentence)}
                            ],
                            temperature=base_cfg["phase1"]["temperature"],
                            response_format=PerturbationSet,
                        )

                        parsed = resp.choices[0].message.parsed
                        if parsed is None:
                            raise ValueError("No parsed output")

                        # Validate: all perturbations must be distinct and different from original
                        perturbations = [
                            parsed.perturbation_1,
                            parsed.perturbation_2,
                            parsed.perturbation_3,
                            parsed.perturbation_4,
                        ]

                        if len(set(perturbations)) != 4:
                            raise ValueError("Perturbations are not all distinct")

                        if sentence in perturbations:
                            raise ValueError("Original sentence found in perturbations")

                        # Success - save result
                        results.append({
                            "id": sample["ID"],
                            "dataset": dataset,
                            "label": sample["Label"],
                            "original": sentence,
                            "perturbations": perturbations,
                        })
                        success = True
                        pbar.update(1)
                        time.sleep(0.5)  # Rate limit
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Retry {attempt + 1}/{max_retries} for {sample['ID']}: {e}")
                            time.sleep(2 ** attempt)
                        else:
                            logger.warning(f"Skipping {sample['ID']}, trying next sample: {e}")

        # Save results as JSONL per dataset
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"  Saved {len(results)} perturbations to {output_file}")

    print("=" * 80)
    print("Phase 1 Complete.")
    print("=" * 80)


# ============================================================================
# Phase 2: Bias Detector Quiz (BDQ)
# ============================================================================


def run_bdq(base_cfg: dict, model_cfg: dict):
    """Phase 2: Run Bias Detector Quiz to find position bias."""
    cfg = merge_configs(base_cfg, model_cfg)
    model_name = cfg["model"]["model"].replace("/", "_")

    print("=" * 80)
    print("DCQ Phase 2: Bias Detector Quiz")
    print("=" * 80)
    print(f"Model: {cfg['model']['model']}")
    print(f"Provider: {cfg['model']['provider']}")
    print("-" * 80)

    # Setup paths
    base_dir = Path(cfg["output"]["base_dir"])
    perturbations_dir = base_dir / "phase1_perturbations"
    output_dir = base_dir / "phase2_bdq" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "bdq_results.jsonl"
    summary_file = output_dir / "bias_summary.json"

    # Load perturbation files from phase1_perturbations/
    # Optionally sample a subset for BDQ (faster bias detection)
    bdq_samples_per_dataset = cfg["phase2"].get("samples_per_dataset", None)

    perturbations = []
    for jsonl_file in sorted(perturbations_dir.glob("*.jsonl")):
        dataset_samples = []
        with open(jsonl_file) as f:
            for line in f:
                dataset_samples.append(json.loads(line))

        # Sample if limit specified
        if bdq_samples_per_dataset and len(dataset_samples) > bdq_samples_per_dataset:
            random.seed(42)
            dataset_samples = random.sample(dataset_samples, bdq_samples_per_dataset)
            logger.info(f"Loaded {jsonl_file.name}: {bdq_samples_per_dataset} samples (sampled)")
        else:
            logger.info(f"Loaded {jsonl_file.name}: {len(dataset_samples)} samples")

        perturbations.extend(dataset_samples)

    logger.info(f"Total: {len(perturbations)} perturbation sets")

    # Setup client
    client = make_client(cfg["model"])

    # Run BDQ
    position_counts = Counter()
    results = []

    # Group by dataset for progress tracking
    by_dataset = {}
    for p in perturbations:
        if p["dataset"] not in by_dataset:
            by_dataset[p["dataset"]] = []
        by_dataset[p["dataset"]].append(p)

    for dataset in sorted(by_dataset.keys()):
        samples = by_dataset[dataset]
        with tqdm(total=len(samples), desc=f"  {dataset:12s}") as pbar:
            for sample in samples:
                # Create quiz with ONLY perturbations (no original)
                options = sample["perturbations"]
                random.seed(42)  # Deterministic shuffle
                random.shuffle(options)

                prompt = BDQ_PROMPT.format(
                    split=cfg["sampling"]["split"],
                    dataset_name=sample["dataset"],
                    option_a=options[0],
                    option_b=options[1],
                    option_c=options[2],
                    option_d=options[3],
                )

                # Get answer (raw text, as per original DCQ implementation)
                try:
                    # OpenAI uses max_completion_tokens, others use max_tokens
                    token_param = "max_completion_tokens" if cfg["model"]["provider"] == "openai" else "max_tokens"
                    resp = client.chat.completions.create(
                        model=cfg["model"]["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=cfg["phase2"]["temperature"],
                        **{token_param: cfg["phase2"]["max_tokens"]},
                    )

                    raw_answer = (resp.choices[0].message.content or "").strip().upper()
                    # Extract single letter A-E
                    answer = raw_answer[0] if raw_answer and raw_answer[0] in "ABCDE" else "E"

                except Exception as e:
                    logger.error(f"Error for {sample['id']}: {e}")
                    answer = "E"

                time.sleep(0.5)  # Rate limit

                position_counts[answer] += 1

                results.append({
                    "id": sample["id"],
                    "dataset": sample["dataset"],
                    "answer": answer,
                    "options": options,
                })

                pbar.update(1)

    # Save results
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Compute bias summary using original DCQ threshold: ceil(n/5) = random chance
    total = sum(position_counts.values())
    random_chance_count = math.ceil(total / 5)  # Threshold from original paper

    # Non-preferred = positions selected LESS THAN random chance, excluding E
    non_preferred = [
        pos
        for pos in ["A", "B", "C", "D"]
        if position_counts.get(pos, 0) < random_chance_count
    ]

    # Fallback: if no bias detected, use all positions (as per original implementation)
    if not non_preferred:
        non_preferred = ["A", "B", "C", "D"]
        logger.info("No position bias detected, using all positions A-D")

    bias_summary = {
        "model": cfg["model"]["model"],
        "total_samples": total,
        "random_chance_threshold": random_chance_count,
        "position_frequencies": {
            pos: {"count": count, "percentage": count / total * 100}
            for pos, count in position_counts.items()
        },
        "non_preferred_positions": non_preferred,
    }

    with open(summary_file, "w") as f:
        json.dump(bias_summary, f, indent=2)

    # Print summary
    print()
    print("Position Bias Summary:")
    for pos in ["A", "B", "C", "D", "E"]:
        count = position_counts.get(pos, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {pos}: {pct:5.1f}% ({count:4d})")

    print()
    print(f"Non-preferred positions (< {cfg['phase3']['bias_threshold'] * 100}%):")
    print(f"  {', '.join(bias_summary['non_preferred_positions'])}")

    print("=" * 80)
    print("Phase 2 Complete. Results saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")
    print("=" * 80)


# ============================================================================
# Phase 3: Bias Compensator Quiz (BCQ)
# ============================================================================


def run_bcq(base_cfg: dict, model_cfg: dict):
    """Phase 3: Run Bias Compensator Quiz to detect contamination."""
    cfg = merge_configs(base_cfg, model_cfg)
    model_name = cfg["model"]["model"].replace("/", "_")

    print("=" * 80)
    print("DCQ Phase 3: Bias Compensator Quiz")
    print("=" * 80)
    print(f"Model: {cfg['model']['model']}")
    print(f"Provider: {cfg['model']['provider']}")
    print("-" * 80)

    # Setup paths
    base_dir = Path(cfg["output"]["base_dir"])
    perturbations_dir = base_dir / "phase1_perturbations"
    bias_summary_file = base_dir / "phase2_bdq" / model_name / "bias_summary.json"

    output_dir = base_dir / "phase3_bcq" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(bias_summary_file) as f:
        bias_summary = json.load(f)

    non_preferred = bias_summary["non_preferred_positions"]
    bdq_position_counts = {
        pos: bias_summary["position_frequencies"].get(pos, {}).get("count", 0)
        for pos in ["A", "B", "C", "D", "E"]
    }

    logger.info(f"Non-preferred positions: {non_preferred}")

    # Setup client
    client = make_client(cfg["model"])

    # Process each dataset separately
    for jsonl_file in sorted(perturbations_dir.glob("*.jsonl")):
        dataset = jsonl_file.stem  # e.g., "ABSTRCT" from "ABSTRCT.jsonl"

        results_file = output_dir / f"{dataset}_bcq_results.jsonl"
        report_file = output_dir / f"{dataset}_contamination_report.json"

        # Skip if already exists
        if report_file.exists():
            logger.info(f"Skipping {dataset} - {report_file} already exists")
            continue

        # Load perturbations for this dataset
        samples = []
        with open(jsonl_file) as f:
            for line in f:
                samples.append(json.loads(line))

        logger.info(f"Processing {dataset}: {len(samples)} samples")

        # Run BCQ for this dataset
        results = []
        bcq_per_position = {pos: 0 for pos in non_preferred}
        dataset_correct = 0

        with tqdm(total=len(samples), desc=f"  {dataset:12s}") as pbar:
            for sample in samples:
                # For each non-preferred position, create BCQ
                position_results = []

                for position in non_preferred:
                    # Insert original at this position
                    pos_idx = ord(position) - ord("A")

                    if pos_idx >= 4:  # Position E - skip
                        continue

                    # Build options: perturbations with original at pos_idx
                    options = sample["perturbations"].copy()
                    options[pos_idx] = sample["original"]

                    prompt = BCQ_PROMPT.format(
                        split=cfg["sampling"]["split"],
                        dataset_name=sample["dataset"],
                        option_a=options[0],
                        option_b=options[1],
                        option_c=options[2],
                        option_d=options[3],
                    )

                    # Get answer (raw text, as per original DCQ implementation)
                    try:
                        # OpenAI uses max_completion_tokens, others use max_tokens
                        token_param = "max_completion_tokens" if cfg["model"]["provider"] == "openai" else "max_tokens"
                        resp = client.chat.completions.create(
                            model=cfg["model"]["model"],
                            messages=[{"role": "user", "content": prompt}],
                            temperature=cfg["phase3"]["temperature"],
                            **{token_param: cfg["phase3"]["max_tokens"]},
                        )

                        raw_answer = (resp.choices[0].message.content or "").strip().upper()
                        answer = raw_answer[0] if raw_answer and raw_answer[0] in "ABCDE" else "E"

                    except Exception as e:
                        logger.error(f"Error for {sample['id']} at position {position}: {e}")
                        answer = "E"

                    time.sleep(0.5)  # Rate limit

                    is_correct = answer == position
                    position_results.append({
                        "position": position,
                        "answer": answer,
                        "correct": is_correct,
                    })

                    # Track per-position stats for this dataset
                    if is_correct:
                        bcq_per_position[position] += 1

                # Take MAX accuracy across positions (as per paper)
                sample_correct = any(r["correct"] for r in position_results)
                if sample_correct:
                    dataset_correct += 1

                results.append({
                    "id": sample["id"],
                    "dataset": sample["dataset"],
                    "label": sample["label"],
                    "position_results": position_results,
                    "correct": sample_correct,
                })

                pbar.update(1)

        # Save results for this dataset
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        # Compute min/max contamination for this dataset using Cohen's κ
        total = len(results)

        # Build triples: (position, bcq_correct_count, bdq_bias_count)
        # Scale BDQ bias proportionally to this dataset's sample count
        num_datasets = len(list(perturbations_dir.glob("*.jsonl")))
        triples = [
            (pos, bcq_per_position[pos], bdq_position_counts.get(pos, 0) // num_datasets)
            for pos in non_preferred
        ]
        # Sort by BCQ correct count (descending), then by BDQ bias (ascending)
        triples.sort(key=lambda x: (-x[1], x[2]))

        max_count = triples[0][1]
        max_cont_positional_bias = triples[0][2]
        max_cont_level = max_count / total if total > 0 else 0

        # Cohen's Kappa for min contamination
        if total > max_cont_positional_bias:
            kappa = (max_count - max_cont_positional_bias) / (total - max_cont_positional_bias)
        else:
            kappa = 0

        # Min contamination = max(kappa, second_best_position_rate)
        if len(triples) > 1:
            second_max_count = triples[1][1]
            min_cont_level = max(kappa, second_max_count / total if total > 0 else 0)
        else:
            min_cont_level = kappa

        report = {
            "model": cfg["model"]["model"],
            "dataset": dataset,
            "non_preferred_positions": non_preferred,
            "bcq_per_position": bcq_per_position,
            "total": total,
            "correct": dataset_correct,
            "max_contamination": max_cont_level * 100,
            "min_contamination": min_cont_level * 100,
            "contamination_range": f"[{min_cont_level * 100:.1f}%, {max_cont_level * 100:.1f}%]",
        }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"    {dataset}: [{min_cont_level * 100:.1f}%, {max_cont_level * 100:.1f}%] (n={total})")

    print()
    print("=" * 80)
    print("Phase 3 Complete. Per-dataset results saved to:")
    print(f"  {output_dir}/{{DATASET}}_bcq_results.jsonl")
    print(f"  {output_dir}/{{DATASET}}_contamination_report.json")
    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    base_config_path = Path(sys.argv[2])

    base_cfg = load_config(base_config_path)

    if command == "generate":
        generate_perturbations(base_cfg)

    elif command == "bdq":
        if len(sys.argv) < 4:
            print("ERROR: bdq requires model config")
            print("Usage: uv run gaic/dcq/experiment.py bdq <base_config> <model_config>")
            sys.exit(1)

        model_config_path = Path(sys.argv[3])
        model_cfg = load_config(model_config_path)
        run_bdq(base_cfg, model_cfg)

    elif command == "bcq":
        if len(sys.argv) < 4:
            print("ERROR: bcq requires model config")
            print("Usage: uv run gaic/dcq/experiment.py bcq <base_config> <model_config>")
            sys.exit(1)

        model_config_path = Path(sys.argv[3])
        model_cfg = load_config(model_config_path)
        run_bcq(base_cfg, model_cfg)

    else:
        print(f"ERROR: Unknown command '{command}'")
        print("Available commands: generate, bdq, bcq")
        sys.exit(1)


if __name__ == "__main__":
    main()
