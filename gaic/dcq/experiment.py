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
from gaic.dcq.schemas import PerturbationSet, QuizAnswer

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
    """Load dev.jsonl and filter by enabled datasets."""
    data = {}
    with open(GAIC_DATA_DIR / "dev.jsonl") as f:
        for line in f:
            sample = json.loads(line)
            dataset = sample["Dataset"]
            if dataset in datasets:
                if dataset not in data:
                    data[dataset] = []
                data[dataset].append(sample)
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
    output_file = output_dir / base_cfg["phase1"]["output_file"]

    # Load data
    datasets = base_cfg["datasets"]["enabled"]
    data = load_dev_data(datasets)

    # Sample balanced
    all_samples = []
    for dataset, samples in data.items():
        selected = sample_balanced(samples, base_cfg["sampling"]["samples_per_dataset"])
        all_samples.extend([(dataset, s) for s in selected])

    logger.info(f"Total samples to perturb: {len(all_samples)}")

    # Setup client
    client = make_client(base_cfg["phase1"])
    max_retries = base_cfg["phase1"]["max_retries"]

    # Generate perturbations
    results = []
    with tqdm(total=len(all_samples), desc="Generating perturbations") as pbar:
        for dataset, sample in all_samples:
            sentence = sample["Sentence"]

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
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {sample['ID']}: {e}")
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to generate perturbations for {sample['ID']} after {max_retries} attempts")
                        raise

            pbar.update(1)

    # Save results as JSONL
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print("=" * 80)
    print(f"Phase 1 Complete. Saved {len(results)} perturbation sets to:")
    print(f"  {output_file}")
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
    input_file = base_dir / "phase1_perturbations" / cfg["phase1"]["output_file"]
    output_dir = base_dir / "phase2_bdq" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "bdq_results.jsonl"
    summary_file = output_dir / "bias_summary.json"

    # Load perturbations
    perturbations = []
    with open(input_file) as f:
        for line in f:
            perturbations.append(json.loads(line))

    logger.info(f"Loaded {len(perturbations)} perturbation sets")

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

                # Get answer
                try:
                    resp = client.beta.chat.completions.parse(
                        model=cfg["model"]["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=cfg["phase2"]["temperature"],
                        max_tokens=cfg["phase2"]["max_tokens"],
                        response_format=QuizAnswer,
                    )

                    parsed = resp.choices[0].message.parsed
                    answer = parsed.answer if parsed else "E"

                except Exception as e:
                    logger.error(f"Error for {sample['id']}: {e}")
                    answer = "E"

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
    perturbations_file = base_dir / "phase1_perturbations" / cfg["phase1"]["output_file"]
    bias_summary_file = base_dir / "phase2_bdq" / model_name / "bias_summary.json"

    output_dir = base_dir / "phase3_bcq" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "bcq_results.jsonl"
    report_file = output_dir / "contamination_report.json"

    # Load inputs
    perturbations = []
    with open(perturbations_file) as f:
        for line in f:
            perturbations.append(json.loads(line))

    with open(bias_summary_file) as f:
        bias_summary = json.load(f)

    non_preferred = bias_summary["non_preferred_positions"]
    bdq_position_counts = {
        pos: bias_summary["position_frequencies"].get(pos, {}).get("count", 0)
        for pos in ["A", "B", "C", "D", "E"]
    }

    logger.info(f"Non-preferred positions: {non_preferred}")
    logger.info(f"Testing {len(perturbations)} samples")

    # Setup client
    client = make_client(cfg["model"])

    # Run BCQ
    results = []
    by_dataset = {}

    # Group by dataset
    for p in perturbations:
        if p["dataset"] not in by_dataset:
            by_dataset[p["dataset"]] = []
        by_dataset[p["dataset"]].append(p)

    for dataset in sorted(by_dataset.keys()):
        samples = by_dataset[dataset]
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

                    # Get answer
                    try:
                        resp = client.beta.chat.completions.parse(
                            model=cfg["model"]["model"],
                            messages=[{"role": "user", "content": prompt}],
                            temperature=cfg["phase3"]["temperature"],
                            max_tokens=cfg["phase3"]["max_tokens"],
                            response_format=QuizAnswer,
                        )

                        parsed = resp.choices[0].message.parsed
                        answer = parsed.answer if parsed else "E"

                    except Exception as e:
                        logger.error(f"Error for {sample['id']} at position {position}: {e}")
                        answer = "E"

                    is_correct = answer == position
                    position_results.append({
                        "position": position,
                        "answer": answer,
                        "correct": is_correct,
                    })

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

        contamination_rate = dataset_correct / len(samples) * 100
        print(f"    Contamination: {contamination_rate:.1f}% ({dataset_correct}/{len(samples)})")

    # Save results
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Compute per-position BCQ statistics (as per original DCQ implementation)
    bcq_per_position = {pos: 0 for pos in non_preferred}
    for result in results:
        for pr in result["position_results"]:
            if pr["correct"]:
                bcq_per_position[pr["position"]] += 1

    # Compute max/min contamination following original paper methodology
    overall_total = len(results)

    # Build triples: (position, bcq_correct_count, bdq_bias_count)
    triples = [
        (pos, bcq_per_position[pos], bdq_position_counts.get(pos, 0))
        for pos in non_preferred
    ]
    # Sort by BCQ correct count (descending), then by BDQ bias (ascending)
    triples.sort(key=lambda x: (-x[1], x[2]))

    max_count = triples[0][1]
    max_cont_positional_bias = triples[0][2]
    max_cont_level = max_count / overall_total if overall_total > 0 else 0

    # Cohen's Kappa for min contamination
    if overall_total > max_cont_positional_bias:
        kappa = (max_count - max_cont_positional_bias) / (overall_total - max_cont_positional_bias)
    else:
        kappa = 0

    # Min contamination = max(kappa, second_best_position_rate)
    if len(triples) > 1:
        second_max_count = triples[1][1]
        min_cont_level = max(kappa, second_max_count / overall_total if overall_total > 0 else 0)
    else:
        min_cont_level = kappa

    # Per-dataset stats
    by_dataset_stats = {}
    for dataset in by_dataset.keys():
        dataset_results = [r for r in results if r["dataset"] == dataset]
        correct = sum(1 for r in dataset_results if r["correct"])
        total = len(dataset_results)

        by_dataset_stats[dataset] = {
            "correct": correct,
            "total": total,
            "contamination_rate": correct / total * 100 if total > 0 else 0,
        }

    report = {
        "model": cfg["model"]["model"],
        "non_preferred_positions": non_preferred,
        "bcq_per_position": bcq_per_position,
        "overall": {
            "total": overall_total,
            "max_contamination": max_cont_level * 100,
            "min_contamination": min_cont_level * 100,
            "contamination_range": f"[{min_cont_level * 100:.1f}%, {max_cont_level * 100:.1f}%]",
        },
        "by_dataset": by_dataset_stats,
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 80)
    print(f"CONTAMINATION RANGE: [{min_cont_level * 100:.1f}%, {max_cont_level * 100:.1f}%]")
    print(f"  (n={overall_total} samples)")
    print("=" * 80)
    print("Phase 3 Complete. Results saved to:")
    print(f"  {results_file}")
    print(f"  {report_file}")
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
