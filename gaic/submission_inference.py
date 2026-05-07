"""
Submission inference script for GAIC shared task.

Runs inference on test.jsonl (or dev.jsonl for validation) and produces submission files.

Usage:
    uv run gaic/submission_inference.py --config config/submission/gpt5.2_dynamic.toml
    uv run gaic/submission_inference.py --model <model> --provider <provider> --context-strategy dynamic
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from sklearn.metrics import classification_report
from tqdm import tqdm

from config.paths import CONTEXT_DIR, GAIC_DATA_DIR
from gaic.unified_experiment import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    _load_dataset_json,
    assemble_context,
    classify,
    dataset_from_id,
    load_config,
    load_context,
    load_document_context,
    make_client,
)

if TYPE_CHECKING:
    from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFINITION_INHERITANCE = {"TAPE": "TACO", "TAUS": "TACO"}

METRICS_HEADER = f"{'Dataset':<12} {'N':>6} {'Macro-F1':>10} {'Acc':>8} {'Arg-F1':>8} {'NoArg-F1':>10}"
METRICS_SEP = "-" * 60


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Configuration for submission inference."""

    model: str
    provider: str
    temperature: float = 0.0
    context_strategy: str = "dynamic"  # c0, c1, dynamic
    input_file: str = "test.jsonl"
    output_dir: Path = field(default_factory=lambda: Path("submissions"))
    reasoning: bool = False
    datasets: list[str] | None = None  # None = all datasets
    max_workers: int = 8


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset or overall."""

    n_samples: int
    macro_f1: float
    accuracy: float
    argument_f1: float
    no_argument_f1: float

    def format_row(self, name: str) -> str:
        return (
            f"{name:<12} {self.n_samples:>6} "
            f"{self.macro_f1:>10.4f} {self.accuracy:>8.4f} "
            f"{self.argument_f1:>8.4f} {self.no_argument_f1:>10.4f}"
        )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(input_file: str) -> list[dict]:
    """Load samples from JSONL file, adding dataset field."""
    samples = []
    with open(GAIC_DATA_DIR / input_file) as f:
        for line in f:
            item = json.loads(line)
            item["dataset"] = dataset_from_id(item["id"])
            samples.append(item)
    return samples


def load_labels(input_file: str) -> dict[str, str] | None:
    """Load labels if available (e.g., dev.jsonl -> dev_labels.jsonl)."""
    base = input_file.replace(".jsonl", "")
    labels_file = GAIC_DATA_DIR / f"{base}_labels.jsonl"

    if not labels_file.exists():
        return None

    labels = {}
    with open(labels_file) as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]
    return labels


def load_completed_ids(log_path: Path) -> set[str]:
    """Load IDs of already completed samples from log file (for checkpointing)."""
    completed = set()
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    completed.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ---------------------------------------------------------------------------
# Context Handling
# ---------------------------------------------------------------------------


def get_context_sources(dataset: str, strategy: str) -> list[str]:
    """Determine context sources based on strategy and dataset capabilities."""
    if strategy == "c0":
        return []

    if strategy == "c1":
        return ["definition", "guideline", "document_context"]

    # dynamic: include only what's available
    capabilities = _load_dataset_json(dataset).get("capabilities", {})
    sources = []

    if capabilities.get("has_definition", False):
        sources.append("definition")
    elif dataset in DEFINITION_INHERITANCE:
        parent_caps = _load_dataset_json(DEFINITION_INHERITANCE[dataset]).get("capabilities", {})
        if parent_caps.get("has_definition", False):
            sources.append("definition")

    if capabilities.get("has_guidelines", False):
        sources.append("guideline")
    if capabilities.get("has_document_context", False):
        sources.append("document_context")

    return sources


def load_context_with_inheritance(dataset: str, sources: list[str]) -> dict[str, str]:
    """Load context, handling definition inheritance for TAPE/TAUS."""
    result = {}

    if "definition" in sources and dataset in DEFINITION_INHERITANCE:
        source_dataset = DEFINITION_INHERITANCE[dataset]
        def_path = CONTEXT_DIR / source_dataset / "definition.md"
        if def_path.exists():
            content = def_path.read_text().strip()
            if content:
                result["definition"] = content
        sources = [s for s in sources if s != "definition"]

    result.update(load_context(dataset, sources))
    return result


# ---------------------------------------------------------------------------
# Sample Processing
# ---------------------------------------------------------------------------


def process_sample(
    sample: dict,
    dataset: str,
    context_parts: dict[str, str],
    use_doc_context: bool,
    client: OpenAI,
    client_cfg: dict,
    context_strategy: str,
) -> dict:
    """Process a single sample: load context, classify, return result."""
    doc_context = ""
    if use_doc_context:
        doc_context = load_document_context(dataset, sample["id"])

    full_context = assemble_context(context_parts, doc_context)
    system_prompt = SYSTEM_PROMPT.format(context=full_context)
    user_prompt = USER_PROMPT.format(sentence=sample["sentence"])

    result = classify(client, client_cfg, sample["sentence"], full_context)

    # Build complete list of context sources used
    sources_used = list(context_parts.keys())
    if doc_context:
        sources_used.append("document_context")

    return {
        "submission": {"id": sample["id"], "label": result["label"]},
        "log": {
            "id": sample["id"],
            "dataset": dataset,
            "sentence": sample["sentence"],
            "label": result["label"],
            "reason": result.get("reason", ""),
            "context_strategy": context_strategy,
            "context_sources_used": sources_used,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    }


def process_dataset(
    dataset: str,
    samples: list[dict],
    config: InferenceConfig,
    client: OpenAI,
    client_cfg: dict,
    labels: dict[str, str] | None,
    f_sub,
    f_log,
) -> None:
    """Process all samples for a dataset with parallel API calls."""
    sources = get_context_sources(dataset, config.context_strategy)
    context_parts = load_context_with_inheritance(dataset, sources)
    use_doc_context = "document_context" in sources

    logger.info(f"{dataset}: {len(samples)} samples, context: {list(context_parts.keys())}")

    dataset_preds = {}
    results = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                process_sample, sample, dataset, context_parts, use_doc_context,
                client, client_cfg, config.context_strategy
            ): sample
            for sample in samples
        }

        for future in tqdm(as_completed(futures), total=len(samples), desc=dataset, leave=False):
            result = future.result()
            results.append(result)

    # Write results in original order
    sample_order = {s["id"]: i for i, s in enumerate(samples)}
    results.sort(key=lambda r: sample_order[r["submission"]["id"]])

    for result in results:
        f_sub.write(json.dumps(result["submission"]) + "\n")
        f_log.write(json.dumps(result["log"]) + "\n")
        dataset_preds[result["submission"]["id"]] = result["submission"]["label"]

    f_sub.flush()
    f_log.flush()

    if labels:
        metrics = compute_metrics(dataset_preds, labels, dataset)
        if metrics:
            logger.info(f"  → {metrics.format_row(dataset)}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_metrics(
    predictions: dict[str, str],
    labels: dict[str, str],
    dataset: str | None = None,
) -> DatasetMetrics | None:
    """Compute metrics for predictions against labels."""
    y_true, y_pred = [], []
    for sample_id, true_label in labels.items():
        if sample_id in predictions:
            if dataset is None or dataset_from_id(sample_id) == dataset:
                y_true.append(true_label)
                y_pred.append(predictions[sample_id])

    if not y_true:
        return None

    report: dict[str, Any] = classification_report(
        y_true, y_pred,
        target_names=["Argument", "No-Argument"],
        digits=4, output_dict=True, zero_division=0,
    )  # type: ignore[assignment]

    return DatasetMetrics(
        n_samples=len(y_true),
        macro_f1=report["macro avg"]["f1-score"],
        accuracy=report["accuracy"],
        argument_f1=report["Argument"]["f1-score"],
        no_argument_f1=report["No-Argument"]["f1-score"],
    )


def format_metrics_table(per_dataset: dict[str, DatasetMetrics], overall: DatasetMetrics | None) -> str:
    """Format metrics as a table string."""
    lines = [METRICS_HEADER, METRICS_SEP]
    for dataset, metrics in per_dataset.items():
        lines.append(metrics.format_row(dataset))
    lines.append(METRICS_SEP)
    if overall:
        lines.append(overall.format_row("TOTAL"))
    return "\n".join(lines)


def evaluate(submission_path: Path, labels_file: str = "dev.jsonl") -> dict:
    """Evaluate a submission file against labels (overall and per-dataset)."""
    labels = load_labels(labels_file)
    if not labels:
        logger.error(f"No labels found for {labels_file}")
        return {}

    predictions = {}
    with open(submission_path) as f:
        for line in f:
            record = json.loads(line)
            predictions[record["id"]] = record["label"]

    if not predictions:
        logger.error("No predictions found")
        return {}

    datasets = sorted(set(dataset_from_id(id_) for id_ in predictions.keys()))

    per_dataset = {}
    for dataset in datasets:
        metrics = compute_metrics(predictions, labels, dataset)
        if metrics:
            per_dataset[dataset] = metrics

    overall = compute_metrics(predictions, labels)

    table = format_metrics_table(per_dataset, overall)
    logger.info(f"\n{'='*60}\nPer-Dataset Results:\n{table}\n{'='*60}")

    metrics_path = submission_path.parent / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Submission: {submission_path}\n")
        f.write(f"Labels: {labels_file}\n\n")
        f.write(table + "\n")
    logger.info(f"Metrics saved: {metrics_path}")

    return {"overall": overall, "per_dataset": per_dataset}


# ---------------------------------------------------------------------------
# Main Inference
# ---------------------------------------------------------------------------


def run_inference(config: InferenceConfig) -> tuple[Path, Path]:
    """Run inference on data and produce submission files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = config.output_dir / "submission.jsonl"
    log_path = config.output_dir / "inference_log.jsonl"

    completed_ids = load_completed_ids(log_path)
    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} samples already completed")

    client_cfg = {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "reasoning": config.reasoning,
    }
    client = make_client(client_cfg)

    samples = load_data(config.input_file)
    labels = load_labels(config.input_file)
    logger.info(f"Loaded {len(samples)} samples from {config.input_file}")
    if labels:
        logger.info("Labels available: will calculate F1 score")

    remaining = [s for s in samples if s["id"] not in completed_ids]

    # Filter by datasets if specified
    if config.datasets:
        remaining = [s for s in remaining if s["dataset"] in config.datasets]
        logger.info(f"Filtering to datasets: {config.datasets}")

    logger.info(f"Processing {len(remaining)} remaining samples")

    if not remaining:
        logger.info("All samples already completed")
        if labels:
            evaluate(submission_path, config.input_file)
        return submission_path, log_path

    by_dataset = defaultdict(list)
    for sample in remaining:
        by_dataset[sample["dataset"]].append(sample)

    with open(submission_path, "a") as f_sub, open(log_path, "a") as f_log:
        for dataset in tqdm(sorted(by_dataset.keys()), desc="Datasets"):
            process_dataset(
                dataset, by_dataset[dataset], config, client, client_cfg, labels, f_sub, f_log
            )

    logger.info(f"\nSubmission: {submission_path}")
    logger.info(f"Log: {log_path}")

    if labels:
        logger.info("\nFinal Evaluation:")
        evaluate(submission_path, config.input_file)

    return submission_path, log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> InferenceConfig:
    """Build InferenceConfig from CLI args and optional TOML config."""
    toml_cfg = load_config(args.config) if args.config else {}
    llm_cfg = toml_cfg.get("llm", {})
    sub_cfg = toml_cfg.get("submission", {})

    return InferenceConfig(
        model=args.model or llm_cfg.get("model"),
        provider=args.provider if args.provider != "together_ai" else llm_cfg.get("provider", args.provider),
        temperature=args.temperature if args.temperature != 0.0 else llm_cfg.get("temperature", 0.0),
        reasoning=args.reasoning or llm_cfg.get("reasoning", False),
        context_strategy=(
            args.context_strategy if args.context_strategy != "dynamic"
            else sub_cfg.get("context_strategy", "dynamic")
        ),
        input_file=(
            args.input_file if args.input_file != "test.jsonl"
            else sub_cfg.get("input_file", "test.jsonl")
        ),
        output_dir=Path(
            args.output_dir if args.output_dir != Path("submissions")
            else sub_cfg.get("output_dir", "submissions")
        ),
        datasets=args.datasets or sub_cfg.get("datasets"),
        max_workers=args.max_workers if args.max_workers != 8 else sub_cfg.get("max_workers", 8),
    )


def main():
    parser = argparse.ArgumentParser(description="Run inference on GAIC test set for submission")
    parser.add_argument("--config", type=Path, help="TOML config file")
    parser.add_argument("--model", type=str, help="Model name (required if no --config)")
    parser.add_argument("--provider", type=str, default="together_ai", help="LLM provider")
    parser.add_argument("--context-strategy", type=str, default="dynamic", choices=["c0", "c1", "dynamic"])
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--input-file", type=str, default="test.jsonl")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel API calls (default: 8)")
    parser.add_argument("--datasets", nargs="+", help="Filter to specific datasets (e.g., ABSTRCT ACQUA)")

    args = parser.parse_args()
    config = build_config(args)

    if not config.model:
        parser.error("--model is required (or provide --config with llm.model)")

    logger.info(f"Model: {config.model}")
    logger.info(f"Provider: {config.provider}")
    logger.info(f"Context strategy: {config.context_strategy}")
    logger.info(f"Input file: {config.input_file}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Max workers: {config.max_workers}")
    if config.datasets:
        logger.info(f"Datasets: {config.datasets}")

    run_inference(config)


if __name__ == "__main__":
    main()
