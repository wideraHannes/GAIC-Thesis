import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger


DATA_DIR = Path(__file__).parent.parent / "data" / "GAIC-2026" / "data"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "experiments" / "zero_shot_guideline_outputs"
)
SAMPLE_SIZE_PER_DATASET = 10  # Number of samples per dataset (balanced)

MODEL = "llama3.1:8b"

# Guideline mode: "CORRECT" uses matching guideline per dataset,
# "NO_GUIDELINE" uses a simple 1-liner definition,
# or specify a dataset name to use that guideline for all datasets
GUIDELINE_MODE = "NO_GUIDELINE"  # Options: "NO_GUIDELINE", "ABSTRCT", "ARGUMINSCI", "PE", "USELEC", "CORRECT"

ALL_GUIDELINE_MODES = [
    "NO_GUIDELINE",
    "CORRECT",
    "ABSTRCT",
    "ARGUMINSCI",
    "PE",
    "USELEC",
]

# Simple 1-liner definition used when NO_GUIDELINE mode is selected
SIMPLE_ARGUMENT_DEFINITION = (
    "An argument is a claim supported by reasoning or evidence."
)

DATASETS = [
    "ABSTRCT",
    "ARGUMINSCI",
    "PE",
    "USELEC",
]


def load_data():
    """Load training data and labels."""
    texts = {}
    labels = {}

    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            texts[item["id"]] = item["sentence"]

    with open(DATA_DIR / "train_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]

    return texts, labels


def get_dataset_from_id(id_: str) -> str:
    """Extract dataset name from sample ID."""
    return id_.rsplit("-", 2)[0]


def load_guidelines(dataset: str) -> str:
    """Load the argument definition guidelines for a dataset."""
    guideline_path = DATA_DIR / dataset / "guidelines" / f"{dataset}-Guidelines.md"
    logger.info(f"Loading guidelines from: {guideline_path}")
    if guideline_path.exists():
        return guideline_path.read_text().strip()

    raise FileNotFoundError(f"Guidelines file not found for dataset: {dataset}")


def classify_zero_shot(client, text: str, guidelines: str | None) -> str:
    """Classify a single text using zero-shot prompting with optional guidelines."""
    if guidelines is None:
        # Simple mode without detailed guidelines
        system_prompt = f"""You are a text classifier. Decide if the following text contains an argument or not.
{SIMPLE_ARGUMENT_DEFINITION}
Return ONLY the label, nothing else: 'Argument' or 'No-Argument'"""
    else:
        system_prompt = f"""## Role
You are a text classifier. Decide if the following text contains an argument or not.

## Argument Definition
{guidelines}

## Output
Return ONLY the label, nothing else: 'Argument' or 'No-Argument'"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
        extra_body={
            "options": {
                "num_ctx": 8192,  # <-- biggest speed win (try 4096/8192)
            }
        },
    )
    return response.choices[0].message.content.strip()


def normalize_label(pred: str) -> str:
    """Normalize prediction to standard label format."""
    pred_lower = pred.lower().strip()
    if "no-argument" in pred_lower or "no argument" in pred_lower or pred_lower == "no":
        return "No-Argument"
    elif "argument" in pred_lower or pred_lower == "yes":
        return "Argument"
    else:
        return pred


def compute_metrics(sample_results: list) -> dict:
    """Compute classification metrics for a list of sample results."""
    if not sample_results:
        return {
            "total_samples": 0,
            "valid_predictions": 0,
            "unparseable_predictions": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    y_true = [r["true_label"] for r in sample_results]
    y_pred = [r["normalized_prediction"] for r in sample_results]

    valid_indices = [
        i for i, p in enumerate(y_pred) if p in ["Argument", "No-Argument"]
    ]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if len(valid_indices) == 0:
        return {
            "total_samples": len(sample_results),
            "valid_predictions": 0,
            "unparseable_predictions": len(sample_results),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    return {
        "total_samples": len(sample_results),
        "valid_predictions": len(valid_indices),
        "unparseable_predictions": len(sample_results) - len(valid_indices),
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "precision": precision_score(
            y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0
        ),
        "recall": recall_score(
            y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0
        ),
        "f1_score": f1_score(
            y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0
        ),
        # todo classification_report
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    }


def run_experiment(guideline_mode: str):
    """Run a single experiment with the specified guideline mode."""
    logger.info(f"Running experiment with guideline mode: {guideline_mode}")
    texts, labels = load_data()

    # Load guidelines for each dataset (skip if NO_GUIDELINE mode)
    guidelines_by_dataset = {}
    if guideline_mode != "NO_GUIDELINE":
        for ds in DATASETS:
            guidelines = load_guidelines(ds)
            if guidelines:
                guidelines_by_dataset[ds] = guidelines
            else:
                raise FileNotFoundError(f"Guidelines not found for dataset: {ds}")

    # Group IDs by dataset
    ids_by_dataset = {ds: [] for ds in DATASETS}
    for id_ in texts.keys():
        ds = get_dataset_from_id(id_)
        if ds in ids_by_dataset:
            ids_by_dataset[ds].append(id_)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy_key")

    all_results = {}
    all_samples = []

    for dataset in tqdm(DATASETS, desc="Datasets"):
        dataset_ids = ids_by_dataset[dataset]

        # Select guideline based on mode
        if guideline_mode == "NO_GUIDELINE":
            guideline_source = "NO_GUIDELINE"
            guidelines = None
        elif guideline_mode == "CORRECT":
            guideline_source = dataset
            guidelines = guidelines_by_dataset[guideline_source]
        else:
            guideline_source = guideline_mode
            guidelines = guidelines_by_dataset[guideline_source]

        # Sample balanced from this dataset
        argument_ids = [i for i in dataset_ids if labels[i] == "Argument"][
            : SAMPLE_SIZE_PER_DATASET // 2
        ]
        no_argument_ids = [i for i in dataset_ids if labels[i] == "No-Argument"][
            : SAMPLE_SIZE_PER_DATASET // 2
        ]
        sample_ids = argument_ids + no_argument_ids

        dataset_results = []

        for id_ in sample_ids:
            text = texts[id_]
            true_label = labels[id_]

            raw_pred = classify_zero_shot(client, text, guidelines)
            normalized = normalize_label(raw_pred)

            result = {
                "id": id_,
                "dataset": dataset,
                "guideline_used": guideline_source,
                "text": text,
                "true_label": true_label,
                "raw_prediction": raw_pred,
                "normalized_prediction": normalized,
                "correct": normalized == true_label,
            }
            dataset_results.append(result)
            all_samples.append(result)

        metrics = compute_metrics(dataset_results)
        all_results[dataset] = {
            "guidelines": guidelines if guidelines else SIMPLE_ARGUMENT_DEFINITION,
            "metrics": metrics,
            "samples": dataset_results,
        }

    # Compute overall metrics
    overall_metrics = compute_metrics(all_samples)

    # Build output JSON
    output = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
            "guideline_mode": guideline_mode,
            "uses_dataset_specific_guidelines": guideline_mode == "CORRECT",
            "overall": overall_metrics,
            "by_dataset": {ds: all_results[ds]["metrics"] for ds in DATASETS},
            "ranking": sorted(
                [(ds, all_results[ds]["metrics"]["accuracy"]) for ds in DATASETS],
                key=lambda x: x[1],
                reverse=True,
            ),
        },
        "datasets": all_results,
    }

    # Save to JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR
        / f"zero_shot_guideline_{guideline_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_file}")
    return output_file, output


def run_all_experiments():
    """Run experiments with all guideline modes (4x4 matrix + CORRECT baseline)."""
    logger.info(f"Running all {len(ALL_GUIDELINE_MODES)} guideline mode experiments")
    output_files = []
    all_outputs = {}

    for mode in ALL_GUIDELINE_MODES:
        output_file, output = run_experiment(mode)
        output_files.append(output_file)
        all_outputs[mode] = output

    # Build summary with 4x4 matrix
    matrix = {}
    for guideline_mode in ALL_GUIDELINE_MODES:
        if guideline_mode == "CORRECT":
            continue
        matrix[guideline_mode] = {}
        for dataset in DATASETS:
            metrics = all_outputs[guideline_mode]["summary"]["by_dataset"][dataset]
            matrix[guideline_mode][dataset] = {
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
            }

    # Add CORRECT baseline (diagonal)
    correct_results = {}
    for dataset in DATASETS:
        metrics = all_outputs["CORRECT"]["summary"]["by_dataset"][dataset]
        correct_results[dataset] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
        }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
        "datasets": DATASETS,
        "guideline_modes": ALL_GUIDELINE_MODES,
        "matrix_4x4": matrix,
        "correct_baseline": correct_results,
        "overall_by_mode": {
            mode: all_outputs[mode]["summary"]["overall"]
            for mode in ALL_GUIDELINE_MODES
        },
        "output_files": [str(f) for f in output_files],
    }

    # Save summary file
    summary_file = (
        OUTPUT_DIR
        / f"zero_shot_guideline_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All experiments completed!")
    logger.info("Output files:")
    for f in output_files:
        logger.info(f"  - {f}")
    logger.info(f"Summary file: {summary_file}")

    return output_files, summary_file


def main():
    """Run a single experiment with the configured GUIDELINE_MODE."""
    run_experiment(GUIDELINE_MODE)
    # Returns tuple but we ignore it for single runs


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_experiments()
    else:
        main()
