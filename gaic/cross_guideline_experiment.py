"""
Cross-Guideline Experiment

Tests the 6 datasets WITHOUT annotation guidelines against each of the 4 available
guidelines to identify which annotation style was likely used for each dataset.

Datasets to test (no guidelines): ACQUA, AEC, AFS, FINARG, IAM, SCIARK
Guidelines to apply: ABSTRCT, ARGUMINSCI, PE, USELEC
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "GAIC-2026" / "data"
OUTPUT_BASE = Path(__file__).parent.parent / "experiments" / "cross_guideline_outputs"

SAMPLE_SIZE_PER_DATASET = 10  # Number of samples per dataset (balanced)
MODEL = "llama3.1:8b"

# Datasets without guidelines - to be tested
DATASETS_TO_TEST = ["ACQUA", "AEC", "AFS", "FINARG", "IAM", "SCIARK"]

# Datasets with guidelines - to use as annotation sources
GUIDELINE_SOURCES = ["ABSTRCT", "ARGUMINSCI", "PE", "USELEC"]


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


def classify_zero_shot(client, text: str, guidelines: str) -> str:
    """Classify a single text using zero-shot prompting with guidelines."""
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
                "num_ctx": 8192,
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
    }


def run_cross_guideline_experiment():
    """Main experiment: test 6 datasets against 4 guidelines."""
    logger.info("Starting cross-guideline experiment")
    logger.info(f"Datasets to test: {DATASETS_TO_TEST}")
    logger.info(f"Guidelines to apply: {GUIDELINE_SOURCES}")

    texts, labels = load_data()

    # Load all guidelines upfront
    guidelines_by_source = {}
    for guideline_name in GUIDELINE_SOURCES:
        guidelines_by_source[guideline_name] = load_guidelines(guideline_name)

    # Group IDs by dataset
    ids_by_dataset = {ds: [] for ds in DATASETS_TO_TEST}
    for id_ in texts.keys():
        ds = get_dataset_from_id(id_)
        if ds in ids_by_dataset:
            ids_by_dataset[ds].append(id_)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy_key")

    # Results matrix: {guideline: {dataset: metrics}}
    results_matrix = {}
    all_guideline_outputs = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for guideline_name in tqdm(GUIDELINE_SOURCES, desc="Guidelines"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with guideline: {guideline_name}")
        logger.info(f"{'='*60}")

        guideline_text = guidelines_by_source[guideline_name]
        output_dir = OUTPUT_BASE / guideline_name
        output_dir.mkdir(parents=True, exist_ok=True)

        guideline_results = {}
        all_samples_for_guideline = []

        for dataset in tqdm(DATASETS_TO_TEST, desc=f"Datasets ({guideline_name})", leave=False):
            dataset_ids = ids_by_dataset[dataset]

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

                raw_pred = classify_zero_shot(client, text, guideline_text)
                normalized = normalize_label(raw_pred)

                result = {
                    "id": id_,
                    "dataset": dataset,
                    "guideline_used": guideline_name,
                    "text": text,
                    "true_label": true_label,
                    "raw_prediction": raw_pred,
                    "normalized_prediction": normalized,
                    "correct": normalized == true_label,
                }
                dataset_results.append(result)
                all_samples_for_guideline.append(result)

            metrics = compute_metrics(dataset_results)
            guideline_results[dataset] = {
                "metrics": metrics,
                "samples": dataset_results,
            }

            logger.info(
                f"  {dataset}: accuracy={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1_score']:.3f}"
            )

        # Store metrics in matrix
        results_matrix[guideline_name] = {
            ds: guideline_results[ds]["metrics"] for ds in DATASETS_TO_TEST
        }

        # Compute overall metrics for this guideline
        overall_metrics = compute_metrics(all_samples_for_guideline)

        # Build output for this guideline
        guideline_output = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
                "guideline_source": guideline_name,
                "datasets_tested": DATASETS_TO_TEST,
                "overall": overall_metrics,
                "by_dataset": {ds: guideline_results[ds]["metrics"] for ds in DATASETS_TO_TEST},
                "ranking": sorted(
                    [
                        (ds, guideline_results[ds]["metrics"]["accuracy"])
                        for ds in DATASETS_TO_TEST
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                ),
            },
            "datasets": guideline_results,
        }

        all_guideline_outputs[guideline_name] = guideline_output

        # Save results for this guideline
        output_file = output_dir / f"cross_guideline_{guideline_name}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(guideline_output, f, indent=2)
        logger.info(f"Saved results to: {output_file}")

    # Generate summary: best guideline per dataset
    best_guideline_per_dataset = {}
    for dataset in DATASETS_TO_TEST:
        best_guideline = None
        best_accuracy = -1
        best_f1 = -1
        for guideline_name in GUIDELINE_SOURCES:
            metrics = results_matrix[guideline_name][dataset]
            # Use accuracy as primary, F1 as tiebreaker
            if metrics["accuracy"] > best_accuracy or (
                metrics["accuracy"] == best_accuracy and metrics["f1_score"] > best_f1
            ):
                best_accuracy = metrics["accuracy"]
                best_f1 = metrics["f1_score"]
                best_guideline = guideline_name
        best_guideline_per_dataset[dataset] = {
            "best_guideline": best_guideline,
            "accuracy": best_accuracy,
            "f1_score": best_f1,
            "all_results": {
                g: {
                    "accuracy": results_matrix[g][dataset]["accuracy"],
                    "f1_score": results_matrix[g][dataset]["f1_score"],
                }
                for g in GUIDELINE_SOURCES
            },
        }

    # Build full summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
        "datasets_tested": DATASETS_TO_TEST,
        "guideline_sources": GUIDELINE_SOURCES,
        "matrix_4x6": {
            guideline: {
                dataset: {
                    "accuracy": results_matrix[guideline][dataset]["accuracy"],
                    "f1_score": results_matrix[guideline][dataset]["f1_score"],
                }
                for dataset in DATASETS_TO_TEST
            }
            for guideline in GUIDELINE_SOURCES
        },
        "best_guideline_per_dataset": best_guideline_per_dataset,
        "overall_by_guideline": {
            g: all_guideline_outputs[g]["summary"]["overall"] for g in GUIDELINE_SOURCES
        },
    }

    # Save summary file
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_BASE / f"cross_guideline_SUMMARY_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("\nBest guideline per dataset:")
    for dataset, info in best_guideline_per_dataset.items():
        logger.info(
            f"  {dataset}: {info['best_guideline']} "
            f"(accuracy={info['accuracy']:.3f}, f1={info['f1_score']:.3f})"
        )

    return summary_file, summary


def main():
    """Run the cross-guideline experiment."""
    run_cross_guideline_experiment()


if __name__ == "__main__":
    main()
