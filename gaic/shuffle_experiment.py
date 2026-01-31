"""Word shuffle experiment to test shortcut learning in argument identification.

This experiment tests whether LLMs rely on lexical/syntactic shortcuts by measuring
accuracy degradation when word order is randomized.

Hypothesis: If the model relies on semantic understanding, shuffled sentences should
significantly degrade accuracy. If it relies on keyword detection (shortcut learning),
accuracy may remain high even with shuffled input.
"""

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "GAIC-2026" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "shuffle_outputs"

DATASET = "ABSTRCT"
SAMPLE_SIZE = 50  # Larger sample for statistical significance
MODEL = "llama3.1:8b"


def deterministic_shuffle(sentence: str, seed_suffix: str = "") -> str:
    """Shuffle words deterministically based on sentence content.

    Uses hash of sentence as random seed for reproducibility.
    Same sentence always shuffles the same way across runs.
    """
    words = sentence.split()
    # Create seed from sentence hash for reproducibility
    seed = int(hashlib.md5((sentence + seed_suffix).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


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


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics."""
    valid_indices = [i for i, p in enumerate(y_pred) if p in ["Argument", "No-Argument"]]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if len(valid_indices) == 0:
        return {
            "total_samples": len(y_true),
            "valid_predictions": 0,
            "unparseable_predictions": len(y_true),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    return {
        "total_samples": len(y_true),
        "valid_predictions": len(valid_indices),
        "unparseable_predictions": len(y_true) - len(valid_indices),
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


def run_shuffle_experiment():
    """Compare baseline vs shuffled classification on ABSTRCT."""
    logger.info(f"Running shuffle experiment on {DATASET} with {SAMPLE_SIZE} samples")

    # Load data and guidelines
    texts, labels = load_data()
    guidelines = load_guidelines(DATASET)

    # Filter for ABSTRCT samples
    dataset_ids = [id_ for id_ in texts.keys() if get_dataset_from_id(id_) == DATASET]
    logger.info(f"Found {len(dataset_ids)} samples for {DATASET}")

    # Sample balanced dataset
    argument_ids = [i for i in dataset_ids if labels[i] == "Argument"][: SAMPLE_SIZE // 2]
    no_argument_ids = [i for i in dataset_ids if labels[i] == "No-Argument"][
        : SAMPLE_SIZE // 2
    ]
    sample_ids = argument_ids + no_argument_ids
    logger.info(
        f"Sampled {len(argument_ids)} Argument and {len(no_argument_ids)} No-Argument"
    )

    # Initialize client
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy_key")

    samples = []
    baseline_true = []
    baseline_pred = []
    shuffled_true = []
    shuffled_pred = []

    for id_ in tqdm(sample_ids, desc="Processing samples"):
        original_text = texts[id_]
        shuffled_text = deterministic_shuffle(original_text)
        true_label = labels[id_]

        # Baseline classification (original text)
        baseline_raw = classify_zero_shot(client, original_text, guidelines)
        baseline_norm = normalize_label(baseline_raw)

        # Shuffled classification (same sample, shuffled words)
        shuffled_raw = classify_zero_shot(client, shuffled_text, guidelines)
        shuffled_norm = normalize_label(shuffled_raw)

        sample_result = {
            "id": id_,
            "original_text": original_text,
            "shuffled_text": shuffled_text,
            "true_label": true_label,
            "baseline_raw_prediction": baseline_raw,
            "baseline_prediction": baseline_norm,
            "shuffled_raw_prediction": shuffled_raw,
            "shuffled_prediction": shuffled_norm,
            "baseline_correct": baseline_norm == true_label,
            "shuffled_correct": shuffled_norm == true_label,
            "predictions_match": baseline_norm == shuffled_norm,
        }
        samples.append(sample_result)

        baseline_true.append(true_label)
        baseline_pred.append(baseline_norm)
        shuffled_true.append(true_label)
        shuffled_pred.append(shuffled_norm)

    # Compute metrics
    baseline_metrics = compute_metrics(baseline_true, baseline_pred)
    shuffled_metrics = compute_metrics(shuffled_true, shuffled_pred)

    # Per-class analysis
    argument_samples = [s for s in samples if s["true_label"] == "Argument"]
    no_argument_samples = [s for s in samples if s["true_label"] == "No-Argument"]

    argument_baseline_acc = (
        sum(1 for s in argument_samples if s["baseline_correct"]) / len(argument_samples)
        if argument_samples
        else 0
    )
    argument_shuffled_acc = (
        sum(1 for s in argument_samples if s["shuffled_correct"]) / len(argument_samples)
        if argument_samples
        else 0
    )
    no_argument_baseline_acc = (
        sum(1 for s in no_argument_samples if s["baseline_correct"])
        / len(no_argument_samples)
        if no_argument_samples
        else 0
    )
    no_argument_shuffled_acc = (
        sum(1 for s in no_argument_samples if s["shuffled_correct"])
        / len(no_argument_samples)
        if no_argument_samples
        else 0
    )

    # Prediction consistency analysis
    predictions_match_count = sum(1 for s in samples if s["predictions_match"])
    flipped_correct_to_incorrect = sum(
        1 for s in samples if s["baseline_correct"] and not s["shuffled_correct"]
    )
    flipped_incorrect_to_correct = sum(
        1 for s in samples if not s["baseline_correct"] and s["shuffled_correct"]
    )

    # Build output
    output = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "dataset": DATASET,
            "guideline": DATASET,
            "sample_size": len(samples),
            "baseline_accuracy": baseline_metrics["accuracy"],
            "shuffled_accuracy": shuffled_metrics["accuracy"],
            "accuracy_drop": baseline_metrics["accuracy"] - shuffled_metrics["accuracy"],
            "baseline_f1": baseline_metrics["f1_score"],
            "shuffled_f1": shuffled_metrics["f1_score"],
            "f1_drop": baseline_metrics["f1_score"] - shuffled_metrics["f1_score"],
        },
        "detailed_metrics": {
            "baseline": baseline_metrics,
            "shuffled": shuffled_metrics,
        },
        "per_class_analysis": {
            "Argument": {
                "count": len(argument_samples),
                "baseline_accuracy": argument_baseline_acc,
                "shuffled_accuracy": argument_shuffled_acc,
                "accuracy_drop": argument_baseline_acc - argument_shuffled_acc,
            },
            "No-Argument": {
                "count": len(no_argument_samples),
                "baseline_accuracy": no_argument_baseline_acc,
                "shuffled_accuracy": no_argument_shuffled_acc,
                "accuracy_drop": no_argument_baseline_acc - no_argument_shuffled_acc,
            },
        },
        "consistency_analysis": {
            "predictions_match": predictions_match_count,
            "predictions_match_pct": predictions_match_count / len(samples),
            "flipped_correct_to_incorrect": flipped_correct_to_incorrect,
            "flipped_incorrect_to_correct": flipped_incorrect_to_correct,
        },
        "samples": samples,
    }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR / f"shuffle_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("SHUFFLE EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Sample size: {len(samples)}")
    logger.info("-" * 60)
    logger.info(f"Baseline accuracy: {baseline_metrics['accuracy']:.2%}")
    logger.info(f"Shuffled accuracy: {shuffled_metrics['accuracy']:.2%}")
    logger.info(
        f"Accuracy drop: {baseline_metrics['accuracy'] - shuffled_metrics['accuracy']:.2%}"
    )
    logger.info("-" * 60)
    logger.info(f"Baseline F1: {baseline_metrics['f1_score']:.2%}")
    logger.info(f"Shuffled F1: {shuffled_metrics['f1_score']:.2%}")
    logger.info("-" * 60)
    logger.info(f"Predictions match: {predictions_match_count}/{len(samples)}")
    logger.info(f"Flipped correct→incorrect: {flipped_correct_to_incorrect}")
    logger.info(f"Flipped incorrect→correct: {flipped_incorrect_to_correct}")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_file}")

    return output_file, output


if __name__ == "__main__":
    run_shuffle_experiment()
