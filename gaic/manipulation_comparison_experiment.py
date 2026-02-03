"""
Manipulation Comparison Experiment: Original vs Feger vs Shuffle

Compares three conditions on the ABSTRCT dataset:
1. Original: unmodified sentences
2. Feger manipulation: remove stop words, function words, discourse markers, punctuation
3. Shuffle manipulation: randomize word order

Reports F1, accuracy, precision, recall for each condition.
"""

import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from helper import manipulate_sentence

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "GAIC-2026" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "manipulation_comparison_outputs"

DATASET = "ABSTRCT"
SAMPLE_SIZE = 100  # Total samples (balanced: 50 Argument, 50 No-Argument)
MODEL = "llama3.1:8b"


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


def deterministic_shuffle(sentence: str, seed_suffix: str = "") -> str:
    """Shuffle words deterministically based on sentence content."""
    words = sentence.split()
    seed = int(hashlib.md5((sentence + seed_suffix).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


def classify_zero_shot(client, text: str) -> str:
    """Classify a single text using zero-shot prompting."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a text classifier. Decide if the following text contains an argument or not.

### `argument`
Label a sentence/span as **`argument`** if it expresses an **argumentative component**:
- a **Claim**: a concluding/assertive statement by the author about the study outcome, or
- a **Premise**: an observed **fact/measurement/outcome/side-effect report** from the trial used as evidence.

### `no argument`
Label as **`no argument`** if the sentence/span contains **neither a claim nor an evidence premise**.

# Return Format
Return ONLY the label, nothing else: 'Argument' or 'No-Argument'"""
            },
            {"role": "user", "content": text}
        ],
        temperature=0,
        max_tokens=10,
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
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "precision": float(precision_score(
            y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0
        )),
        "recall": float(recall_score(
            y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0
        )),
    }


def main():
    print("=" * 70)
    print("MANIPULATION COMPARISON EXPERIMENT")
    print(f"Dataset: {DATASET} | Samples: {SAMPLE_SIZE} | Model: {MODEL}")
    print("=" * 70)

    # Load data
    texts, labels = load_data()

    # Filter for dataset samples
    dataset_ids = [id_ for id_ in texts.keys() if get_dataset_from_id(id_) == DATASET]
    print(f"Found {len(dataset_ids)} total samples for {DATASET}")

    # Sample balanced dataset
    argument_ids = [i for i in dataset_ids if labels[i] == "Argument"][:SAMPLE_SIZE // 2]
    no_argument_ids = [i for i in dataset_ids if labels[i] == "No-Argument"][:SAMPLE_SIZE // 2]
    sample_ids = argument_ids + no_argument_ids
    print(f"Using {len(argument_ids)} Argument + {len(no_argument_ids)} No-Argument samples")

    # Initialize client
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy_key")

    # Storage for results
    samples = []
    original_preds = []
    feger_preds = []
    shuffle_preds = []
    true_labels = []
    inference_times = []

    # Process each sample
    for id_ in tqdm(sample_ids, desc="Processing samples"):
        original_text = texts[id_]
        true_label = labels[id_]

        # Create manipulated versions
        feger_text = manipulate_sentence(original_text)
        shuffle_text = deterministic_shuffle(original_text)

        # Classify all three versions
        start = time.time()
        original_raw = classify_zero_shot(client, original_text)
        inference_times.append(time.time() - start)

        start = time.time()
        feger_raw = classify_zero_shot(client, feger_text)
        inference_times.append(time.time() - start)

        start = time.time()
        shuffle_raw = classify_zero_shot(client, shuffle_text)
        inference_times.append(time.time() - start)

        # Normalize predictions
        original_pred = normalize_label(original_raw)
        feger_pred = normalize_label(feger_raw)
        shuffle_pred = normalize_label(shuffle_raw)

        # Store results
        sample_result = {
            "id": id_,
            "true_label": true_label,
            "original_text": original_text,
            "feger_text": feger_text,
            "shuffle_text": shuffle_text,
            "original_raw": original_raw,
            "feger_raw": feger_raw,
            "shuffle_raw": shuffle_raw,
            "original_prediction": original_pred,
            "feger_prediction": feger_pred,
            "shuffle_prediction": shuffle_pred,
            "original_correct": original_pred == true_label,
            "feger_correct": feger_pred == true_label,
            "shuffle_correct": shuffle_pred == true_label,
        }
        samples.append(sample_result)

        true_labels.append(true_label)
        original_preds.append(original_pred)
        feger_preds.append(feger_pred)
        shuffle_preds.append(shuffle_pred)

    # Compute metrics for each condition
    original_metrics = compute_metrics(true_labels, original_preds)
    feger_metrics = compute_metrics(true_labels, feger_preds)
    shuffle_metrics = compute_metrics(true_labels, shuffle_preds)

    # Compute deltas
    delta_feger_f1 = original_metrics["f1_score"] - feger_metrics["f1_score"]
    delta_shuffle_f1 = original_metrics["f1_score"] - shuffle_metrics["f1_score"]
    delta_feger_acc = original_metrics["accuracy"] - feger_metrics["accuracy"]
    delta_shuffle_acc = original_metrics["accuracy"] - shuffle_metrics["accuracy"]

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<15} {'F1':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 55)
    print(f"{'Original':<15} {original_metrics['f1_score']:>10.2%} {original_metrics['accuracy']:>10.2%} {original_metrics['precision']:>10.2%} {original_metrics['recall']:>10.2%}")
    print(f"{'Feger':<15} {feger_metrics['f1_score']:>10.2%} {feger_metrics['accuracy']:>10.2%} {feger_metrics['precision']:>10.2%} {feger_metrics['recall']:>10.2%}")
    print(f"{'Shuffle':<15} {shuffle_metrics['f1_score']:>10.2%} {shuffle_metrics['accuracy']:>10.2%} {shuffle_metrics['precision']:>10.2%} {shuffle_metrics['recall']:>10.2%}")
    print("-" * 55)

    print(f"\n{'Delta (Original - Manipulated)':}")
    print(f"  Feger:   ΔF1 = {delta_feger_f1:+.2%}  |  ΔAcc = {delta_feger_acc:+.2%}")
    print(f"  Shuffle: ΔF1 = {delta_shuffle_f1:+.2%}  |  ΔAcc = {delta_shuffle_acc:+.2%}")

    # Reference comparison
    print(f"\n{'Reference (Feger et al. encoders):'}")
    print(f"  Encoder Δ ≤ 0.02 (2%)")
    print(f"  Your LLM Feger Δ = {delta_feger_f1:.2%} ({delta_feger_f1/0.02:.1f}x encoder)")

    print("=" * 70)

    # Build output JSON
    output = {
        "experiment": {
            "name": "manipulation_comparison",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "dataset": DATASET,
            "sample_size": SAMPLE_SIZE,
            "avg_inference_time_seconds": sum(inference_times) / len(inference_times) if inference_times else 0,
        },
        "metrics": {
            "original": original_metrics,
            "feger": feger_metrics,
            "shuffle": shuffle_metrics,
        },
        "deltas": {
            "feger": {
                "delta_f1": delta_feger_f1,
                "delta_accuracy": delta_feger_acc,
                "delta_precision": original_metrics["precision"] - feger_metrics["precision"],
                "delta_recall": original_metrics["recall"] - feger_metrics["recall"],
            },
            "shuffle": {
                "delta_f1": delta_shuffle_f1,
                "delta_accuracy": delta_shuffle_acc,
                "delta_precision": original_metrics["precision"] - shuffle_metrics["precision"],
                "delta_recall": original_metrics["recall"] - shuffle_metrics["recall"],
            },
        },
        "comparison": {
            "encoder_reference_delta": 0.02,
            "feger_vs_encoder_ratio": delta_feger_f1 / 0.02 if delta_feger_f1 > 0 else 0,
            "shuffle_vs_encoder_ratio": delta_shuffle_f1 / 0.02 if delta_shuffle_f1 > 0 else 0,
            "feger_vs_shuffle": delta_feger_f1 - delta_shuffle_f1,
        },
        "samples": samples,
    }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"comparison_{DATASET}_{timestamp}_{MODEL.replace(':', '_')}.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_file, output


if __name__ == "__main__":
    main()
