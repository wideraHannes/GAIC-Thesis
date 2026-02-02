"""
Zero-shot classification with manipulation experiment.

Replicates Feger et al.'s manipulation study: compares model F1 score on
original sentences vs. manipulated sentences (function words removed).

If the LLM relies on lexical shortcuts, F1 should drop significantly
when function words are removed. If it truly understands argument structure,
F1 should remain relatively stable.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import f1_score

from helper import manipulate_sentence


DATA_DIR = Path(__file__).parent.parent / "data" / "GAIC-2026" / "data"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "experiments" / "zero_shot_manipulation_outputs"
)
SAMPLE_SIZE_PER_DATASET = 10  # Number of samples per dataset (balanced)
MODEL = "llama3.1:8b"

DATASETS = [
    "ABSTRCT",
    "ACQUA",
    "AEC",
    "AFS",
    "ARGUMINSCI",
    "FINARG",
    "IAM",
    "PE",
    "SCIARK",
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


def classify_zero_shot(client, text: str) -> str:
    """Classify a single text using zero-shot prompting."""
    response = client.responses.create(
        model=MODEL,
        instructions="""You are a text classifier. Decide if the following text contains an argument or not.
        ### `argument`

        Label a sentence/span as **`argument`** if it expresses an **argumentative component**:

        - a **Claim**: a concluding/assertive statement by the author about the study outcome (a statement/claim that needs support), or
        - a **Premise**: an observed **fact/measurement/outcome/side-effect report** from the trial used as evidence supporting or attacking a claim.

        In this guideline, an argument is understood as a **claim that is supported or attacked by evidence (premises)**.

        ### `no argument`

        Label as **`no argument`** if the sentence/span contains **neither a claim nor an evidence premise** (i.e., it does not make an outcome assertion and does not report observations/measurements as evidence).
        
        
        # Return Format
        Return ONLY the label, nothing else: 'Argument' or 'No-Argument'""",
        input=text,
        temperature=0,
        max_output_tokens=5,
    )
    return response.output_text.strip()


def normalize_label(pred: str) -> str:
    """Normalize prediction to standard label format."""
    pred_lower = pred.lower().strip()
    if "no-argument" in pred_lower or "no argument" in pred_lower or pred_lower == "no":
        return "No-Argument"
    elif "argument" in pred_lower or pred_lower == "yes":
        return "Argument"
    else:
        return pred


def compute_f1(results: list, prediction_key: str) -> float:
    """Compute F1 score for a specific prediction type."""
    if not results:
        return 0.0

    y_true = [r["true_label"] for r in results]
    y_pred = [r[prediction_key] for r in results]

    # Filter to valid predictions only
    valid_indices = [
        i for i, p in enumerate(y_pred) if p in ["Argument", "No-Argument"]
    ]

    if not valid_indices:
        return 0.0

    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    return float(f1_score(y_true_valid, y_pred_valid, pos_label="Argument", zero_division=0))


def main():
    texts, labels = load_data()

    # Group IDs by dataset
    ids_by_dataset = {ds: [] for ds in DATASETS}
    for id_ in texts.keys():
        ds = get_dataset_from_id(id_)
        if ds in ids_by_dataset:
            ids_by_dataset[ds].append(id_)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy_key")

    all_results = {}
    all_samples = []
    inference_times = []

    print("=" * 70)
    print("MANIPULATION EXPERIMENT: Original vs. Function-Words-Removed")
    print("=" * 70)

    for dataset in tqdm(DATASETS, desc="Datasets"):
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

        for id_ in tqdm(sample_ids, desc=f"  {dataset}", leave=False):
            text = texts[id_]
            true_label = labels[id_]

            # Manipulate text (remove function words)
            manipulated_text = manipulate_sentence(text)

            # Classify original text
            start_time = time.time()
            original_raw = classify_zero_shot(client, text)
            original_time = time.time() - start_time

            # Classify manipulated text
            start_time = time.time()
            manipulated_raw = classify_zero_shot(client, manipulated_text)
            manipulated_time = time.time() - start_time

            inference_times.extend([original_time, manipulated_time])

            # Normalize predictions
            original_pred = normalize_label(original_raw)
            manipulated_pred = normalize_label(manipulated_raw)

            result = {
                "id": id_,
                "dataset": dataset,
                "original_text": text,
                "manipulated_text": manipulated_text,
                "true_label": true_label,
                "original_raw_prediction": original_raw,
                "original_prediction": original_pred,
                "manipulated_raw_prediction": manipulated_raw,
                "manipulated_prediction": manipulated_pred,
                "original_correct": original_pred == true_label,
                "manipulated_correct": manipulated_pred == true_label,
            }
            dataset_results.append(result)
            all_samples.append(result)

        # Compute metrics for this dataset
        original_f1 = compute_f1(dataset_results, "original_prediction")
        manipulated_f1 = compute_f1(dataset_results, "manipulated_prediction")
        delta = original_f1 - manipulated_f1

        all_results[dataset] = {
            "metrics": {
                "original_f1": original_f1,
                "manipulated_f1": manipulated_f1,
                "delta": delta,
                "total_samples": len(dataset_results),
            },
            "samples": dataset_results,
        }

        print(
            f"\n{dataset}: Orig F1={original_f1:.2%} | Manip F1={manipulated_f1:.2%} | Delta={delta:+.2%}"
        )

    # Compute overall metrics
    overall_original_f1 = compute_f1(all_samples, "original_prediction")
    overall_manipulated_f1 = compute_f1(all_samples, "manipulated_prediction")
    overall_delta = overall_original_f1 - overall_manipulated_f1

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Original F1:    {overall_original_f1:.2%}")
    print(f"Manipulated F1: {overall_manipulated_f1:.2%}")
    print(f"Delta (Orig - Manip): {overall_delta:+.2%}")
    print("=" * 70)

    if overall_delta > 0.1:
        print("Significant F1 drop suggests reliance on lexical shortcuts")
    elif overall_delta < -0.05:
        print("F1 improved with manipulation (unexpected)")
    else:
        print("Relatively stable F1 suggests robust understanding")

    # Build output JSON
    output = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
            "avg_inference_time_seconds": sum(inference_times) / len(inference_times)
            if inference_times
            else 0,
            "overall": {
                "original_f1": overall_original_f1,
                "manipulated_f1": overall_manipulated_f1,
                "delta": overall_delta,
                "total_samples": len(all_samples),
            },
            "by_dataset": {ds: all_results[ds]["metrics"] for ds in DATASETS},
            "ranking_by_delta": sorted(
                [(ds, all_results[ds]["metrics"]["delta"]) for ds in DATASETS],
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
        / f"manipulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MODEL.replace(':', '_')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
