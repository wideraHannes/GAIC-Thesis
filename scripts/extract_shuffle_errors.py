#!/usr/bin/env python3
"""
Extract sentences where shuffle manipulation changed the prediction.
Outputs a CSV for manual annotation.
"""

import json
from pathlib import Path
import pandas as pd

V1_DIR = Path("experiments/v1")

def extract_shuffle_changes(json_path: Path) -> list[dict]:
    """Extract samples where shuffle changed the prediction."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    model = data.get("model", json_path.parent.name)
    context = data.get("config", {}).get("experiment", {}).get("experiment_name", "unknown")

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        for sample in dataset_data.get("samples", []):
            pred_orig = sample.get("pred_original")
            pred_shuffle = sample.get("pred_shuffle")
            true_label = sample.get("true_label")

            # Identify prediction change type
            if pred_orig == pred_shuffle:
                change_type = "no_change"
            elif pred_orig == true_label and pred_shuffle != true_label:
                change_type = "correct_to_wrong"  # Most interesting!
            elif pred_orig != true_label and pred_shuffle == true_label:
                change_type = "wrong_to_correct"  # Also interesting
            else:
                change_type = "wrong_to_wrong_different"

            if change_type != "no_change":
                rows.append({
                    "model": model,
                    "context": context,
                    "dataset": dataset_name,
                    "id": sample.get("id"),
                    "sentence": sample.get("sentence"),
                    "sent_shuffle": sample.get("sent_shuffle"),
                    "true_label": true_label,
                    "pred_original": pred_orig,
                    "pred_shuffle": pred_shuffle,
                    "change_type": change_type,
                    # Annotation columns (to fill manually)
                    "has_discourse_marker": "",
                    "has_hedging": "",
                    "has_causal_structure": "",
                    "has_contrastive": "",
                    "claim_premise_structure": "",
                    "notes": "",
                })

    return rows


def main():
    all_rows = []

    # Find all JSON result files
    for json_file in V1_DIR.rglob("*.json"):
        print(f"Processing: {json_file}")
        rows = extract_shuffle_changes(json_file)
        all_rows.extend(rows)
        print(f"  Found {len(rows)} prediction changes")

    df = pd.DataFrame(all_rows)

    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Total prediction changes: {len(df)}")
    print(f"\nBy change type:")
    print(df["change_type"].value_counts())
    print(f"\nBy model:")
    print(df["model"].value_counts())

    # Focus on "correct_to_wrong" - these are the key errors
    c2w = df[df["change_type"] == "correct_to_wrong"]
    print(f"\n=== Correct→Wrong (most important): {len(c2w)} samples ===")
    print(c2w["dataset"].value_counts())

    # Export for annotation
    output_path = Path("experiments/v1/shuffle_error_analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"\nExported to: {output_path}")

    # Export just correct_to_wrong for focused annotation
    c2w_path = Path("experiments/v1/shuffle_correct_to_wrong.csv")
    c2w.to_csv(c2w_path, index=False)
    print(f"Exported correct→wrong subset to: {c2w_path}")


if __name__ == "__main__":
    main()
