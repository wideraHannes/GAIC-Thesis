"""
Data contamination test for LLM pre-training data leakage.

Probes whether the model memorized sentences from the GAIC benchmark datasets
by prompting it with the first half of a sentence and checking if it can
reproduce the second half verbatim. Follows the methodology from Golchin &
Surdeanu (2023) and the GPT-4 technical report.
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "GAIC-2026" / "data"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "experiments" / "contamination_test_outputs"
)
SAMPLE_SIZE_PER_DATASET = 10
MODEL = "llama3.1:8b"
SEED = 42

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

# Domain descriptions used to give the model context about where the sentence
# comes from, making the completion task more targeted (following Golchin &
# Surdeanu's "guided prompting" approach).
DATASET_DOMAINS = {
    "ABSTRCT": "a scientific medical abstract",
    "ACQUA": "a legal court decision about acquired rights",
    "AEC": "an argumentative essay written by a student",
    "AFS": "an argumentative essay or debate forum",
    "ARGUMINSCI": "a scientific research article",
    "FINARG": "a financial earnings call transcript",
    "IAM": "an internet argument or online discussion",
    "PE": "a persuasive essay written by a student",
    "SCIARK": "a scientific research article",
    "USELEC": "a U.S. presidential election debate transcript",
}


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


def split_sentence(sentence: str) -> tuple[str, str]:
    """Split a sentence roughly in half at a word boundary."""
    words = sentence.split()
    mid = len(words) // 2
    # Ensure at least 3 words on each side
    mid = max(3, min(mid, len(words) - 3))
    first_half = " ".join(words[:mid])
    second_half = " ".join(words[mid:])
    return first_half, second_half


def prompt_completion(client, first_half: str, dataset: str) -> str:
    """Ask the model to complete a partial sentence."""
    domain = DATASET_DOMAINS.get(dataset, "an academic text")
    prompt = (
        f"Complete the following sentence from {domain}. "
        f"Output ONLY the rest of the sentence, nothing else.\n\n"
        f'"{first_half} ...'
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for comparison."""
    import re

    text = text.strip().strip('"').strip("'")
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())


def compute_overlap_metrics(reference: str, completion: str) -> dict:
    """Compute token-level overlap between reference and completion."""
    ref_norm = normalize(reference)
    comp_norm = normalize(completion)

    ref_tokens = ref_norm.split()
    comp_tokens = comp_norm.split()

    if not ref_tokens or not comp_tokens:
        return {
            "exact_match": False,
            "rouge_l": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
            "longest_common_subsequence": 0,
            "ref_token_count": len(ref_tokens),
            "comp_token_count": len(comp_tokens),
        }

    # Longest common subsequence
    lcs_len = _lcs_length(ref_tokens, comp_tokens)

    precision = lcs_len / len(comp_tokens) if comp_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    rouge_l = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Simple token overlap (bag-of-words)
    ref_set = set(ref_tokens)
    comp_set = set(comp_tokens)
    overlap = ref_set & comp_set
    bow_precision = len(overlap) / len(comp_set) if comp_set else 0.0
    bow_recall = len(overlap) / len(ref_set) if ref_set else 0.0
    bow_f1 = (
        (2 * bow_precision * bow_recall) / (bow_precision + bow_recall)
        if (bow_precision + bow_recall) > 0
        else 0.0
    )

    return {
        "exact_match": ref_norm == comp_norm,
        "rouge_l": rouge_l,
        "token_precision": bow_precision,
        "token_recall": bow_recall,
        "token_f1": bow_f1,
        "longest_common_subsequence": lcs_len,
        "ref_token_count": len(ref_tokens),
        "comp_token_count": len(comp_tokens),
    }


def _lcs_length(x: list, y: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def main():
    random.seed(SEED)

    texts, labels = load_data()

    # Group IDs by dataset
    ids_by_dataset = {ds: [] for ds in DATASETS}
    for id_ in texts:
        ds = get_dataset_from_id(id_)
        if ds in ids_by_dataset:
            ids_by_dataset[ds].append(id_)

    # Filter to sentences with enough words for a meaningful split
    MIN_WORDS = 8
    for ds in DATASETS:
        ids_by_dataset[ds] = [
            id_ for id_ in ids_by_dataset[ds] if len(texts[id_].split()) >= MIN_WORDS
        ]

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy_key")

    all_results = {}
    all_samples = []

    for dataset in tqdm(DATASETS, desc="Datasets"):
        dataset_ids = ids_by_dataset[dataset]
        sample_ids = random.sample(
            dataset_ids, min(SAMPLE_SIZE_PER_DATASET, len(dataset_ids))
        )

        dataset_samples = []

        for id_ in tqdm(sample_ids, desc=f"  {dataset}", leave=False):
            sentence = texts[id_]
            first_half, second_half = split_sentence(sentence)

            start_time = time.time()
            completion = prompt_completion(client, first_half, dataset)
            elapsed = time.time() - start_time

            metrics = compute_overlap_metrics(second_half, completion)

            sample = {
                "id": id_,
                "dataset": dataset,
                "label": labels[id_],
                "full_sentence": sentence,
                "prompt_prefix": first_half,
                "expected_suffix": second_half,
                "model_completion": completion,
                "inference_time": elapsed,
                **metrics,
            }
            dataset_samples.append(sample)
            all_samples.append(sample)

        # Aggregate dataset-level metrics
        n = len(dataset_samples)
        ds_metrics = {
            "n_samples": n,
            "exact_match_rate": sum(s["exact_match"] for s in dataset_samples) / n
            if n
            else 0.0,
            "mean_rouge_l": sum(s["rouge_l"] for s in dataset_samples) / n
            if n
            else 0.0,
            "mean_token_f1": sum(s["token_f1"] for s in dataset_samples) / n
            if n
            else 0.0,
            "mean_token_recall": sum(s["token_recall"] for s in dataset_samples) / n
            if n
            else 0.0,
        }

        all_results[dataset] = {"metrics": ds_metrics, "samples": dataset_samples}

        print(
            f"\n{dataset}: "
            f"Exact={ds_metrics['exact_match_rate']:.0%} | "
            f"ROUGE-L={ds_metrics['mean_rouge_l']:.2f} | "
            f"Token-F1={ds_metrics['mean_token_f1']:.2f} | "
            f"Token-Recall={ds_metrics['mean_token_recall']:.2f}"
        )

    # Overall summary
    n_total = len(all_samples)
    overall = {
        "n_samples": n_total,
        "exact_match_rate": sum(s["exact_match"] for s in all_samples) / n_total
        if n_total
        else 0.0,
        "mean_rouge_l": sum(s["rouge_l"] for s in all_samples) / n_total
        if n_total
        else 0.0,
        "mean_token_f1": sum(s["token_f1"] for s in all_samples) / n_total
        if n_total
        else 0.0,
        "mean_token_recall": sum(s["token_recall"] for s in all_samples) / n_total
        if n_total
        else 0.0,
    }

    print(
        f"\n{'='*60}\n"
        f"OVERALL: "
        f"Exact={overall['exact_match_rate']:.0%} | "
        f"ROUGE-L={overall['mean_rouge_l']:.2f} | "
        f"Token-F1={overall['mean_token_f1']:.2f} | "
        f"Token-Recall={overall['mean_token_recall']:.2f}\n"
        f"{'='*60}"
    )

    # Contamination assessment
    print("\n--- Contamination Assessment ---")
    for ds in DATASETS:
        m = all_results[ds]["metrics"]
        if m["exact_match_rate"] > 0.3:
            level = "HIGH"
        elif m["mean_rouge_l"] > 0.5:
            level = "MODERATE"
        elif m["mean_rouge_l"] > 0.3:
            level = "LOW"
        else:
            level = "MINIMAL"
        print(f"  {ds:15s}: {level} (ROUGE-L={m['mean_rouge_l']:.2f})")

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
            "min_words": MIN_WORDS,
            "seed": SEED,
            "method": "sentence_completion",
            "description": (
                "Data contamination test via sentence completion. "
                "The model is prompted with the first half of training sentences "
                "and asked to complete them. High ROUGE-L or exact match rates "
                "suggest the model memorized these sentences during pre-training."
            ),
        },
        "overall": overall,
        "by_dataset": {ds: all_results[ds]["metrics"] for ds in DATASETS},
        "ranking_by_rouge_l": sorted(
            [(ds, all_results[ds]["metrics"]["mean_rouge_l"]) for ds in DATASETS],
            key=lambda x: x[1],
            reverse=True,
        ),
        "datasets": all_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR
        / f"contamination_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
