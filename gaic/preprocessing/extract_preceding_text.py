"""Extract preceding sentences for all samples in GAIC datasets.

For datasets with document context (ABSTRCT, ARGUMINSCI, FINARG, PE, SCIARK, USELEC),
this script extracts the 2 sentences immediately preceding each argumentative sentence
and writes them to individual files: context/{DATASET}/data/{ID}.txt

Usage:
    uv run python gaic/preprocessing/extract_preceding_text.py
"""

import json
import re
from pathlib import Path

from loguru import logger

from config.paths import CONTEXT_DIR, GAIC_DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Datasets where the source text provides document context (preceding sentences)
DATASETS_WITH_DOCUMENT_CONTEXT = [
    "ABSTRCT",
    "ARGUMINSCI",
    "FINARG",
    "PE",
    "SCIARK",
    "USELEC",
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in parts if s.strip()]


def get_preceding_sentences(doc_text: str, target: str, n: int = 2) -> list[str]:
    """Return the n sentences immediately before target in doc_text."""
    idx = doc_text.find(target)
    if idx == -1:
        return []
    before = doc_text[:idx].strip()
    sentences = split_sentences(before)
    return sentences[-n:]


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------


def extract_preceding_sentences_for_dataset(dataset: str, output_dir: Path) -> int:
    """Load all train+dev samples for a dataset and extract preceding sentences.

    Writes each sample's preceding text to context/{dataset}/data/{id}.txt
    Returns the count of successfully processed samples.
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    total_count = 0
    for split in ("train", "dev"):
        jsonl_path = GAIC_DATA_DIR / f"{split}.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"  {split}.jsonl not found, skipping")
            continue

        logger.info(f"  Processing {split}.jsonl...")
        count = 0
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                if item["id"].rsplit("-", 2)[0] != dataset:
                    continue

                doc_rel = item.get("document", "")
                if not doc_rel:
                    logger.warning(f"    No document for {item['id']}, skipping")
                    continue

                # Resolve path: stored as "./DATASET/data/..." relative to GAIC_DATA_DIR
                doc_path = GAIC_DATA_DIR / doc_rel.lstrip("./")
                if not doc_path.exists():
                    logger.warning(f"    Document not found: {doc_path}")
                    continue

                doc_text = doc_path.read_text()
                preceding = get_preceding_sentences(doc_text, item["sentence"])

                # Write to individual file: context/{dataset}/data/{id}.txt
                output_file = data_dir / f"{item['id']}.txt"
                with open(output_file, "w") as f_out:
                    f_out.write("\n".join(preceding))

                count += 1

        logger.info(f"    Extracted preceding sentences for {count} samples")
        total_count += count

    return total_count




def process_dataset(dataset: str) -> dict:
    """Process a single dataset: extract preceding sentences and write to individual files.

    Returns a status dict with 'dataset', 'status' ('OK'/'FAIL'), 'count', and 'error' (if any).
    """
    logger.info(f"Processing {dataset}...")
    output_dir = CONTEXT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        count = extract_preceding_sentences_for_dataset(dataset, output_dir)
        if count == 0:
            logger.warning(f"  No entries found for {dataset}")
            return {"dataset": dataset, "status": "OK", "count": 0, "error": None}

        logger.info(f"  {dataset}: OK ({count} samples)")
        return {
            "dataset": dataset,
            "status": "OK",
            "count": count,
            "error": None,
        }

    except Exception as e:
        logger.error(f"  {dataset}: FAIL - {e}")
        return {"dataset": dataset, "status": "FAIL", "count": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run preceding sentence extraction for all datasets with document context."""
    logger.info(
        f"Starting preceding sentence extraction for {len(DATASETS_WITH_DOCUMENT_CONTEXT)} datasets"
    )
    logger.info(f"Output directory: {CONTEXT_DIR}")

    results = []
    for dataset in DATASETS_WITH_DOCUMENT_CONTEXT:
        result = process_dataset(dataset)
        results.append(result)

    # Summary
    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] == "FAIL"]
    total_samples = sum(r["count"] for r in ok)

    logger.info("=" * 60)
    logger.info(f"Extraction complete: {len(ok)} OK, {len(fail)} FAIL")
    logger.info(f"Total samples processed: {total_samples}")
    for r in results:
        status_icon = "✓" if r["status"] == "OK" else "✗"
        msg = f"  {status_icon} {r['dataset']}: {r['count']} samples"
        if r["error"]:
            msg += f" (Error: {r['error']})"
        logger.info(msg)
    logger.info("=" * 60)

    if fail:
        logger.warning(f"{len(fail)} dataset(s) failed. Check logs above for details.")


if __name__ == "__main__":
    main()
