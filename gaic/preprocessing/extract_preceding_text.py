"""Extract preceding sentences for all samples in GAIC datasets.

For datasets with document context (ABSTRCT, ARGUMINSCI, FINARG, PE, SCIARK, USELEC),
this script extracts the 2 sentences immediately preceding each argumentative sentence
and writes them to context/{DATASET}/data/preceding.jsonl.

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


def extract_preceding_sentences_for_dataset(dataset: str) -> list[dict]:
    """Load all train+dev samples for a dataset and extract preceding sentences."""
    entries = []
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
                entries.append(
                    {"id": item["id"], "split": split, "preceding": preceding}
                )
                count += 1

        logger.info(f"    Extracted preceding sentences for {count} samples")

    return entries


def write_preceding_data(output_dir: Path, entries: list[dict]) -> None:
    """Write preceding sentence data to context/{DATASET}/data/preceding.jsonl."""
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    output_path = data_dir / "preceding.jsonl"
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        f"  Written {len(entries)} entries to {output_path.relative_to(CONTEXT_DIR.parent)}"
    )


def process_dataset(dataset: str) -> dict:
    """Process a single dataset: extract preceding sentences and write output.

    Returns a status dict with 'dataset', 'status' ('OK'/'FAIL'), 'count', and 'error' (if any).
    """
    logger.info(f"Processing {dataset}...")
    output_dir = CONTEXT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = extract_preceding_sentences_for_dataset(dataset)
        if not entries:
            logger.warning(f"  No entries found for {dataset}")
            return {"dataset": dataset, "status": "OK", "count": 0, "error": None}

        write_preceding_data(output_dir, entries)
        logger.info(f"  {dataset}: OK ({len(entries)} samples)")
        return {
            "dataset": dataset,
            "status": "OK",
            "count": len(entries),
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
