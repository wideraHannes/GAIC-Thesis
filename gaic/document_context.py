"""
Document context extraction utilities for GAIC experiments.

Handles extraction of document context from the GAIC dataset for experiments.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class DocumentContextExtractor:
    """Extract document context for sentences in GAIC datasets."""

    def __init__(self, data_dir: Path):
        """Initialize with data directory path."""
        self.data_dir = data_dir
        self._document_cache: Dict[str, Dict] = {}
        self._load_documents()

    def _load_documents(self):
        """Load all documents from train.jsonl into memory for fast lookup."""
        train_file = self.data_dir / "train.jsonl"
        if not train_file.exists():
            print(f"Warning: train.jsonl not found at {train_file}")
            return

        # Build cache: id -> full document info
        with open(train_file) as f:
            for line in f:
                item = json.loads(line)
                doc_id = self._get_document_id(item["id"])

                # Group sentences by document
                if doc_id not in self._document_cache:
                    self._document_cache[doc_id] = {
                        "sentences": [],
                        "sentence_ids": [],
                    }

                self._document_cache[doc_id]["sentences"].append(item["sentence"])
                self._document_cache[doc_id]["sentence_ids"].append(item["id"])

    def _get_document_id(self, sample_id: str) -> str:
        """
        Extract document ID from sample ID.

        Sample ID format: DATASET-DOCID-SENTID
        Example: ABSTRCT-12345-001
        """
        parts = sample_id.rsplit("-", 2)
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"  # DATASET-DOCID
        return sample_id

    def get_preceding_sentences(
        self, sample_id: str, num_sentences: int = 2
    ) -> Optional[str]:
        """
        Get N preceding sentences for a given sample.

        Args:
            sample_id: The sample ID (e.g., "ABSTRCT-12345-003")
            num_sentences: Number of preceding sentences to include

        Returns:
            String with preceding sentences joined by space, or None if not available
        """
        doc_id = self._get_document_id(sample_id)

        if doc_id not in self._document_cache:
            return None

        doc_data = self._document_cache[doc_id]

        # Find the position of current sentence
        try:
            current_idx = doc_data["sentence_ids"].index(sample_id)
        except ValueError:
            return None

        # Get preceding sentences
        start_idx = max(0, current_idx - num_sentences)
        preceding = doc_data["sentences"][start_idx:current_idx]

        if not preceding:
            return None

        return " ".join(preceding)

    def get_full_document(self, sample_id: str) -> Optional[str]:
        """
        Get the full document text for a given sample.

        Args:
            sample_id: The sample ID

        Returns:
            Full document text or None if not available
        """
        doc_id = self._get_document_id(sample_id)

        if doc_id not in self._document_cache:
            return None

        return " ".join(self._document_cache[doc_id]["sentences"])

    def get_paragraph_context(
        self, sample_id: str, window_size: int = 2
    ) -> Optional[str]:
        """
        Get surrounding sentences (paragraph context).

        Args:
            sample_id: The sample ID
            window_size: Number of sentences before and after

        Returns:
            Paragraph context or None if not available
        """
        doc_id = self._get_document_id(sample_id)

        if doc_id not in self._document_cache:
            return None

        doc_data = self._document_cache[doc_id]

        # Find the position of current sentence
        try:
            current_idx = doc_data["sentence_ids"].index(sample_id)
        except ValueError:
            return None

        # Get surrounding sentences
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(doc_data["sentences"]), current_idx + window_size + 1)

        context_sentences = doc_data["sentences"][start_idx:end_idx]

        # Exclude the current sentence
        context_sentences = [
            s
            for i, s in enumerate(context_sentences, start=start_idx)
            if i != current_idx
        ]

        if not context_sentences:
            return None

        return " ".join(context_sentences)

    def has_document_context(self, dataset: str, context_info: dict) -> bool:
        """
        Check if document context is available for a dataset.

        Args:
            dataset: Dataset name
            context_info: Context information from dataset.json

        Returns:
            True if document context is available
        """
        return context_info.get("document_context", {}).get("available", False)


def main():
    """Example usage."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "GAIC-2026" / "data"
    extractor = DocumentContextExtractor(data_dir)

    # Example: Get preceding sentences for a sample
    sample_id = "ABSTRCT-19439498-000"
    preceding = extractor.get_preceding_sentences(sample_id, num_sentences=2)

    if preceding:
        print(f"Preceding sentences for {sample_id}:")
        print(preceding)
    else:
        print(f"No preceding sentences available for {sample_id}")


if __name__ == "__main__":
    main()
