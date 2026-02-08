"""Extract structured context from dataset papers and annotation guidelines.

Uses Kreuzberg for PDF-to-text extraction and GPT-5.2 (via Portkey) for
structured information extraction. Produces per-dataset context files
(definition.md, guideline.md, preceding_text.md, dataset.json) under
the project's context/ directory.

Usage:
    uv run python gaic/preprocessing/extract_context.py
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from kreuzberg import extract_file_sync
from loguru import logger
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL

from config.paths import CONTEXT_DIR, GAIC_DATA_DIR, PROJECT_ROOT
from gaic.preprocessing.prompts import (
    DEFINITION_SYSTEM_PROMPT,
    DEFINITION_USER_PROMPT,
    GUIDELINES_SYSTEM_PROMPT,
    GUIDELINES_USER_PROMPT,
)
from gaic.preprocessing.schemas import DefinitionExtraction, GuidelinesExtraction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "@azure-openai-foundry/gpt-5.2-chat"

ALL_DATASETS = [
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

# Datasets that have annotation guideline PDFs
DATASETS_WITH_GUIDELINES = {"ABSTRCT", "ARGUMINSCI", "PE", "USELEC"}

# Datasets where the source text provides document context (preceding sentences)
DATASETS_WITH_DOCUMENT_CONTEXT = {
    "ABSTRCT",
    "ARGUMINSCI",
    "FINARG",
    "PE",
    "SCIARK",
    "USELEC",
}

# ---------------------------------------------------------------------------
# Client & PDF helpers
# ---------------------------------------------------------------------------


def create_client() -> OpenAI:
    """Create an OpenAI client configured with Portkey gateway."""
    load_dotenv()
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PORTKEY_API_KEY not found in environment. Please set it in your .env file."
        )
    return OpenAI(base_url=PORTKEY_GATEWAY_URL, api_key=api_key)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text content from a PDF file using Kreuzberg."""
    result = extract_file_sync(pdf_path)
    return result.content


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


def extract_definition(
    client: OpenAI, dataset: str, paper_text: str
) -> DefinitionExtraction:
    """Extract argument definition from paper text using structured output."""
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": DEFINITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DEFINITION_USER_PROMPT.format(
                    dataset_name=dataset, paper_text=paper_text
                ),
            },
        ],
        seed=42,
        response_format=DefinitionExtraction,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(
            f"Empty response from model for definition extraction of {dataset}"
        )
    return parsed


def extract_guidelines(
    client: OpenAI, dataset: str, guidelines_text: str
) -> GuidelinesExtraction:
    """Extract annotation guidelines from guidelines text using structured output."""
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": GUIDELINES_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GUIDELINES_USER_PROMPT.format(
                    dataset_name=dataset, guidelines_text=guidelines_text
                ),
            },
        ],
        seed=42,
        response_format=GuidelinesExtraction,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(
            f"Empty response from model for guidelines extraction of {dataset}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_definition(output_dir: Path, definition_text: str) -> None:
    """Write definition.md file."""
    (output_dir / "definition.md").write_text(definition_text + "\n")


def write_guideline(output_dir: Path, guideline_text: str | None) -> None:
    """Write guideline.md file. Writes 'Not available' if no guideline."""
    text = guideline_text if guideline_text else "Not available"
    (output_dir / "guideline.md").write_text(text + "\n")


def write_dataset_json(
    output_dir: Path,
    dataset: str,
    definition: str,
    guideline: str | None,
    paper_pdf_path: Path | None,
    paper_text_chars: int | None,
    guidelines_pdf_path: Path | None,
    guidelines_text_chars: int | None,
) -> None:
    """Write dataset.json with all extracted data and metadata."""
    has_doc_context = dataset in DATASETS_WITH_DOCUMENT_CONTEXT
    data = {
        "dataset": dataset,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "definition": definition,
        "guideline": guideline,
        "document_context": {
            "available": has_doc_context,
            "strategy": "2 preceding sentences" if has_doc_context else None,
            "preceding_sentences": 2 if has_doc_context else None,
        },
        "sources": {
            "paper_pdf": str(paper_pdf_path.relative_to(PROJECT_ROOT))
            if paper_pdf_path
            else None,
            "paper_text_chars": paper_text_chars,
            "guidelines_pdf": str(guidelines_pdf_path.relative_to(PROJECT_ROOT))
            if guidelines_pdf_path
            else None,
            "guidelines_text_chars": guidelines_text_chars,
        },
        "capabilities": {
            "has_definition": definition is not None,
            "has_guidelines": guideline is not None,
            "has_document_context": has_doc_context,
        },
    }
    (output_dir / "dataset.json").write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def process_dataset(client: OpenAI, dataset: str) -> dict:
    """Process a single dataset: extract PDFs, call LLM, write outputs.

    Returns a status dict with 'dataset', 'status' ('OK'/'FAIL'), and 'error' (if any).
    """
    logger.info(f"Processing {dataset}...")
    output_dir = CONTEXT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    definition_text = None
    guideline_text = None
    paper_pdf_path = None
    paper_text_chars = None
    guidelines_pdf_path = None
    guidelines_text_chars = None

    try:
        # --- Extract definition from paper PDF ---
        paper_pdf_path = GAIC_DATA_DIR / dataset / "paper" / f"{dataset}.pdf"
        if not paper_pdf_path.exists():
            raise FileNotFoundError(f"Paper PDF not found: {paper_pdf_path}")

        logger.info(f"  Extracting text from {paper_pdf_path.name}...")
        paper_text = extract_pdf_text(paper_pdf_path)
        paper_text_chars = len(paper_text)
        logger.info(f"  Paper text: {paper_text_chars} chars")

        logger.info(f"  Extracting definition via {MODEL}...")
        definition_result = extract_definition(client, dataset, paper_text)
        definition_text = definition_result.definition
        write_definition(output_dir, definition_text)
        logger.info(f"  Definition extracted ({len(definition_text)} chars)")

        # --- Extract guidelines (only for datasets that have them) ---
        if dataset in DATASETS_WITH_GUIDELINES:
            guidelines_pdf_path = (
                GAIC_DATA_DIR / dataset / "guidelines" / f"{dataset}-Guidelines.pdf"
            )
            if not guidelines_pdf_path.exists():
                raise FileNotFoundError(
                    f"Guidelines PDF not found: {guidelines_pdf_path}"
                )

            logger.info(f"  Extracting text from {guidelines_pdf_path.name}...")
            guidelines_text_raw = extract_pdf_text(guidelines_pdf_path)
            guidelines_text_chars = len(guidelines_text_raw)
            logger.info(f"  Guidelines text: {guidelines_text_chars} chars")

            logger.info(f"  Extracting guidelines via {MODEL}...")
            guidelines_result = extract_guidelines(client, dataset, guidelines_text_raw)
            guideline_text = guidelines_result.guidelines
            logger.info(f"  Guidelines extracted ({len(guideline_text)} chars)")

        write_guideline(output_dir, guideline_text)
        write_dataset_json(
            output_dir=output_dir,
            dataset=dataset,
            definition=definition_text,
            guideline=guideline_text,
            paper_pdf_path=paper_pdf_path,
            paper_text_chars=paper_text_chars,
            guidelines_pdf_path=guidelines_pdf_path,
            guidelines_text_chars=guidelines_text_chars,
        )

        logger.info(f"  {dataset}: OK")
        return {"dataset": dataset, "status": "OK", "error": None}

    except Exception as e:
        logger.error(f"  {dataset}: FAIL - {e}")

        # Write partial outputs for whatever we have
        if definition_text:
            write_definition(output_dir, definition_text)
        if guideline_text or dataset not in DATASETS_WITH_GUIDELINES:
            write_guideline(output_dir, guideline_text)
        if definition_text:
            write_dataset_json(
                output_dir=output_dir,
                dataset=dataset,
                definition=definition_text,
                guideline=guideline_text,
                paper_pdf_path=paper_pdf_path,
                paper_text_chars=paper_text_chars,
                guidelines_pdf_path=guidelines_pdf_path,
                guidelines_text_chars=guidelines_text_chars,
            )

        return {"dataset": dataset, "status": "FAIL", "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run context extraction for all 10 datasets."""
    logger.info(f"Starting context extraction for {len(ALL_DATASETS)} datasets")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Output directory: {CONTEXT_DIR}")

    client = create_client()

    results = []
    for dataset in ALL_DATASETS:
        result = process_dataset(client, dataset)
        results.append(result)

    # Summary
    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] == "FAIL"]

    logger.info("=" * 60)
    logger.info(f"Extraction complete: {len(ok)} OK, {len(fail)} FAIL")
    for r in results:
        status_icon = "OK" if r["status"] == "OK" else "FAIL"
        msg = f"  {r['dataset']}: {status_icon}"
        if r["error"]:
            msg += f" ({r['error']})"
        logger.info(msg)
    logger.info("=" * 60)

    if fail:
        logger.warning(f"{len(fail)} dataset(s) failed. Check logs above for details.")


if __name__ == "__main__":
    main()
