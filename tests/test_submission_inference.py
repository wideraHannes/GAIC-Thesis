"""Tests for submission inference pipeline.

Run with: uv run pytest tests/test_submission_inference.py -v
"""

import json
from pathlib import Path

import pytest

from config.paths import CONTEXT_DIR, GAIC_DATA_DIR
from gaic.submission_inference import get_context_sources, load_data
from gaic.unified_experiment import _load_dataset_json, load_context, load_document_context


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "ABSTRCT", "ACQUA", "AEC", "AFS", "ARGUMINSCI",
    "FINARG", "IAM", "PE", "SCIARK", "TACO", "TAPE", "TAUS", "USELEC"
]

# Expected capabilities per dataset (based on dataset.json files)
EXPECTED_CAPABILITIES = {
    "ABSTRCT": {"definition": True, "guideline": True, "document_context": True},
    "ACQUA": {"definition": True, "guideline": False, "document_context": False},
    "AEC": {"definition": True, "guideline": False, "document_context": False},
    "AFS": {"definition": True, "guideline": False, "document_context": False},
    "ARGUMINSCI": {"definition": True, "guideline": True, "document_context": True},
    "FINARG": {"definition": True, "guideline": False, "document_context": True},
    "IAM": {"definition": True, "guideline": False, "document_context": False},
    "PE": {"definition": True, "guideline": True, "document_context": True},
    "SCIARK": {"definition": True, "guideline": False, "document_context": True},
    "TACO": {"definition": True, "guideline": True, "document_context": True},
    "TAPE": {"definition": True, "guideline": True, "document_context": True},
    "TAUS": {"definition": True, "guideline": True, "document_context": True},
    "USELEC": {"definition": True, "guideline": True, "document_context": True},
}


# ---------------------------------------------------------------------------
# Test: Dataset capabilities match expectations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_dataset_capabilities(dataset: str):
    """Verify dataset.json capabilities match expectations."""
    if dataset not in EXPECTED_CAPABILITIES:
        pytest.skip(f"No expected capabilities defined for {dataset}")

    caps = _load_dataset_json(dataset).get("capabilities", {})
    expected = EXPECTED_CAPABILITIES[dataset]

    assert caps.get("has_definition", False) == expected["definition"], \
        f"{dataset}: has_definition mismatch"
    assert caps.get("has_guidelines", False) == expected["guideline"], \
        f"{dataset}: has_guidelines mismatch"
    assert caps.get("has_document_context", False) == expected["document_context"], \
        f"{dataset}: has_document_context mismatch"


# ---------------------------------------------------------------------------
# Test: Context files exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_context_files_exist(dataset: str):
    """Verify context files exist for datasets that claim to have them."""
    if dataset not in EXPECTED_CAPABILITIES:
        pytest.skip(f"No expected capabilities defined for {dataset}")

    expected = EXPECTED_CAPABILITIES[dataset]
    dataset_dir = CONTEXT_DIR / dataset

    if expected["definition"]:
        def_path = dataset_dir / "definition.md"
        assert def_path.exists(), f"{dataset}: definition.md not found"
        assert def_path.read_text().strip(), f"{dataset}: definition.md is empty"

    if expected["guideline"]:
        guide_path = dataset_dir / "guideline.md"
        assert guide_path.exists(), f"{dataset}: guideline.md not found"
        assert guide_path.read_text().strip(), f"{dataset}: guideline.md is empty"

    if expected["document_context"]:
        data_dir = dataset_dir / "data"
        assert data_dir.exists(), f"{dataset}: data/ directory not found"
        files = list(data_dir.glob("*.txt"))
        assert len(files) > 0, f"{dataset}: no .txt files in data/"


# ---------------------------------------------------------------------------
# Test: get_context_sources returns correct sources
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_context_sources_dynamic(dataset: str):
    """Verify get_context_sources returns correct sources for dynamic strategy."""
    if dataset not in EXPECTED_CAPABILITIES:
        pytest.skip(f"No expected capabilities defined for {dataset}")

    sources = get_context_sources(dataset, "dynamic")
    expected = EXPECTED_CAPABILITIES[dataset]

    if expected["definition"]:
        assert "definition" in sources, f"{dataset}: definition missing from sources"
    else:
        assert "definition" not in sources, f"{dataset}: definition should not be in sources"

    if expected["guideline"]:
        assert "guideline" in sources, f"{dataset}: guideline missing from sources"

    if expected["document_context"]:
        assert "document_context" in sources, f"{dataset}: document_context missing from sources"


def test_get_context_sources_c0():
    """c0 strategy should return empty sources."""
    for dataset in ["ABSTRCT", "TAPE", "TACO"]:
        sources = get_context_sources(dataset, "c0")
        assert sources == [], f"{dataset}: c0 should return empty sources"


def test_get_context_sources_c1():
    """c1 strategy should return all three sources."""
    for dataset in ["ABSTRCT", "TAPE", "TACO"]:
        sources = get_context_sources(dataset, "c1")
        assert sources == ["definition", "guideline", "document_context"], \
            f"{dataset}: c1 should return all sources"


# ---------------------------------------------------------------------------
# Test: load_context loads files correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_load_context(dataset: str):
    """Verify load_context loads definition and guideline correctly."""
    if dataset not in EXPECTED_CAPABILITIES:
        pytest.skip(f"No expected capabilities defined for {dataset}")

    expected = EXPECTED_CAPABILITIES[dataset]
    sources_to_load = []
    if expected["definition"]:
        sources_to_load.append("definition")
    if expected["guideline"]:
        sources_to_load.append("guideline")

    context = load_context(dataset, sources_to_load)

    if expected["definition"]:
        assert "definition" in context, f"{dataset}: definition not loaded"
        assert len(context["definition"]) > 50, f"{dataset}: definition too short"

    if expected["guideline"]:
        assert "guideline" in context, f"{dataset}: guideline not loaded"
        assert len(context["guideline"]) > 50, f"{dataset}: guideline too short"


# ---------------------------------------------------------------------------
# Test: Document context loading for TAPE/TAUS/TACO
# ---------------------------------------------------------------------------

def test_document_context_tape_taus_taco():
    """Verify document context files exist and load for TAPE, TAUS, TACO samples."""
    test_file = GAIC_DATA_DIR / "test.jsonl"
    if not test_file.exists():
        pytest.skip("test.jsonl not found")

    samples_checked = {"TAPE": 0, "TAUS": 0, "TACO": 0}

    with open(test_file) as f:
        for line in f:
            item = json.loads(line)
            dataset = item["id"].rsplit("-", 2)[0]

            if dataset in samples_checked and samples_checked[dataset] < 3:
                doc_context = load_document_context(dataset, item["id"])
                assert doc_context, f"{item['id']}: document context is empty"
                assert len(doc_context) > 10, f"{item['id']}: document context too short"
                samples_checked[dataset] += 1

            if all(v >= 3 for v in samples_checked.values()):
                break

    for dataset, count in samples_checked.items():
        assert count >= 3, f"{dataset}: only checked {count} samples"


# ---------------------------------------------------------------------------
# Test: Sample ID to file mapping
# ---------------------------------------------------------------------------

def test_sample_id_file_mapping():
    """Verify sample IDs map to correct document context files."""
    for dataset in ["TAPE", "TAUS", "TACO"]:
        data_dir = CONTEXT_DIR / dataset / "data"
        if not data_dir.exists():
            pytest.fail(f"{dataset}: data/ directory not found")

        files = list(data_dir.glob(f"{dataset}-test-*.txt"))
        assert len(files) > 0, f"{dataset}: no test sample context files found"

        # Check a specific file loads correctly
        sample_id = files[0].stem  # e.g., "TAPE-test-1"
        content = load_document_context(dataset, sample_id)
        assert content, f"{dataset}: failed to load context for {sample_id}"


# ---------------------------------------------------------------------------
# Test: Full pipeline smoke test (no API calls)
# ---------------------------------------------------------------------------

def test_pipeline_context_assembly():
    """Test that context can be assembled for all datasets without API calls."""
    from gaic.unified_experiment import assemble_context

    for dataset in ["TAPE", "TAUS", "TACO", "ABSTRCT"]:
        sources = get_context_sources(dataset, "dynamic")
        static_sources = [s for s in sources if s != "document_context"]
        context_parts = load_context(dataset, static_sources)

        # Load a sample document context
        doc_context = ""
        if "document_context" in sources:
            data_dir = CONTEXT_DIR / dataset / "data"
            files = list(data_dir.glob(f"{dataset}-*.txt"))
            if files:
                sample_id = files[0].stem
                doc_context = load_document_context(dataset, sample_id)

        full_context = assemble_context(context_parts, doc_context)

        assert full_context, f"{dataset}: assembled context is empty"
        assert len(full_context) > 100, f"{dataset}: assembled context too short"

        # Verify expected sections are present
        if "definition" in context_parts:
            assert "Definition" in full_context or "definition" in full_context.lower()
        if "guideline" in context_parts:
            assert "Guideline" in full_context or "guideline" in full_context.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
