#!/usr/bin/env python3
"""
Test script to verify DCQ perturbations are valid.

Based on DCQ method (https://github.com/shahriargolchin/DCQ):
- Perturbations use word-level synonym substitution
- Structure and meaning must be preserved
- Must be DISTINCT from original (not identical)

Checks:
1. Not identical to original
2. Similar length (structure preserved)
3. Some word overlap (not completely unrelated)
"""

import json
from pathlib import Path
import sys


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity (set-based word overlap)."""
    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union


def test_sample(sample: dict, verbose: bool = False) -> dict:
    """Test a single sample's perturbations against the original."""
    original = sample["original"]
    perturbations = sample["perturbations"]
    sample_id = sample["id"]

    results = {
        "id": sample_id,
        "passed": True,
        "failures": [],
        "metrics": [],
    }

    orig_words = len(tokenize(original))

    for i, perturb in enumerate(perturbations):
        # Clean perturbation (remove leading A), B), etc.)
        clean_perturb = perturb.strip()
        if clean_perturb.startswith(("A)", "B)", "C)", "D)")):
            clean_perturb = clean_perturb[2:].strip()

        pert_words = len(tokenize(clean_perturb))
        jaccard = jaccard_similarity(original, clean_perturb)
        is_identical = original.strip().lower() == clean_perturb.lower()

        # Length ratio (structure preservation check)
        length_ratio = pert_words / orig_words if orig_words > 0 else 0.0

        metrics = {
            "perturbation_idx": i,
            "jaccard": round(jaccard, 4),
            "length_ratio": round(length_ratio, 4),
            "is_identical": is_identical,
        }
        results["metrics"].append(metrics)

        # Check conditions
        failures = []

        # 1. Must NOT be identical (DCQ requirement: options exclude original)
        if is_identical:
            failures.append(f"Perturbation {i} is IDENTICAL to original")

        # 2. Length should be similar (structure preserved)
        #    Allow some deviation for acronym expansions (e.g., RCP -> Rich Client Platform)
        if length_ratio < 0.55 or length_ratio > 1.85:
            failures.append(f"Perturbation {i} length ratio {length_ratio:.2f} (structure not preserved?)")

        # 3. Should have meaningful word overlap
        #    Short texts (<5 words) may have complete synonym substitution
        if orig_words >= 5 and jaccard < 0.15:
            failures.append(f"Perturbation {i} Jaccard {jaccard:.2f} too low (texts not similar enough?)")

        if failures:
            results["passed"] = False
            results["failures"].extend(failures)
            if verbose:
                print(f"  [{sample_id}] Perturbation {i}:")
                print(f"    Original:  {original[:80]}...")
                print(f"    Perturbed: {clean_perturb[:80]}...")
                print(f"    Failures:  {failures}")

    return results


def test_file(filepath: Path, verbose: bool = False) -> dict:
    """Test all samples in a JSONL file."""
    print(f"\nTesting: {filepath.name}")
    print("-" * 60)

    with open(filepath) as f:
        samples = [json.loads(line) for line in f]

    total = len(samples)
    passed = 0
    failed_samples = []
    all_jaccard = []
    all_length_ratios = []

    for sample in samples:
        result = test_sample(sample, verbose=verbose)
        if result["passed"]:
            passed += 1
        else:
            failed_samples.append(result)

        for m in result["metrics"]:
            all_jaccard.append(m["jaccard"])
            all_length_ratios.append(m["length_ratio"])

    print(f"Samples: {total}, Passed: {passed}, Failed: {total - passed}")
    if all_jaccard:
        print(f"Jaccard:      mean={sum(all_jaccard)/len(all_jaccard):.4f}, "
              f"min={min(all_jaccard):.4f}, max={max(all_jaccard):.4f}")
        print(f"Length ratio: mean={sum(all_length_ratios)/len(all_length_ratios):.4f}, "
              f"min={min(all_length_ratios):.4f}, max={max(all_length_ratios):.4f}")

    return {
        "file": filepath.name,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "failed_samples": failed_samples,
    }


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    data_dir = Path(__file__).parent
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("DCQ Perturbation Validation")
    print("=" * 60)
    print(f"Testing {len(jsonl_files)} files...")

    all_results = []
    total_passed = 0
    total_samples = 0

    for filepath in jsonl_files:
        result = test_file(filepath, verbose=verbose)
        all_results.append(result)
        total_passed += result["passed"]
        total_samples += result["total"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files: {len(jsonl_files)}, Samples: {total_samples}, "
          f"Passed: {total_passed}, Failed: {total_samples - total_passed}")

    if total_passed == total_samples:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n✗ {total_samples - total_passed} SAMPLES FAILED")
        if verbose:
            for result in all_results:
                for sample in result["failed_samples"]:
                    print(f"  {sample['id']}: {sample['failures']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
