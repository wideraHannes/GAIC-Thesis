"""Fix contamination range calculation (min != max)."""
import json
from pathlib import Path

DCQ_DIR = Path(__file__).parent
MODELS = ["mistral-medium-latest", "gpt-5.2-2025-12-11"]

for model in MODELS:
    bcq_dir = DCQ_DIR / "phase3_bcq" / model
    bdq_summary = DCQ_DIR / "phase2_bdq" / model / "bias_summary.json"

    if not bdq_summary.exists():
        print(f"Skip {model}: no BDQ summary")
        continue

    with open(bdq_summary) as f:
        bias = json.load(f)

    bdq_total = bias["total_samples"]
    bdq_counts = {k: v["count"] for k, v in bias["position_frequencies"].items()}
    non_preferred = bias["non_preferred_positions"]

    for report_file in bcq_dir.glob("*_contamination_report.json"):
        if report_file.name.startswith("dep_"):
            continue

        with open(report_file) as f:
            report = json.load(f)

        total = report["total"]
        bcq_per_pos = report["bcq_per_position"]

        # Build triples with CORRECT scaling (float, not integer division)
        triples = [
            (pos, bcq_per_pos.get(pos, 0), (bdq_counts.get(pos, 0) / bdq_total) * total)
            for pos in non_preferred
        ]
        triples.sort(key=lambda x: (-x[1], x[2]))

        max_count = triples[0][1]
        max_bias = triples[0][2]
        max_cont = max_count / total if total > 0 else 0

        # Cohen's Kappa
        kappa = (max_count - max_bias) / (total - max_bias) if total > max_bias else 0

        # Min = max(kappa, second_best_rate)
        if len(triples) > 1:
            second_rate = triples[1][1] / total if total > 0 else 0
            min_cont = max(kappa, second_rate)
        else:
            min_cont = kappa

        # Update report
        old_min, old_max = report["min_contamination"], report["max_contamination"]
        report["min_contamination"] = min_cont * 100
        report["max_contamination"] = max_cont * 100
        report["contamination_range"] = f"[{min_cont * 100:.1f}%, {max_cont * 100:.1f}%]"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"{model}/{report['dataset']}: [{old_min:.1f}%, {old_max:.1f}%] -> {report['contamination_range']}")
