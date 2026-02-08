"""
Analysis script for unified experiment results.

Loads and analyzes results from manipulation and context experiments.
"""

import json
from pathlib import Path
import statistics


class ResultsAnalyzer:
    """Analyze experiment results."""

    def __init__(self, results_dir: Path):
        """Initialize with results directory."""
        self.results_dir = results_dir

    def load_latest_results(self, experiment_type: str) -> dict:
        """Load the most recent results file for an experiment type."""
        pattern = f"{experiment_type}_*.json"
        files = sorted(self.results_dir.glob(pattern), reverse=True)

        if not files:
            raise FileNotFoundError(f"No results found for {experiment_type}")

        with open(files[0]) as f:
            return json.load(f)

    def analyze_manipulation(self, results: dict) -> dict:
        """Analyze manipulation experiment results."""
        analysis = {
            "summary": {},
            "datasets": {},
            "overall": {},
        }

        # Per-dataset analysis
        for dataset, data in results["datasets"].items():
            dataset_analysis = {}

            # Extract F1 scores
            for manip in ["M0", "M1", "M2"]:
                if manip in data["manipulations"]:
                    f1 = data["manipulations"][manip]["metrics"]["f1_macro"]
                    dataset_analysis[f"f1_{manip}"] = f1

            # Extract deltas
            if "delta_feger" in data:
                dataset_analysis["delta_feger"] = data["delta_feger"]
            if "delta_shuffle" in data:
                dataset_analysis["delta_shuffle"] = data["delta_shuffle"]

            analysis["datasets"][dataset] = dataset_analysis

        # Overall averages
        all_deltas_feger = [
            d.get("delta_feger")
            for d in analysis["datasets"].values()
            if "delta_feger" in d
        ]
        all_deltas_shuffle = [
            d.get("delta_shuffle")
            for d in analysis["datasets"].values()
            if "delta_shuffle" in d
        ]

        if all_deltas_feger:
            analysis["overall"]["mean_delta_feger"] = statistics.mean(all_deltas_feger)
            analysis["overall"]["median_delta_feger"] = statistics.median(
                all_deltas_feger
            )
            analysis["overall"]["min_delta_feger"] = min(all_deltas_feger)
            analysis["overall"]["max_delta_feger"] = max(all_deltas_feger)

        if all_deltas_shuffle:
            analysis["overall"]["mean_delta_shuffle"] = statistics.mean(
                all_deltas_shuffle
            )
            analysis["overall"]["median_delta_shuffle"] = statistics.median(
                all_deltas_shuffle
            )

        # Comparison to Feger et al. threshold (≤ 0.02)
        analysis["overall"]["datasets_within_feger_threshold"] = sum(
            1 for d in all_deltas_feger if abs(d) <= 0.02
        )
        analysis["overall"]["datasets_exceeding_feger_threshold"] = sum(
            1 for d in all_deltas_feger if abs(d) > 0.02
        )

        return analysis

    def analyze_context(self, results: dict) -> dict:
        """Analyze context experiment results."""
        analysis = {
            "summary": {},
            "datasets": {},
            "overall": {},
        }

        # Per-dataset analysis
        for dataset, data in results["datasets"].items():
            dataset_analysis = {}

            # Extract F1 scores for each context level
            for level in ["C0", "C1", "C2", "C3"]:
                if level in data["context_levels"]:
                    f1 = data["context_levels"][level]["metrics"]["f1_macro"]
                    dataset_analysis[f"f1_{level}"] = f1

            # Extract deltas
            for level in ["C1", "C2", "C3"]:
                delta_key = f"delta_{level}"
                if delta_key in data:
                    dataset_analysis[delta_key] = data[delta_key]

            # Best context level
            f1_scores = {
                level: data["context_levels"][level]["metrics"]["f1_macro"]
                for level in data["context_levels"].keys()
            }
            best_level = max(f1_scores.keys(), key=lambda x: f1_scores[x])
            dataset_analysis["best_context_level"] = best_level
            dataset_analysis["best_f1"] = f1_scores[best_level]

            analysis["datasets"][dataset] = dataset_analysis

        # Overall averages per context level
        for level in ["C0", "C1", "C2", "C3"]:
            f1_values = [
                d[f"f1_{level}"]
                for d in analysis["datasets"].values()
                if f"f1_{level}" in d
            ]
            if f1_values:
                analysis["overall"][f"mean_f1_{level}"] = statistics.mean(f1_values)
                analysis["overall"][f"median_f1_{level}"] = statistics.median(f1_values)

        # Overall deltas
        for level in ["C1", "C2", "C3"]:
            delta_key = f"delta_{level}"
            delta_values = [
                d[delta_key] for d in analysis["datasets"].values() if delta_key in d
            ]
            if delta_values:
                analysis["overall"][f"mean_{delta_key}"] = statistics.mean(delta_values)
                analysis["overall"][f"median_{delta_key}"] = statistics.median(
                    delta_values
                )

        # Best overall context level
        context_levels = ["C0", "C1", "C2", "C3"]
        mean_f1s = {
            level: analysis["overall"].get(f"mean_f1_{level}", 0)
            for level in context_levels
            if f"mean_f1_{level}" in analysis["overall"]
        }
        if mean_f1s:
            best_overall = max(mean_f1s.keys(), key=lambda x: mean_f1s[x])
            analysis["overall"]["best_context_level"] = best_overall
            analysis["overall"]["best_mean_f1"] = mean_f1s[best_overall]

        return analysis

    def print_manipulation_summary(self, analysis: dict):
        """Print summary of manipulation analysis."""
        print("\n" + "=" * 80)
        print("PART 1: MANIPULATION EXPERIMENT ANALYSIS")
        print("=" * 80)

        print("\n📊 Overall Statistics:")
        overall = analysis["overall"]

        if "mean_delta_feger" in overall:
            print(f"  Mean Δ_feger:   {overall['mean_delta_feger']:+.4f}")
            print(f"  Median Δ_feger: {overall['median_delta_feger']:+.4f}")
            print(
                f"  Range:          [{overall['min_delta_feger']:+.4f}, {overall['max_delta_feger']:+.4f}]"
            )

            print("\n  Feger et al. threshold (|Δ| ≤ 0.02):")
            print(
                f"    Within threshold:    {overall['datasets_within_feger_threshold']} datasets"
            )
            print(
                f"    Exceeding threshold: {overall['datasets_exceeding_feger_threshold']} datasets"
            )

        if "mean_delta_shuffle" in overall:
            print(f"\n  Mean Δ_shuffle:   {overall['mean_delta_shuffle']:+.4f}")
            print(f"  Median Δ_shuffle: {overall['median_delta_shuffle']:+.4f}")

        print("\n📈 Per-Dataset Results:")
        for dataset, data in sorted(analysis["datasets"].items()):
            print(f"\n  {dataset}:")
            if "f1_M0" in data:
                print(f"    F1 M0 (baseline):  {data['f1_M0']:.4f}")
            if "f1_M1" in data:
                print(f"    F1 M1 (Feger):     {data['f1_M1']:.4f}")
                if "delta_feger" in data:
                    print(f"    Δ_feger:           {data['delta_feger']:+.4f}")
            if "f1_M2" in data:
                print(f"    F1 M2 (shuffle):   {data['f1_M2']:.4f}")
                if "delta_shuffle" in data:
                    print(f"    Δ_shuffle:         {data['delta_shuffle']:+.4f}")

    def print_context_summary(self, analysis: dict):
        """Print summary of context analysis."""
        print("\n" + "=" * 80)
        print("PART 2: CONTEXT UTILIZATION ANALYSIS")
        print("=" * 80)

        print("\n📊 Overall Statistics:")
        overall = analysis["overall"]

        print("  Mean F1 by Context Level:")
        for level in ["C0", "C1", "C2", "C3"]:
            key = f"mean_f1_{level}"
            if key in overall:
                print(f"    {level}: {overall[key]:.4f}")

        print("\n  Mean Improvement over C0:")
        for level in ["C1", "C2", "C3"]:
            key = f"mean_delta_{level}"
            if key in overall:
                print(f"    Δ_{level}: {overall[key]:+.4f}")

        if "best_context_level" in overall:
            print(f"\n  🏆 Best Overall Context: {overall['best_context_level']}")
            print(f"     Mean F1: {overall['best_mean_f1']:.4f}")

        print("\n📈 Per-Dataset Results:")
        for dataset, data in sorted(analysis["datasets"].items()):
            print(f"\n  {dataset}:")
            for level in ["C0", "C1", "C2", "C3"]:
                f1_key = f"f1_{level}"
                if f1_key in data:
                    print(f"    F1 {level}: {data[f1_key]:.4f}", end="")
                    delta_key = f"delta_{level}"
                    if delta_key in data:
                        print(f"  (Δ: {data[delta_key]:+.4f})", end="")
                    print()

            if "best_context_level" in data:
                print(
                    f"    🏆 Best: {data['best_context_level']} (F1: {data['best_f1']:.4f})"
                )


def main():
    """Main entry point for analysis."""
    import sys

    # Get results directory
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).parent.parent / "experiments" / "unified_outputs"

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    analyzer = ResultsAnalyzer(results_dir)

    # Analyze manipulation results
    try:
        print("Loading manipulation results...")
        manip_results = analyzer.load_latest_results("manipulation")
        manip_analysis = analyzer.analyze_manipulation(manip_results)
        analyzer.print_manipulation_summary(manip_analysis)

        # Save analysis
        output_file = results_dir / "manipulation_analysis_latest.json"
        with open(output_file, "w") as f:
            json.dump(manip_analysis, f, indent=2)
        print(f"\n✅ Analysis saved to: {output_file}")

    except FileNotFoundError:
        print("⚠️  No manipulation results found")

    # Analyze context results
    try:
        print("\n\nLoading context results...")
        context_results = analyzer.load_latest_results("context")
        context_analysis = analyzer.analyze_context(context_results)
        analyzer.print_context_summary(context_analysis)

        # Save analysis
        output_file = results_dir / "context_analysis_latest.json"
        with open(output_file, "w") as f:
            json.dump(context_analysis, f, indent=2)
        print(f"\n✅ Analysis saved to: {output_file}")

    except FileNotFoundError:
        print("⚠️  No context results found")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
