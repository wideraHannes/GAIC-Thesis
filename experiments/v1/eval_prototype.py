#!/usr/bin/env python3
"""
Prototype visualizations for V1 results analysis.
1. Prediction Bias Shift (existing data)
2. Δ Scatter Plots (existing data)
3. Perturbation Curves (new experiment)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12, "axes.labelsize": 10})
sns.set_theme(style="whitegrid", palette="colorblind")

V1_DIR = Path("experiments/v1")

# =============================================================================
# LOAD EXISTING DATA
# =============================================================================

def load_all_results() -> list[dict]:
    """Load all JSON result files."""
    results = []
    for json_file in V1_DIR.rglob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            data["_file"] = json_file.name
            data["_model_dir"] = json_file.parent.name
            results.append(data)
    return results


def extract_predictions(results: list[dict]) -> pd.DataFrame:
    """Extract sample-level predictions for bias analysis."""
    rows = []
    for result in results:
        model = result.get("model", result["_model_dir"])
        context = result.get("config", {}).get("experiment", {}).get("experiment_name", "unknown")

        for dataset_name, dataset_data in result.get("datasets", {}).items():
            for sample in dataset_data.get("samples", []):
                rows.append({
                    "model": model,
                    "context": context,
                    "dataset": dataset_name,
                    "true_label": sample.get("true_label"),
                    "pred_original": sample.get("pred_original"),
                    "pred_content_only": sample.get("pred_content_only"),
                    "pred_shuffle": sample.get("pred_shuffle"),
                    "sentence": sample.get("sentence"),
                })
    return pd.DataFrame(rows)


def extract_metrics(results: list[dict]) -> pd.DataFrame:
    """Extract F1 metrics for delta analysis."""
    rows = []
    for result in results:
        model = result.get("model", result["_model_dir"])
        context = result.get("config", {}).get("experiment", {}).get("experiment_name", "unknown")

        for dataset_name, dataset_data in result.get("datasets", {}).items():
            reports = dataset_data.get("reports", {})
            for manipulation in ["original", "content_only", "shuffle"]:
                report = reports.get(manipulation, {})
                macro_f1 = report.get("macro avg", {}).get("f1-score", np.nan)
                rows.append({
                    "model": model,
                    "context": context,
                    "dataset": dataset_name,
                    "manipulation": manipulation,
                    "macro_f1": macro_f1,
                })
    return pd.DataFrame(rows)


# =============================================================================
# PLOT 1: PREDICTION BIAS SHIFT - GENERIC
# =============================================================================

def plot_prediction_bias_shift(pred_df: pd.DataFrame, model_filter: str = None):
    """
    Show how manipulation shifts prediction distribution.
    Grouped bar: % Argument for each manipulation condition.
    Automatically discovers models and datasets from data.
    """
    if model_filter:
        pred_df = pred_df[pred_df["model"].str.contains(model_filter, case=False)]

    # Generic model name shortening
    def shorten_model_name(name: str) -> str:
        name = str(name).split("/")[-1]
        name = name.replace("-latest", "").replace("-instruct", "")
        name = name.replace("@azure-openai-foundry_", "").replace("@azure-openai-foundry/", "")
        return name[:15]

    # Calculate prediction rates per model/context/dataset/manipulation
    bias_data = []
    for (model, context, dataset), group in pred_df.groupby(["model", "context", "dataset"]):
        n = len(group)
        if n == 0:
            continue
        for manip, col in [("original", "pred_original"),
                           ("content_only", "pred_content_only"),
                           ("shuffle", "pred_shuffle")]:
            if col in group.columns:
                arg_rate = (group[col] == "Argument").sum() / n
                bias_data.append({
                    "model": shorten_model_name(model),
                    "context": context,
                    "model_context": f"{shorten_model_name(model)} {context}",
                    "dataset": dataset,
                    "manipulation": manip,
                    "arg_rate": arg_rate,
                })

    bias_df = pd.DataFrame(bias_data)

    if bias_df.empty:
        print("No data for bias shift plot")
        return

    # Get unique model+context combinations
    model_contexts = sorted(bias_df["model_context"].unique())
    n_plots = len(model_contexts)

    # Create subplots - max 4 per row
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True, squeeze=False)
    axes = axes.flatten()

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    manip_order = ["original", "content_only", "shuffle"]
    manip_colors = {"original": "#3498db", "content_only": "#f39c12", "shuffle": "#2ecc71"}

    for ax, model_ctx in zip(axes[:n_plots], model_contexts):
        model_data = bias_df[bias_df["model_context"] == model_ctx]
        datasets = sorted(model_data["dataset"].unique())

        # Pivot for grouped bar
        pivot = model_data.pivot(index="dataset", columns="manipulation", values="arg_rate")
        # Reorder columns if available
        pivot = pivot[[c for c in manip_order if c in pivot.columns]]

        x = np.arange(len(pivot))
        width = 0.25
        n_manips = len(pivot.columns)

        for i, manip in enumerate(pivot.columns):
            color = manip_colors.get(manip, "#333333")
            ax.bar(x + i * width - (n_manips - 1) * width / 2, pivot[manip], width,
                   label=manip, alpha=0.8, color=color)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Dataset", fontsize=10)
        ax.set_ylabel("P(Argument)", fontsize=10)
        ax.set_title(model_ctx, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Prediction Bias Shift Under Manipulation", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(V1_DIR / "plot_bias_shift.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {V1_DIR / 'plot_bias_shift.png'}")


# =============================================================================
# PLOT 2: DELTA SCATTER PLOT (Model × Context level) - GENERIC
# =============================================================================

def plot_delta_scatter(metrics_df: pd.DataFrame):
    """
    Scatter plot: Δ_content_only vs Δ_shuffle averaged per model+context.
    Each point = one model+context combination (e.g., GPT-4.1 c1, Mistral c2).
    Automatically discovers models and contexts from data.
    """
    # Calculate deltas per dataset first
    pivot = metrics_df.pivot_table(
        values="macro_f1",
        index=["model", "context", "dataset"],
        columns="manipulation",
        aggfunc="mean"
    ).reset_index()

    pivot["delta_content"] = pivot["content_only"] - pivot["original"]
    pivot["delta_shuffle"] = pivot["shuffle"] - pivot["original"]
    pivot["baseline_f1"] = pivot["original"]

    # Aggregate to model+context level (average across datasets)
    agg = pivot.groupby(["model", "context"]).agg({
        "delta_content": "mean",
        "delta_shuffle": "mean",
        "baseline_f1": "mean",
        "dataset": "count",  # number of datasets
    }).reset_index()
    agg.columns = ["model", "context", "delta_content", "delta_shuffle", "baseline_f1", "n_datasets"]

    # Create readable labels - generic cleanup
    def shorten_model_name(name: str) -> str:
        name = str(name).split("/")[-1]  # Remove provider prefix
        name = name.replace("-latest", "").replace("-instruct", "")
        name = name.replace("@azure-openai-foundry_", "").replace("@azure-openai-foundry/", "")
        # Capitalize known models
        if "gpt-4" in name.lower():
            name = name.upper().replace(".", "")
        return name[:20]  # Truncate long names

    agg["model_short"] = agg["model"].apply(shorten_model_name)
    agg["label"] = agg["model_short"] + " " + agg["context"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Auto-generate colors for discovered models
    unique_models = agg["model_short"].unique()
    color_palette = sns.color_palette("husl", len(unique_models))
    model_colors = {m: color_palette[i] for i, m in enumerate(unique_models)}

    # Auto-generate markers for discovered contexts
    available_markers = ["s", "^", "D", "o", "v", "<", ">", "p", "h"]
    unique_contexts = sorted(agg["context"].unique())
    context_markers = {c: available_markers[i % len(available_markers)] for i, c in enumerate(unique_contexts)}

    for _, row in agg.iterrows():
        color = model_colors.get(row["model_short"], "#333333")
        marker = context_markers.get(row["context"], "o")

        # Size by baseline F1 (better models = larger points)
        size = 100 + row["baseline_f1"] * 300

        ax.scatter(
            row["delta_content"],
            row["delta_shuffle"],
            c=[color],
            marker=marker,
            s=size,
            alpha=0.8,
            edgecolors="white",
            linewidth=1.5,
        )

        # Label each point
        ax.annotate(
            row["label"],
            (row["delta_content"], row["delta_shuffle"]),
            fontsize=9,
            fontweight="bold",
            xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Reference lines
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    # Encoder baseline zone (Feger et al.: Δ ≤ 0.02)
    ax.axhspan(-0.02, 0.02, alpha=0.15, color="red", label="Encoder zone (Feger: Δ ≤ 0.02)")
    ax.axvspan(-0.02, 0.02, alpha=0.15, color="red")

    # Dynamic quadrant annotations based on data range
    x_min, x_max = agg["delta_content"].min(), agg["delta_content"].max()
    y_min, y_max = agg["delta_shuffle"].min(), agg["delta_shuffle"].max()

    if x_max > 0.05:
        ax.text(x_max * 0.7, y_min * 0.7, "Content helps\nShuffle hurts", fontsize=10, ha="center", alpha=0.6,
                style="italic", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    if x_min < -0.1 and y_min < -0.1:
        ax.text(x_min * 0.7, y_min * 0.7, "Both hurt\n(structure-dependent)", fontsize=10, ha="center", alpha=0.6,
                style="italic", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    # Legend for models
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=12, label=m)
                     for m, c in model_colors.items()]
    context_handles = [Line2D([0], [0], marker=m, color="gray", markersize=10, linestyle="", label=c)
                       for c, m in context_markers.items()]

    legend1 = ax.legend(handles=model_handles, title="Model", loc="upper left", fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=context_handles, title="Context", loc="lower left", fontsize=9)

    ax.set_xlabel("Δ Content-Only (avg F1 change when removing function words)", fontsize=11)
    ax.set_ylabel("Δ Shuffle (avg F1 change when shuffling word order)", fontsize=11)
    ax.set_title("Manipulation Sensitivity by Model × Context\n(averaged across datasets, point size = baseline F1)", fontweight="bold", fontsize=13)

    # Dynamic axis limits with padding
    padding = 0.05
    ax.set_xlim(min(x_min - padding, -0.02 - padding), max(x_max + padding, 0.02 + padding))
    ax.set_ylim(min(y_min - padding, -0.02 - padding), max(y_max + padding, 0.02 + padding))

    plt.tight_layout()
    plt.savefig(V1_DIR / "plot_delta_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {V1_DIR / 'plot_delta_scatter.png'}")

    # Print summary table
    print("\nSummary Table:")
    print(agg[["label", "delta_content", "delta_shuffle", "baseline_f1", "n_datasets"]].round(3).to_string(index=False))

    return agg


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} result files\n")

    # Extract data
    pred_df = extract_predictions(results)
    metrics_df = extract_metrics(results)
    print(f"Predictions: {len(pred_df)} samples")
    print(f"Metrics: {len(metrics_df)} rows\n")

    # Plot 1: Bias Shift
    print("="*60)
    print("PLOT 1: Prediction Bias Shift")
    print("="*60)
    plot_prediction_bias_shift(pred_df)

    # Plot 2: Delta Scatter (Model × Context level)
    print("\n" + "="*60)
    print("PLOT 2: Delta Scatter (Model × Context)")
    print("="*60)
    plot_delta_scatter(metrics_df)
