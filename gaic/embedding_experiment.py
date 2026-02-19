"""
Embedding space analysis for GAIC thesis.

Extracts last-layer hidden states from a causal LM, projects with UMAP,
and measures whether Argument/No-Argument clusters collapse under word shuffling.

Usage:
    uv run gaic/embedding_experiment.py config/experiments/embedding_mistral_7b.toml
"""

import gc
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.paths import PROJECT_ROOT
from gaic.unified_experiment import (
    dataset_from_id,
    load_data,
    sample_balanced,
    shuffle_sentence,
)

MANIPULATIONS = {
    "original": lambda s: s,
    "shuffle": shuffle_sentence,
}


# -- device selection --


def select_device() -> tuple[torch.device, torch.dtype]:
    """Select best available device and compatible dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        # MPS doesn't support bfloat16
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


# -- embedding extraction --


def extract_embeddings(
    model,
    tokenizer,
    sentences: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract last-layer, last-token hidden states for each sentence.

    In causal LMs the last token has attended to all previous tokens,
    making it the richest sentence-level representation.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting embeddings"):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]

        # Index of last real (non-padding) token per batch item
        last_token_idx = attention_mask.sum(dim=1) - 1  # (batch,)

        batch_embeddings = hidden_states[
            torch.arange(hidden_states.size(0), device=device), last_token_idx
        ]  # (batch, hidden_dim)

        all_embeddings.append(batch_embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


# -- separation metric --


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (1 - cosine_similarity)."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(1.0 - dot / norm)


def compute_separation_metrics(
    embeddings: np.ndarray, metadata: list[dict]
) -> dict:
    """Compute centroid cosine distance between Argument/No-Argument per dataset x manipulation."""
    metrics = {}

    datasets = sorted(set(m["dataset"] for m in metadata))
    manipulations = sorted(set(m["manipulation"] for m in metadata))

    for dataset in datasets:
        metrics[dataset] = {}
        for manip in manipulations:
            mask = [
                i
                for i, m in enumerate(metadata)
                if m["dataset"] == dataset and m["manipulation"] == manip
            ]
            if not mask:
                continue

            emb_subset = embeddings[mask]
            labels_subset = [metadata[i]["label"] for i in mask]

            arg_emb = emb_subset[
                [i for i, l in enumerate(labels_subset) if l == "Argument"]
            ]
            noarg_emb = emb_subset[
                [i for i, l in enumerate(labels_subset) if l == "No-Argument"]
            ]

            if len(arg_emb) == 0 or len(noarg_emb) == 0:
                continue

            arg_centroid = arg_emb.mean(axis=0)
            noarg_centroid = noarg_emb.mean(axis=0)
            dist = cosine_distance(arg_centroid, noarg_centroid)
            metrics[dataset][manip] = round(dist, 6)

        # Compute delta (shuffle - original)
        if "original" in metrics[dataset] and "shuffle" in metrics[dataset]:
            metrics[dataset]["delta"] = round(
                metrics[dataset]["shuffle"] - metrics[dataset]["original"], 6
            )

    return metrics


# -- plotting --


def plot_umap_grid(
    umap_coords: np.ndarray,
    metadata: list[dict],
    output_path: Path,
):
    """Create a grid figure: 2 columns (original | shuffled) x N rows (datasets)."""
    datasets = sorted(set(m["dataset"] for m in metadata))
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        n_datasets, 2, figsize=(10, 3 * n_datasets), squeeze=False
    )
    fig.suptitle("UMAP Embedding Space: Original vs Shuffled", fontsize=14, y=1.01)

    colors = {"Argument": "#2196F3", "No-Argument": "#F44336"}
    manip_cols = {"original": 0, "shuffle": 1}

    for row, dataset in enumerate(datasets):
        for manip, col in manip_cols.items():
            ax = axes[row, col]
            mask = [
                i
                for i, m in enumerate(metadata)
                if m["dataset"] == dataset and m["manipulation"] == manip
            ]
            if not mask:
                ax.set_visible(False)
                continue

            coords = umap_coords[mask]
            labels = [metadata[i]["label"] for i in mask]

            for label, color in colors.items():
                label_mask = [i for i, l in enumerate(labels) if l == label]
                if label_mask:
                    ax.scatter(
                        coords[label_mask, 0],
                        coords[label_mask, 1],
                        c=color,
                        label=label,
                        alpha=0.7,
                        s=30,
                        edgecolors="white",
                        linewidths=0.3,
                    )

            if row == 0:
                ax.set_title(manip.capitalize(), fontsize=12)
            if col == 0:
                ax.set_ylabel(dataset, fontsize=10, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])

    # Single legend at bottom
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l
        )
        for l, c in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(str(output_path) + ext, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grid plot saved to {output_path}.png/.pdf")


def plot_umap_combined(
    umap_coords: np.ndarray,
    metadata: list[dict],
    output_path: Path,
):
    """Combined view: all datasets in one plot, 2 subplots (original | shuffled)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("UMAP Embedding Space (All Datasets)", fontsize=14)

    colors = {"Argument": "#2196F3", "No-Argument": "#F44336"}
    manip_titles = {"original": "Original", "shuffle": "Shuffled"}

    for col, (manip, title) in enumerate(manip_titles.items()):
        ax = axes[col]
        mask = [i for i, m in enumerate(metadata) if m["manipulation"] == manip]
        coords = umap_coords[mask]
        labels = [metadata[i]["label"] for i in mask]

        for label, color in colors.items():
            label_mask = [i for i, l in enumerate(labels) if l == label]
            if label_mask:
                ax.scatter(
                    coords[label_mask, 0],
                    coords[label_mask, 1],
                    c=color,
                    label=label,
                    alpha=0.5,
                    s=20,
                    edgecolors="white",
                    linewidths=0.3,
                )

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l
        )
        for l, c in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(str(output_path) + ext, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined plot saved to {output_path}.png/.pdf")


# -- main --


def run(config: dict, config_path: Path | None = None):
    cfg_model = config["model"]
    cfg_umap = config["umap"]
    datasets = config["datasets"]["enabled"]
    sample_size = config["experiment"]["sample_size"]
    experiment_name = config["experiment"].get("experiment_name", "embedding_analysis")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    texts, labels = load_data()

    # -- Collect sentences + metadata --
    logger.info("Collecting sentences across datasets and manipulations...")
    sentences = []
    metadata = []

    for dataset in datasets:
        samples = sample_balanced(texts, labels, dataset, sample_size)
        logger.info(f"{dataset}: {len(samples)} samples")

        for sample_id, sentence, true_label in samples:
            for manip_name, manip_fn in MANIPULATIONS.items():
                manipulated = manip_fn(sentence)
                sentences.append(manipulated)
                metadata.append(
                    {
                        "id": sample_id,
                        "dataset": dataset,
                        "label": true_label,
                        "manipulation": manip_name,
                        "sentence": manipulated,
                    }
                )

    logger.info(f"Total sentences to embed: {len(sentences)}")

    # -- Select device --
    device, dtype = select_device()
    # Override dtype if specified in config
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if "dtype" in cfg_model:
        dtype = dtype_map.get(cfg_model["dtype"], dtype)
        # MPS doesn't support bfloat16
        if device.type == "mps" and dtype == torch.bfloat16:
            dtype = torch.float16
            logger.warning("MPS does not support bfloat16, falling back to float16")
    logger.info(f"Device: {device}, dtype: {dtype}")

    # -- Load model --
    model_name = cfg_model["name"]
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        output_hidden_states=True,
        device_map=device,
    )
    model.eval()

    # -- Extract embeddings --
    batch_size = cfg_model.get("batch_size", 8)
    raw_embeddings = extract_embeddings(model, tokenizer, sentences, batch_size, device)
    logger.info(f"Embeddings shape: {raw_embeddings.shape}")

    # -- Free GPU memory before UMAP --
    del model
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    logger.info("Model unloaded, memory freed")

    # -- UMAP projection --
    logger.info("Running UMAP projection...")
    reducer = umap.UMAP(
        n_neighbors=cfg_umap.get("n_neighbors", 15),
        min_dist=cfg_umap.get("min_dist", 0.1),
        n_components=cfg_umap.get("n_components", 2),
        metric=cfg_umap.get("metric", "cosine"),
        random_state=cfg_umap.get("random_state", 42),
    )
    umap_coords = reducer.fit_transform(raw_embeddings)
    logger.info(f"UMAP projection complete: {umap_coords.shape}")

    # -- Compute separation metrics --
    separation = compute_separation_metrics(raw_embeddings, metadata)
    logger.info("Separation metrics (centroid cosine distance):")
    for dataset, vals in separation.items():
        orig = vals.get("original", "N/A")
        shuf = vals.get("shuffle", "N/A")
        delta = vals.get("delta", "N/A")
        logger.info(f"  {dataset}: original={orig}, shuffle={shuf}, delta={delta}")

    # -- Build output filename stem --
    safe_model = (
        model_name.replace("/", "_").replace(":", "_").replace("@", "").replace(".", "_")
    )
    stem = f"{experiment_name}_{sample_size}_{safe_model}"

    # -- Save NPZ (raw embeddings + UMAP coords) --
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        embeddings=raw_embeddings,
        umap_coords=umap_coords,
    )
    logger.info(f"Embeddings saved to {npz_path}")

    # -- Save JSON (metadata + metrics) --
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path) if config_path else None,
        "config": config,
        "model": model_name,
        "sample_size": sample_size,
        "n_total_embeddings": len(sentences),
        "embedding_dim": int(raw_embeddings.shape[1]),
        "separation_metrics": separation,
        "samples": [
            {
                **m,
                "umap_x": float(umap_coords[i, 0]),
                "umap_y": float(umap_coords[i, 1]),
            }
            for i, m in enumerate(metadata)
        ],
    }
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # -- Generate plots --
    plot_umap_grid(umap_coords, metadata, output_dir / f"{stem}_umap_grid")
    plot_umap_combined(umap_coords, metadata, output_dir / f"{stem}_umap_combined")

    logger.info("Embedding experiment complete.")


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: uv run gaic/embedding_experiment.py <config.toml>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    run(config, config_path=config_path)


if __name__ == "__main__":
    main()
