"""
DCQ (Data Contamination Quiz) Module

Three-phase contamination detection for LLMs based on Golchin & Surdeanu (TACL 2025).

Usage:
    # Phase 1: Generate perturbations (run once)
    uv run gaic/dcq/experiment.py generate config/experiments/dcq/perturbator.toml

    # Phase 2: BDQ for specific model
    uv run gaic/dcq/experiment.py bdq config/experiments/dcq/perturbator.toml config/experiments/dcq/gpt52.toml

    # Phase 3: BCQ for specific model
    uv run gaic/dcq/experiment.py bcq config/experiments/dcq/perturbator.toml config/experiments/dcq/gpt52.toml
"""
