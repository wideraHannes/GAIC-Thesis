# DCQ Configuration Files

Configuration files for the Data Contamination Quiz (DCQ) experiment.

## Files

- **`perturbator.toml`** - Base configuration for all phases (perturbation generation, sampling, datasets)
- **`gpt52.toml`** - Model config for GPT-5.2 (2025-12-11)
- **`mistral_medium.toml`** - Model config for Mistral Medium
- **`ministral_8b.toml`** - Model config for Ministral 8B

## Usage

Each phase requires the base `perturbator.toml`. Phases 2 and 3 additionally need a model-specific config:

```bash
# Phase 1: Generate perturbations (run once for all models)
uv run gaic/dcq/experiment.py generate config/experiments/dcq/perturbator.toml

# Phase 2: BDQ (per model)
uv run gaic/dcq/experiment.py bdq config/experiments/dcq/perturbator.toml config/experiments/dcq/gpt52.toml

# Phase 3: BCQ (per model)
uv run gaic/dcq/experiment.py bcq config/experiments/dcq/perturbator.toml config/experiments/dcq/gpt52.toml
```

## Directory Structure

Results are saved to `experiments/dcq/`:

```
experiments/dcq/
├── phase1_perturbations/
│   └── perturbations.jsonl
├── phase2_bdq/
│   ├── gpt-5.2-2025-12-11/
│   │   ├── bdq_results.jsonl
│   │   └── bias_summary.json
│   └── ...
└── phase3_bcq/
    ├── gpt-5.2-2025-12-11/
    │   ├── bcq_results.jsonl
    │   └── contamination_report.json
    └── ...
```
