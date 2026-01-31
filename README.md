# Beyond Shortcuts: Can LLMs Generalize Arguments in Context?

**Master's Thesis** | Heinrich Heine University Düsseldorf | [Touché @ CLEF 2026 GAIC](https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html)

Investigating whether decoder-based LLMs overcome shortcut learning in argument identification, building on [Feger et al. (ACL 2025)](https://aclanthology.org/2025.acl-long.1280/).

## Thesis Methodology

| Part            | Question                                                  |
| --------------- | --------------------------------------------------------- |
| **1. Diagnose** | Do LLMs rely on shortcuts?                                |
| **2. Measure**  | Do LLMs utilize context (guidelines, documents)?          |
| **3. Improve**  | Can discriminative classification improve generalization? |

## Quick Links

- **[Experiments Summary](experiments/experiments_summary.md)** — Results and analysis
- **[Dataset Overview](experiments/dataset_overview.md)** — 10 benchmark datasets

## Experiments

| Experiment | Command | Thesis Part | Purpose |
|------------|---------|-------------|---------|
| Zero-Shot Baseline | `uv run python gaic/zero_shot.py` | Part 1 | Establish LLM baseline without guidance |
| With Guidelines | `uv run python gaic/zero_shot_with_guidelines.py` | Part 2 | Test if annotation guidelines improve accuracy |
| Cross-Guideline | `uv run python gaic/cross_guideline_experiment.py` | Part 2 | Test guideline transfer to datasets without native guidelines |
| Shuffle Test | `uv run python gaic/shuffle_experiment.py` | Part 1 | Detect shortcut learning via word-order destruction |

Requires: Python 3.13+, local LLM at `localhost:11434` (Ollama)

## Project Structure

```
gaic/           # Source code
experiments/    # Results and analysis notebooks
config/         # Path configuration
data/           # GAIC-2026 dataset (not tracked)
docs/           # Thesis documents
```
