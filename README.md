# GAIC Thesis

Codebase for the master thesis **"Beyond Shortcuts: Can Large Language Models Generalize Arguments in Context?"** at Heinrich Heine University Düsseldorf. This work participates in the [Touché @ CLEF 2026 Shared Task: Generalizability of Argument Identification in Context (GAIC)](https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html). The goal is to investigate whether decoder-based LLMs can overcome the shortcut learning problem identified in encoder models (Feger et al., 2025) and develop robust argument identification systems that generalize across datasets. The experiments explore zero-shot classification, dataset-specific annotation guidelines, and cross-dataset transfer evaluation.

## Run Experiments

Zero-shot classification (generic prompt):

```bash
uv run python gaic/zero_shot.py
```

Zero-shot with dataset-specific guidelines:

```bash
uv run python gaic/zero_shot_with_guidelines.py
```

Results are saved to `experiments/zero_shot_outputs/`.

## Configuration

- Edit `SAMPLE_SIZE_PER_DATASET` in the scripts to adjust sample size
- Requires a local LLM server (e.g., Ollama) at `http://localhost:11434/v1`
