# Experiment Diary: 2026-03-11

## 1. Inference Experiments (Zero-shot)

Ran additional context conditions and new models.

| Model | Context | Mean F1 | Δ Content-Only | Δ Shuffle |
|-------|---------|---------|----------------|-----------|
| GPT-4.1 | c0 (zero-shot) | 0.602 | -0.080 | -0.279 |
| GPT-4.1 | c2 (def+guide) | 0.725 | -0.181 | -0.249 |
| Mistral-medium | c0 | 0.537 | -0.067 | -0.161 |
| Mistral-medium | c2 | 0.675 | -0.165 | -0.253 |
| Ministral-8B | c1 (definition) | 0.593 | -0.113 | -0.250 |
| Qwen 2.5-7B | c1 | 0.605 | -0.058 | -0.118 |

### Findings

- **Context helps**: c0→c2 improves F1 by ~0.12 (GPT-4.1: 0.60→0.73, Mistral-medium: 0.54→0.68)
- **All models show expected negative deltas**: Performance drops when text is manipulated
- **Shuffle has larger impact than content-only**: Models rely on word order more than function words
- **Qwen 2.5-7B most robust**: Smallest deltas (-0.06, -0.12) but also lower F1

---

## 2. Finetuning Experiment

### Setup
- **Model**: Ministral-8B-Instruct-2410 with LoRA (r=16, alpha=32)
- **Training**: ABSTRCT only, 100 samples, 3 epochs
- **Evaluation**: All 10 datasets with manipulation test

### Results

| Dataset | Original | Content-Only | Shuffle | Δ_CO | Δ_Sh |
|---------|----------|--------------|---------|------|------|
| **ABSTRCT** (in-domain) | 0.792 | 0.899 | 0.642 | **+0.107** | -0.150 |
| ACQUA | 0.524 | 0.436 | 0.487 | -0.088 | -0.037 |
| AEC | 0.333 | 0.436 | 0.436 | +0.102 | +0.102 |
| AFS | 0.333 | 0.333 | 0.436 | +0.000 | +0.102 |
| ARGUMINSCI | 0.744 | 0.744 | 0.850 | +0.000 | +0.105 |
| FINARG | 0.436 | 0.560 | 0.436 | +0.125 | +0.000 |
| IAM | 0.436 | 0.436 | 0.451 | +0.000 | +0.015 |
| PE | 0.524 | 0.436 | 0.487 | -0.088 | -0.037 |
| SCIARK | 0.670 | 0.670 | 0.524 | +0.000 | -0.146 |
| USELEC | 0.524 | 0.310 | 0.333 | -0.213 | -0.191 |

### Critical Finding: Extreme Shortcut Learning

**Content-only IMPROVES in-domain performance (+0.107)**

This is the opposite of expected behavior. When semantic content is stripped:
- Expected: F1 drops (model needs meaning)
- Observed: F1 goes UP (model uses surface patterns)

**Interpretation**: The model learned "what ABSTRCT text looks like" not "what an argument is"
- Function words, punctuation, sentence structure → dataset fingerprint
- Stripping content words makes that fingerprint clearer
- This is dataset artifact memorization, not argument identification

### Transfer Performance
- In-domain (ABSTRCT): 0.792 F1
- Out-of-domain average: ~0.47 F1 (near random)
- The model cannot generalize to other datasets

### Implication for Thesis
Single-dataset finetuning **amplifies** shortcut learning rather than reducing it. This is a valuable negative result for RQ3.

---

## 3. Infrastructure Created

- `gaic/finetuning/eval/experiment.py` - Evaluation script for finetuned models
- `gaic/finetuning/eval/__init__.py` - Module init
- `config/experiments/finetuned/ministral_8b_ABSTRCT.toml` - Eval config
- `experiments/finetuned/eval.ipynb` - Analysis notebook with cross-dataset transfer matrix

---

## Next Steps

1. [ ] Run eval notebook to generate transfer matrix plots
2. [ ] Compare finetuned vs base model zero-shot on ABSTRCT
3. [ ] Try multi-dataset finetuning (train on 5, test on 5)
4. [ ] Consider adversarial training with manipulated samples
