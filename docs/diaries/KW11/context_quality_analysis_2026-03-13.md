# Context Quality Analysis: 2026-03-13

## 1. Context Quality Evaluation Experiment

Ran LLM-as-judge evaluation to assess extracted definition and guideline quality.

**Model**: Claude Opus 4.6 (via Portkey)
**Sample size**: 10 per dataset (5 Argument, 5 No-Argument)

### Task A: Definition Quality (Label Explainability)

| Dataset | Mean Score | Derivability Rate |
|---------|------------|-------------------|
| ACQUA | 4.0 | - |
| IAM | 4.0 | - |
| SCIARK | 3.9 | - |
| AFS | 3.8 | - |
| FINARG | 3.8 | - |
| ARGUMINSCI | 3.6 | - |
| PE | 3.6 | - |
| AEC | 3.5 | - |
| **ABSTRCT** | **3.4** | - |
| **USELEC** | **3.1** | - |

**Overall**: Mean 3.67/5, Derivability 62%

### Task B: Definition-Guideline Alignment (4 datasets with guidelines)

| Dataset | Alignment Score |
|---------|-----------------|
| ARGUMINSCI | 4 |
| PE | 4 |
| USELEC | 4 |
| **ABSTRCT** | **2** |

---

## 2. Critical Finding: ABSTRCT Paper vs Guidelines Contradiction

### Verified by Reading Original PDFs

**Paper Definition (ABSTRCT.pdf)**:
> "Evidence is defined as an observation or measurement in the study, which supports or attacks another argument component"

The paper treats **Evidence = Argumentative Component**.

**Annotation Guidelines (ABSTRCT-Guidelines.pdf)**:
> "Whereas the reporting of experimental observations is a fact/premise, the general comparison in solitude has no factual value and is therefore a statement/claim"
> "numerical comparisons or significance statements are **not** arguments unless they are abstracted into a general claim"

The guidelines treat **Evidence/Premises = Non-Argumentative**.

### Example from GAIC Data

| Sample | Paper Definition | Guideline | GAIC Label |
|--------|------------------|-----------|------------|
| "A target IOP of 21 mm Hg...achieved in 53 of 56 (95%)..." | Evidence → Argument | Factual reporting | **No-Argument** |
| "Grade 3 or 4 neutropenia was more frequent...P = .007" | Evidence → Argument | Statistical comparison | **No-Argument** |

### Verdict

- **Extraction is correct** — definition.md and guideline.md faithfully represent sources
- **The problem is real** — Paper's theoretical definition contradicts annotation practice
- **GAIC labels follow guidelines**, not the paper definition
- This explains ABSTRCT's low Task B alignment score (2/5)

---

## 3. Implications for Ablation Study

This finding enables a **context quality ablation**:

| Condition | Hypothesis for ABSTRCT |
|-----------|------------------------|
| C0: No context | Baseline |
| C1: Definition only | May **hurt** (definition says evidence=argument, labels disagree) |
| C2: Guideline only | Should **help** (matches actual labeling practice) |
| C3: Definition + Guideline | Conflict may confuse model |

### Dataset Groupings by Definition Quality

**High-quality** (mean >= 4.0): ACQUA, IAM
**Medium-quality** (3.5-3.9): AEC, AFS, ARGUMINSCI, FINARG, SCIARK, PE
**Low-quality** (< 3.5): ABSTRCT, USELEC

---

## 4. Files Created

- `experiments/context_quality/analyze_context_quality.ipynb` — Visualization notebook
- `experiments/context_quality/context_quality_eval_*.json` — Raw results

---

## Next Steps

1. [ ] Run ablation: c0 vs c1 (def) vs c2 (guide) vs c3 (both) on ABSTRCT
2. [ ] Correlate v1 experiment results with definition quality scores
3. [ ] Test hypothesis: guideline-only context outperforms definition-only for ABSTRCT
4. [ ] Write thesis section on context quality as confounding variable
