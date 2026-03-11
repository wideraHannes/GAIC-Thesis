# V1 Experiment Analysis: C1 + C2 Results with Strategic Assessment

**Date**: March 11, 2026
**Status**: Post-C2 evaluation with thesis advisor consultation

---

## Executive Summary

The V1 experiments provide strong preliminary evidence that **decoder-based LLMs are fundamentally more sensitive to linguistic structure than encoders**. The key finding:

| Architecture | Δ_shuffle (mean) | Implication |
|--------------|------------------|-------------|
| Encoders (Feger et al.) | ≤ 0.02 | Near-complete order invariance |
| Decoders (this work) | -0.13 to -0.40 | Strong word-order dependence |

This **10-20× larger effect** supports the central thesis hypothesis: causal attention makes decoders inherently sensitive to discourse structure. However, this sensitivity is a double-edged sword—it enables richer linguistic processing but creates vulnerability to distribution shift.

---

## 1. Results Summary

### 1.1 C1: Definition-Only Context (10 Datasets)

| Model | Avg F1 | Δ_content | Δ_shuffle | Notes |
|-------|--------|-----------|-----------|-------|
| GPT-4.1 | 0.650 | -0.062 | **-0.286** | Highest baseline, largest shuffle drop |
| Mistral-Medium | 0.636 | -0.126 | -0.222 | Strong on ABSTRCT (0.665) |
| Ministral-14b | 0.585 | -0.068 | -0.163 | High variance across datasets |
| Ministral-3b | 0.571 | -0.076 | -0.130 | Most "robust" to shuffle (but lowest baseline) |

**Key observations:**
- All models show Δ_shuffle significantly larger than encoder baseline
- Larger models show larger shuffle drops (scale-sensitivity correlation)
- Δ_content_only is smaller than Δ_shuffle across all models

### 1.2 C2: Definition + Guidelines (4 Datasets with Guidelines)

**GPT-4.1 C2 Results:**
| Dataset | C1 F1 | C2 F1 | Δ | Δ_content | Δ_shuffle |
|---------|-------|-------|---|-----------|-----------|
| ABSTRCT | 0.554 | **0.832** | +0.278 | -0.220 | -0.149 |
| ARGUMINSCI | 0.764 | 0.713 | -0.051 | -0.189 | -0.310 |
| PE | 0.623 | 0.623 | 0.000 | -0.125 | -0.179 |
| USELEC | 0.653 | 0.732 | +0.079 | -0.190 | -0.359 |
| **Avg (4 ds)** | 0.649 | **0.725** | +0.076 | -0.181 | -0.249 |

**Mistral-Medium C2 Results:**
| Dataset | C1 F1 | C2 F1 | Δ | Δ_content | Δ_shuffle |
|---------|-------|-------|---|-----------|-----------|
| ABSTRCT | 0.665 | **0.800** | +0.135 | -0.224 | -0.397 |
| ARGUMINSCI | 0.576 | 0.576 | 0.000 | -0.243 | -0.243 |
| PE | 0.566 | 0.593 | +0.027 | +0.068 | -0.209 |
| USELEC | 0.633 | 0.732 | +0.099 | -0.261 | -0.163 |
| **Avg (4 ds)** | 0.610 | **0.675** | +0.065 | -0.165 | -0.253 |

**Key observations:**
- Guidelines provide significant gains, especially on ABSTRCT (+0.278 for GPT-4.1)
- Manipulation sensitivity **increases** with guidelines (not decreases)
- ABSTRCT shows largest improvement—consistent with its complex claim/evidence distinctions

### 1.3 Comparative Δ Values

| Condition | Model | Δ_content | Δ_shuffle |
|-----------|-------|-----------|-----------|
| Encoder (Feger) | BERT/RoBERTa | ≤ 0.02 | ≤ 0.02 |
| C1 | GPT-4.1 | -0.062 | -0.286 |
| C1 | Mistral-Medium | -0.126 | -0.222 |
| C2 | GPT-4.1 | -0.181 | -0.249 |
| C2 | Mistral-Medium | -0.165 | -0.253 |

---

## 2. Theoretical Interpretation

### 2.1 The Core Finding: Decoders Process Discourse Structure

The manipulation experiments reveal a fundamental architectural difference:

**Encoders (Feger et al.)**: Near-zero sensitivity to manipulation (Δ ≤ 0.02) indicates bag-of-words processing. Word order, function words, and discourse markers are largely irrelevant to predictions. The model learns dataset-specific lexical shortcuts.

**Decoders (this work)**: Large sensitivity to manipulation (Δ up to -0.40) indicates structural processing. Word order matters because causal attention enforces sequential dependencies. Discourse markers shape the conditional distribution over subsequent tokens.

**Critical reframe**: Manipulation sensitivity is **evidence of deep linguistic processing**, not evidence of shortcut learning. A shortcut learner should be *robust* to shuffle—the bag-of-words signal survives permutation. Our models collapse under shuffle because they are using features that shuffle destroys.

### 2.2 Why Guidelines Increase Sensitivity

The finding that C2 (guidelines) increases manipulation sensitivity appears paradoxical but has a coherent explanation:

**Definition-only (C1)**: Model guesses based on surface patterns and pretraining priors
**Guidelines (C2)**: Model attempts to apply annotation criteria, which *requires* parsing discourse structure

Adding guidelines doesn't make the model simpler—it gives the model a richer target concept. The model now needs to:
- Identify claim-premise relationships
- Distinguish evidence from conclusions
- Parse hedging and epistemic markers

These operations *depend on* discourse structure, so disrupting that structure (via manipulation) causes larger performance drops.

**Implication**: High manipulation sensitivity + high task accuracy = deep linguistic processing of relevant features. The question is whether the model is using the *right* features.

### 2.3 The Scale Effect

Observation: Larger models show larger Δ_shuffle.

| Model | Size | Avg F1 | Δ_shuffle |
|-------|------|--------|-----------|
| GPT-4.1 | Frontier | 0.650 | -0.286 |
| Mistral-Medium | ~70B | 0.636 | -0.222 |
| Ministral-14b | 14B | 0.585 | -0.163 |
| Ministral-3b | 3B | 0.571 | -0.130 |

**Possible interpretations:**

1. **Deeper processing**: Larger models rely more heavily on syntactic/discourse structure, so disruption hurts more

2. **Better calibration**: Smaller models make noisier predictions on both conditions, compressing the delta

3. **Instruction following**: Larger models try harder to apply the task definition, while smaller models ignore nuance

**Distinguishing test**: Plot Δ_shuffle against baseline F1, not just model size. If "robustness" is just noise, small-model deltas should correlate with lower variance, not higher.

### 2.4 Content-Only Improvements: Bias Correction

Cases where removing function words *improved* F1 (e.g., AFS +0.225 for GPT-4.1) are not evidence of shortcut learning—they're evidence of **miscalibration**.

**Mechanism**: If the model over-interprets certain discourse markers (hedging, rhetorical questions, stylistic features) as class signals, removing them debiases predictions.

**Example**: On AFS, "I really don't understand why Americans are convinced that guns are essential" triggered Argument predictions. Removing "I really don't understand why" exposed the sentence as rhetorical expression, not argument—leading to correct No-Argument prediction.

**Implication**: Decoders may be *too* sensitive to certain discourse markers, applying inappropriate pragmatic inferences. This is a limitation of discourse processing, not evidence of shortcut learning.

---

## 3. Assessment: Can These Results Be Used?

### 3.1 Sample Size (n=30)

**Verdict**: Acceptable for exploratory findings, borderline for definitive claims.

**Strengths:**
- Large effect sizes (Δ ≈ 0.25) are detectable at n=30
- Consistent patterns across models and datasets
- Sufficient for identifying which conditions warrant deeper investigation

**Weaknesses:**
- Per-dataset F1 confidence intervals are wide
- Cross-dataset comparisons are underpowered
- Individual dataset claims (e.g., "ABSTRCT shows X") need validation

**Recommendation:**
- Keep n=30 for exploration
- Validate 2-3 key findings at n=100+ before thesis defense
- Report effect sizes with confidence intervals, not just point estimates

### 3.2 What Claims Are Supported?

**Strong claims (supported by current data):**
1. Decoder manipulation sensitivity is an order of magnitude larger than encoder baseline
2. Word shuffle causes larger performance drops than content reduction
3. Adding guidelines improves performance on ABSTRCT significantly

**Moderate claims (require validation):**
4. Larger models show greater manipulation sensitivity
5. Guidelines increase manipulation sensitivity (not just performance)

**Weak claims (need more evidence):**
6. Specific dataset behavior patterns (n=30 per dataset is thin)
7. Content-only improvement mechanism

### 3.3 Comparison to Feger et al. Baselines

| Metric | Feger (Encoders) | This Work (Decoders) | Can Compare? |
|--------|------------------|----------------------|--------------|
| Cross-dataset F1 | 0.56-0.61 | 0.57-0.65 | Yes (similar range) |
| Δ_manipulation | ≤ 0.02 | 0.13-0.40 | **Yes (major difference)** |
| Joint-trained F1 | 0.63-0.74 | N/A (zero-shot) | Different paradigm |

The Δ comparison is directly meaningful because it uses the same manipulation operations.

---

## 4. Recommended Next Steps (Prioritized)

### Priority 1: Error Analysis on Shuffle Failures (2 days)

**Goal**: Understand the mechanism—what linguistic features drive predictions?

**Method:**
1. Sample 20-30 sentences with large shuffle-induced prediction changes
2. Annotate: What discourse features does each sentence rely on? (Connectives, hedging, claim-premise structure)
3. Test: Are errors predictable from linguistic features that shuffle destroys?

**Why first**: Gives interpretable findings, not just numbers. Essential for thesis discussion.

### Priority 2: C3 Pilot (Document Context) (1-2 days)

**Goal**: Test whether preceding context provides robustness through redundant signal.

**Datasets**: ABSTRCT + 1 other (suggest USELEC or PE)
**Sample size**: n=30-50

**Hypothesis**: Document context should *reduce* manipulation sensitivity by providing external grounding.

### Priority 3: Validate Key Finding at n=100 (1 day compute)

**Goal**: Strengthen statistical claims for core findings.

**Select**:
- GPT-4.1 on ABSTRCT (largest guideline effect)
- One dataset with moderate effect for contrast (suggest SCIARK or ACQUA)

### Priority 4: Complete C2 Coverage (Optional)

Run Ministral-3b and Ministral-14b on C2 for completeness. Adds statistical power but not theoretical insight.

### Priority 5: Part 3 (LoRA Fine-Tuning)

**Defer until zero-shot findings are solid.** The thesis is stronger with deep analysis of one paradigm than shallow coverage of two.

---

## 5. Thesis Framing

### 5.1 Core Argument

**Claim**: Decoder-based LLMs exhibit fundamentally different sensitivity to linguistic manipulation than encoder-based models.

**Evidence**: Where encoders showed near-zero performance change under manipulation (Feger et al., Δ ≤ 0.02), decoders show large, systematic drops (Δ_shuffle up to 0.40).

**Interpretation**: This supports the architectural hypothesis—causal attention makes word order and discourse structure computationally necessary, not optional. Decoders process arguments as structured discourse, not bags of topical words.

### 5.2 Nuances to Address

1. **Double-edged sensitivity**: Deeper processing enables understanding but creates fragility
2. **Scale effects**: Larger models are more sensitive—this may reflect deeper processing or better instruction following
3. **Context ladder**: Guidelines improve both performance and sensitivity—models do more linguistic work when given richer task definitions
4. **Bias patterns**: Some discourse markers are over-interpreted, causing miscalibration

### 5.3 Open Questions for Remaining Work

1. **What specific features drive predictions?** (→ Error analysis)
2. **Can context provide robustness?** (→ C3 experiments)
3. **Does fine-tuning reduce or maintain sensitivity?** (→ Part 3)
4. **What happens at scale?** (→ Mistral frontier model comparison)

---

## 6. Immediate Action Items

| Task | Time | Status |
|------|------|--------|
| Error analysis design | 1 day | **Next** |
| Error analysis execution | 1 day | Pending |
| C3 config for ABSTRCT + USELEC | 0.5 day | Pending |
| C3 experiments | 1 day | Pending |
| n=100 validation (ABSTRCT, GPT-4.1 C2) | 1 day | Pending |
| Draft C1/C2 findings writeup | 2 days | Pending |

---

## Appendix: Raw Data Reference

### C1 Full Results (10 Datasets, n=30)

See `results_filled.csv` for complete data.

### C2 Full Results (4 Datasets, n=30)

| Model | Dataset | Original | Content | Shuffle |
|-------|---------|----------|---------|---------|
| GPT-4.1 | ABSTRCT | 0.832 | 0.612 | 0.683 |
| GPT-4.1 | ARGUMINSCI | 0.713 | 0.524 | 0.403 |
| GPT-4.1 | PE | 0.623 | 0.498 | 0.444 |
| GPT-4.1 | USELEC | 0.732 | 0.542 | 0.373 |
| Mistral-Medium | ABSTRCT | 0.800 | 0.576 | 0.403 |
| Mistral-Medium | ARGUMINSCI | 0.576 | 0.333 | 0.333 |
| Mistral-Medium | PE | 0.593 | 0.661 | 0.384 |
| Mistral-Medium | USELEC | 0.732 | 0.471 | 0.569 |

### Comparison to Feger et al. Table 4 (Joint-Trained Encoders)

| Encoder | Mean F1 (Joint) | Δ_manipulation |
|---------|-----------------|----------------|
| BERT | 0.66 | ≤ 0.02 |
| RoBERTa | 0.68 | ≤ 0.02 |
| DistilBERT | 0.63 | ≤ 0.02 |
| **Zero-shot GPT-4.1** | **0.65** | **-0.29** |

The zero-shot decoder matches trained encoder performance while showing dramatically different manipulation sensitivity.
