# Zero-Shot LLMs for Generalizable Argument Identification: Exploiting Context in the GAIC Shared Task

**Student:** Johannes Widera  
**Supervisor:** Marc Feger  
**First Reviewer:** Professor Dr. Martin Mauve  
**Second Reviewer:** Professor Dr. Stefan Dietze  
**Institution:** Heinrich Heine University Düsseldorf  
**Target Venue:** Touché @ CLEF 2026 — Generalizable Argument Identification in Context (GAIC)  
**Date:** April 2026

---

## 1. Introduction and Motivation

### 1.1 The Generalization Problem in Argument Mining

Argument mining systems consistently fail to generalize. Feger et al. [1] demonstrated this at ACL 2025: BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but collapse on cross-dataset transfer (mean F1 = 0.56–0.61). Manipulation experiments revealed the cause — removing stop words, function words, and discourse markers produced negligible performance change (Δ ≤ 0.02). The models had learned to classify based on content words (topic/domain artifacts), not the linguistic structures that define argumentation.

The paper concludes:

> "Benchmarking should therefore build on combined datasets that capture the task's general demands... for which decoder-based argument mining may be of interest." [1]

### 1.2 The GAIC Shared Task: A Direct Response

The GAIC shared task [2], organized by the same research group at Touché @ CLEF 2026, directly addresses this generalization failure. The task objective:

> "Given a sentence from a dataset along with metadata about its provenance, such as the source text and the dataset's annotation guidelines, predict whether the sentence can be annotated as an argument or not. Participants are encouraged to develop robust systems that generalize beyond lexical shortcuts and investigate ways to exploit rich context information for this purpose."

GAIC provides what prior benchmarks lacked: **rich context** including argument definitions extracted from dataset papers, annotation guidelines, and document context (preceding sentences). The task hypothesis is that systems which exploit this context can adapt to each dataset's notion of "argument" without overfitting to dataset-specific lexical patterns.

### 1.3 This Thesis: Testing the GAIC Hypothesis with LLMs

This thesis investigates whether zero-shot decoder-based LLMs can solve the GAIC task by exploiting the provided context. We test three claims:

1. **Context enables generalization** — Zero-shot LLMs with GAIC context can match or approach trained encoder baselines
2. **Decoder LLMs are structurally sensitive** — Unlike encoders, they rely on argumentative structure rather than content-word shortcuts
3. **Results are valid despite contamination** — LLM pre-training data may include benchmark sentences, but this does not explain our findings

---

## 2. Research Questions

### RQ1: Can zero-shot LLMs match trained encoder baselines, and does GAIC context improve performance?

**Motivation:** GAIC provides rich context (definitions, guidelines, document context) specifically to enable cross-dataset generalization. If zero-shot LLMs can exploit this context effectively, they offer a practical alternative to multi-dataset training — no labeled data required, just the metadata GAIC provides.

**Hypothesis:** Adding dataset-specific context will improve F1 over generic prompting. The context ladder (generic → definition → guidelines → document context) will show cumulative gains, with annotation guidelines providing the largest improvement (consistent with GoLLIE [3]).

**Comparison baseline:** Feger et al.'s joint-trained encoders (F1 = 0.63–0.74 on held-out datasets).

### RQ2: Are LLMs more sensitive to argumentative structure than encoders?

**Motivation:** Feger et al. showed encoders ignore argumentative structure (Δ ≤ 0.02 under manipulation). If decoder LLMs show larger performance drops when structure is removed, this indicates they genuinely process argumentation rather than exploiting content-word shortcuts. This would validate the GAIC premise: that structure-aware models can generalize better.

**Hypothesis:** Zero-shot LLMs will show substantially larger Δ than encoders under both:
- **M1 (content-only):** Remove stop words, function words, discourse markers, punctuation
- **M2 (shuffle):** Randomly permute word order

The architectural basis: decoder models use causal attention where word order is a hard constraint, unlike encoders' bidirectional attention which produces largely order-invariant representations [4].

**Comparison baseline:** Feger et al.'s encoder Δ ≤ 0.02.

### RQ3: Does data contamination confound performance and manipulation measurements?

**Motivation:** The 10 GAIC benchmark datasets were published between 2014–2022. LLM pre-training corpora likely include these sentences. If models achieve high F1 through verbatim memorization rather than classification, RQ1 results are invalid. If manipulation sensitivity (RQ2) disappears on contaminated data, the structural sensitivity finding is confounded.

**Hypothesis:** Contamination levels will vary by model and dataset. However, manipulation sensitivity (Δ) will persist even on highly contaminated datasets — because models process structure during generation, not just retrieve memorized labels.

---

## 3. The GAIC Shared Task

### 3.1 Task Definition

Binary sentence-level classification:
- **Input:** Sentence + optional context (definition, guidelines, preceding text)
- **Output:** "Argument" or "No-Argument"
- **Evaluation:** Macro F1 across 10 benchmark datasets + one evaluation-only dataset

### 3.2 Datasets

| Dataset | Domain | Definition | Guidelines | Document Context |
|---------|--------|:----------:|:----------:|:----------------:|
| ABSTRCT | Medical abstracts | ✓ | ✓ | ✓ |
| ARGUMINSCI | Scientific papers | ✓ | ✓ | ✓ |
| PE | Persuasive essays | ✓ | ✓ | ✓ |
| USELEC | Election debates | ✓ | ✓ | ✓ |
| FINARG | Earnings calls | ✓ | — | ✓ |
| SCIARK | Scientific abstracts | ✓ | — | ✓ |
| ACQUA | Legal decisions | ✓ | — | — |
| AEC | Online dialogues | ✓ | — | — |
| AFS | Essays & forums | ✓ | — | — |
| IAM | Online debates | ✓ | — | — |

**Key observation:** Context availability is non-uniform. Only 4 datasets have full context (definition + guidelines + document). This natural variation enables analysis of which context types matter most.

### 3.3 Annotation Scheme Heterogeneity

Each dataset operationalizes "argument" differently:
- **ABSTRCT:** Claims are conclusions about study outcomes; evidence is observations/measurements
- **ACQUA:** Arguments require comparative preference (better/worse)
- **PE:** Hierarchical structure (major claim → claim → premise)
- **USELEC:** Policy stances and judgments requiring justification

A sentence labeled "Argument" in one dataset may be "No-Argument" in another. This heterogeneity is precisely why context matters — models must adapt their classification criteria per dataset.

---

## 4. Methodology

### 4.1 Experimental Design Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RQ1: Performance                        │
│  Context Ladder: C0 → C1 → C2 → C3                             │
│  Compare to: Feger's trained encoders (F1 = 0.63-0.74)         │
├─────────────────────────────────────────────────────────────────┤
│                      RQ2: Structural Sensitivity                │
│  Manipulations: M0 (original) vs M1 (content-only) vs M2 (shuffle) │
│  Compare to: Feger's encoder Δ ≤ 0.02                          │
├─────────────────────────────────────────────────────────────────┤
│                      RQ3: Contamination                         │
│  DCQ sentence completion test per model × dataset              │
│  Analyze: Does contamination explain RQ1/RQ2 findings?         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Models

| Model | Parameters | Role |
|-------|------------|------|
| Ministral 8B | 8B | Small-scale, low contamination baseline |
| Mistral Small 24B | 24B | Mid-scale, pilot model |
| Mistral Medium | ~70B | Large-scale Mistral |
| GPT-4.1 / GPT-5.2 | Frontier | Cross-provider validation |

All models are instruction-tuned (non-reasoning) for clean comparison with Feger's encoder setup.

### 4.3 Context Conditions (RQ1)

| Condition | Context in Prompt | Tests |
|-----------|-------------------|-------|
| C0 | Generic: "An argument is a claim supported by reasoning..." | Baseline LLM knowledge |
| C1 | Dataset-specific definition (extracted from paper) | Does aligning to dataset theory help? |
| C2 | C1 + annotation guidelines | Do operational rules add value? |
| C3 | C2 + 2 preceding sentences | Does local discourse help? |

Context is cumulative: C3 includes everything from C0–C2.

### 4.4 Text Manipulations (RQ2)

| Manipulation | Description | What It Tests |
|--------------|-------------|---------------|
| M0: Original | Unmodified sentence | Baseline |
| M1: Content-only | Remove stopwords, function words, discourse markers, punctuation | Reliance on argumentative scaffolding |
| M2: Shuffle | Random word permutation (seed=42) | Reliance on word order / syntax |

**Delta metrics:**
- Δ_M1 = F1(M0) − F1(M1)
- Δ_M2 = F1(M0) − F1(M2)

Negative deltas indicate the model relied on the removed/disrupted features.

### 4.5 Contamination Testing (RQ3)

Following Golchin & Surdeanu [5]:

1. Split each sentence at midpoint
2. Prompt model: "Complete this sentence from [domain]: {first_half}"
3. Compute ROUGE-L between generated completion and actual second half
4. Aggregate per model × dataset

**Contamination thresholds:**
| ROUGE-L | Level |
|---------|-------|
| > 0.5 | HIGH |
| 0.3–0.5 | MODERATE |
| < 0.3 | LOW |

### 4.6 Data Splits

- **Dev split:** All RQ1/RQ2 experiments (~170 sentences per class per dataset)
- **Test split:** Reserved for final evaluation and GAIC submission
- **Train split:** Not used (zero-shot only)

---

## 5. Preliminary Results

### 5.1 Context Improves Performance (RQ1)

**Setup:** Mistral Small 24B, all 10 datasets, n=30/dataset

| Condition | Mean Macro F1 |
|-----------|---------------|
| C0: Generic definition | 0.58 |
| C1: Dataset-specific definition | 0.63 |

Dataset-specific definitions improve F1 by +0.05 on average. This already approaches Feger's joint-trained encoder baseline (0.63).

### 5.2 LLMs Show High Structural Sensitivity (RQ2)

**Setup:** Mistral Small 24B, ABSTRCT, n=100

| Manipulation | Macro F1 | Δ from M0 |
|--------------|----------|-----------|
| M0: Original | 0.72 | — |
| M1: Content-only | 0.62 | −0.10 |
| M2: Shuffle | 0.43 | −0.29 |

**Comparison to encoders:**
- Δ_M1 = 0.10 → **5× the encoder effect** (Feger: Δ ≤ 0.02)
- Δ_M2 = 0.29 → **14× the encoder effect**

LLMs are dramatically more sensitive to argumentative structure than encoders.

### 5.3 High Contamination in Large Models (RQ3)

| Model | Contamination (ROUGE-L) |
|-------|-------------------------|
| Mistral Medium | 50–80% (HIGH) |
| GPT-5.2 | 50–75% (HIGH) |
| Ministral 8B | 20–30% (LOW) |

**Critical observation:** Despite high contamination, manipulation still causes large Δ. This suggests:
- Models are not doing verbatim retrieval
- They still process structure during generation
- Contamination inflates F1 but does not eliminate structural sensitivity

---

## 6. Expected Contributions

### Empirical
- First systematic evaluation of zero-shot LLMs on the GAIC task
- Evidence that decoder LLMs are structurally sensitive (unlike encoders)
- Quantification of how GAIC context types affect performance
- Contamination analysis across models and datasets

### Methodological
- Context ladder framework for zero-shot argument identification
- Contamination-aware interpretation of LLM benchmark results

### Practical
- Competitive GAIC 2026 submission
- Reproducible experimental codebase

---

## 7. Scope

### Core Requirements
- Zero-shot context ladder evaluation (RQ1)
- Manipulation experiments across models (RQ2)
- Contamination analysis (RQ3)
- Integration of findings: what do contamination + manipulation results together tell us?

### Out of Scope (for this thesis)
- Fine-tuning experiments
- Layer-wise probing / mechanistic analysis
- Reasoning models (chain-of-thought)

---

## 8. Timeline

| Phase | Weeks | Activities |
|-------|-------|------------|
| Complete RQ1/RQ2 experiments | 1–2 | Full context ladder × manipulation × all models |
| RQ3 contamination analysis | 3 | DCQ tests, correlation with performance |
| GAIC submission | 4 | Best system to TIRA (deadline: 7 May) |
| Results analysis + writing | 5–10 | Chapters 4–5, integrate findings |
| Thesis completion | 11–16 | Full draft, revision, defense prep |
| **Submission** | — | **11 August 2026** |

---

## 9. Thesis Structure

| Chapter | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 4 | Motivation, GAIC task, contributions |
| 2. Background | 8 | Argument mining, shortcut learning, LLMs, contamination |
| 3. The GAIC Shared Task | 4 | Task definition, datasets, context availability |
| 4. Methodology | 6 | Context ladder, manipulations, contamination testing |
| 5. Results | 12 | RQ1 (performance), RQ2 (sensitivity), RQ3 (contamination) |
| 6. Discussion | 6 | Integration, comparison to encoders, limitations |
| 7. Conclusion | 2 | Summary, implications, future work |

---

## 10. References

[1] Feger, M., Boland, K., & Dietze, S. (2025). Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments. *ACL 2025*, 23900–23915.

[2] GAIC Shared Task. (2026). Generalizable Argument Identification in Context. *Touché @ CLEF 2026*. https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html

[3] Sainz, O., et al. (2024). GoLLIE: Annotation guidelines improve zero-shot information extraction. *ICLR 2024*.

[4] Sinha, K., et al. (2021). Unnatural language inference. *ACL 2021*.

[5] Golchin, S., & Surdeanu, M. (2024). Time travel in LLMs: Tracing data contamination in large language models. *ICLR 2024*.

[6] Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.
