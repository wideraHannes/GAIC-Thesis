# Beyond Shortcuts: Can Large Language Models Generalize Arguments in Context?

## A Follow-Up Investigation to "Limited Generalizability in Argument Mining"

---

**Student**: Johannes Widera  
**Supervisor:** Marc Feger  
**Institution:** Heinrich Heine University Düsseldorf  
**Date:** January 2026  
**Target Venue:** Touché @ CLEF 2026, _Generalizable Argument Identification in Context (GAIC)_  
**Thesis Reviewers:**

1. Professor Dr. Martin Mauve
2. Professor Dr. Stefan Dietze

---

## 1. Introduction and Motivation

Feger, Boland, and Dietze (2025) presented the first large-scale re-evaluation of argument mining benchmarks at ACL 2025, with a troubling conclusion: **state-of-the-art models learn datasets, not arguments**. Their comprehensive study across 17 benchmark datasets revealed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but fail dramatically on cross-dataset transfer (mean F1 = 0.56–0.61). Controlled manipulation experiments—removing stop words, function words, and discourse markers—showed almost no performance change for these models (Δ ≤ 0.02), demonstrating that they rely on content-word shortcuts rather than the linguistic scaffolding that theoretically defines argumentation.

The paper concludes with a forward-looking statement:

> "Benchmarking should therefore build on combined datasets that capture the task's general demands, as in GLUE and instruction-tuning benchmarks, **for which decoder-based argument mining may be of interest**." (Feger et al., 2025, p. 23907)

This thesis directly addresses this open direction. The GAIC shared task—organized by the same research group—now provides the infrastructure to investigate decoder-based models (LLMs) under the same rigorous generalization framework. GAIC explicitly encourages participants to:

1. **Develop robust systems that generalize beyond lexical shortcuts**
2. **Investigate ways to exploit rich context information** (annotation guidelines, source documents)

This creates a unique opportunity: to determine whether LLMs overcome the shortcut learning problem identified in encoders, or whether they simply learn different shortcuts at a larger scale.

---

## 2. The Shortcut Learning Problem

### 2.1 Theoretical Foundation

Geirhos et al. (2020) define shortcuts as "decision rules that perform well on i.i.d. test data but fail on o.o.d. tests." All learning systems exhibit some form of shortcut learning—humans memorizing patterns for exams, animals solving mazes via unintended cues. **The issue is not the existence of shortcuts, but that they break under distribution shifts**—precisely what cross-dataset argument mining evaluates.

### 2.2 Key Findings from Feger et al. (2025)

The ACL paper established several critical findings that this thesis builds upon:

**Finding 1: Encoders do not learn argument structure.** After removing ~50% of words (stop words, function words, discourse markers, punctuation), BERT, RoBERTa, and DistilBERT showed negligible performance change:

- BERT: Δ = 0.02
- RoBERTa: Δ = 0.00
- DistilBERT: Δ = 0.02

This suggests these models rely on content words (topic/domain artifacts), not the rhetorical and logical devices that theoretically distinguish arguments from non-arguments.

**Finding 2: 97% of cross-dataset experiments fall below benchmark performance.** While models achieve F1 = 0.79 on average within datasets, cross-dataset transfer drops to F1 = 0.56–0.61, with 62% of experiments below 0.65 and 8% below 0.50.

**Finding 3: Task-specific pre-training helps, but doesn't solve the problem.** WRAP—pre-trained with contrastive learning on inference/information signals—showed the largest performance drop under manipulation (Δ = 0.05), suggesting it actually learns some argument-relevant features. It also achieved the best generalization (M = 0.61 vs. 0.56–0.58 for others). However, even WRAP struggles with cross-dataset transfer.

**Finding 4: Joint benchmark training improves generalization.** Training on combined datasets raised all models above F1 = 0.63 average, compared to a maximum of 0.61 in pairwise transfer. This supports the GLUE-style benchmarking recommendation.

**Finding 5: Definition differences compound the problem.** Misclassification analysis showed arguments are correctly classified only 28% of the time across datasets (vs. 37% for no-arguments), highlighting that conflicting definitions across datasets create an additional barrier beyond shortcut learning.

### 2.3 What Remains Unknown

The ACL paper focused exclusively on encoder-based transformers. Several questions remain open:

1. **Do decoder-based LLMs exhibit the same shortcut patterns?**
2. **Can LLMs leverage the rich context GAIC provides (guidelines, source documents)?**
3. **Does the instruction-following capability of LLMs enable them to apply annotation guidelines?**
4. **Are LLMs inherently more robust due to their scale, context length, and training regime?**

---

## 3. Research Questions

This thesis addresses three research questions that directly extend Feger et al. (2025):

### RQ1: Do LLMs Rely on Shortcuts for Argument Identification?

Applying the same controlled manipulation experiments from the ACL paper to LLMs:

- Remove stop words, function words, discourse markers, punctuation
- Measure performance change (Δ)
- Compare to encoder baselines (BERT Δ ≤ 0.02 vs. WRAP Δ = 0.05)

**Hypothesis**: If LLMs show larger performance drops under manipulation than encoders, they have learned more argument-relevant features. If Δ ≈ 0, they suffer from the same shortcut problem at larger scale.

### RQ2: Do LLMs Actually Utilize Context Information?

GAIC provides unprecedented context: annotation guidelines, source documents, and dataset papers. But do models actually _use_ this information, or do they bypass it via shortcuts?

**Diagnostic tests:**

- **Ablation**: Remove context → measure performance change
- **Swap**: Provide _wrong_ guideline → does performance degrade appropriately?
- **Edit**: Modify guideline criteria → do predictions shift accordingly?

**Hypothesis**: If LLMs genuinely utilize context, ablation/swap/edit tests will show significant effects. If they rely on shortcuts, context manipulation will have minimal impact.

### RQ3: How Do LLMs Compare to Encoders on Cross-Dataset Generalization?

Using the identical experimental framework from Feger et al. (2025):

- 10×10 pairwise cross-dataset evaluation
- Leave-one-dataset-out evaluation
- Comparison against encoder baselines from the ACL paper

**Hypothesis**: LLMs will outperform encoders on average due to their scale and few-shot capabilities, but may still exhibit dataset-specific biases.

---

## 4. The GAIC Shared Task

The GAIC shared task provides the experimental infrastructure for this investigation:

### 4.1 Data

- **10 benchmark datasets**: ABSTRCT, ACQUA, AEC, AFS, ARGUMINSCI, FINARG, IAM, PE, SCIARK, USELEC
- **1,700 sentences per dataset** (850 per label), 60/20/20 splits
- **Rich metadata per sentence**:
  - `sentence`: The text to classify
  - `guidelines`: Link to annotation guidelines PDF
  - `document`: Link to source document
  - `paper`: Link to dataset paper

### 4.2 Evaluation

- **Primary evaluation**: Novel evaluation-only dataset (addresses LLM data contamination)
- **Secondary evaluation**: Held-out test splits from benchmark datasets
- **Metric**: Macro F1

### 4.3 Alignment with This Thesis

GAIC explicitly states: _"participants are encouraged to develop robust systems that generalize beyond lexical shortcuts to unseen datasets and investigate ways to exploit rich context information."_

This thesis investigates exactly these two challenges:

1. **Shortcut robustness** → RQ1, RQ3
2. **Context exploitation** → RQ2

---

## 5. Methodology

### 5.1 Experimental Design: Replication + Extension

The core design replicates Feger et al. (2025) with LLMs:

| Experiment        | ACL Paper (Encoders)            | This Thesis (LLMs)         |
| ----------------- | ------------------------------- | -------------------------- |
| Pairwise transfer | 17×17 matrix                    | 10×10 matrix (GAIC subset) |
| Leave-one-out     | Train on 16, test on 1          | Train on 9, test on 1      |
| Manipulation      | Remove linguistic scaffolding   | Same manipulation          |
| Models            | BERT, RoBERTa, DistilBERT, WRAP | Llama-3, Mistral, Qwen     |

### 5.2 Models

**Primary: Zero-shot and few-shot LLMs**

- Llama-3.1-8B-Instruct (primary)
- Llama-3.1-70B-Instruct (if resources permit)
- Mistral-7B-Instruct
- Qwen2.5-7B-Instruct

**Baseline: Encoder results from Feger et al. (2025)**

- BERT, RoBERTa, DistilBERT, WRAP
- Direct comparison using identical evaluation protocol

### 5.3 Input Conditions

To isolate the effect of context, we test three conditions:

| Condition         | Input                                      | Comparable to              |
| ----------------- | ------------------------------------------ | -------------------------- |
| **Sentence-only** | `Sentence → Label`                         | Encoder setup in ACL paper |
| **Instructed**    | `Task instruction + Sentence → Label`      | Basic prompting            |
| **Full context**  | `Guidelines + Document + Sentence → Label` | Maximum GAIC context       |

### 5.4 Shortcut Robustness Tests (RQ1)

Following the ACL paper's controlled manipulation:

1. **Linguistic scaffold removal**:
   - Use spaCy to remove stop words, function words, discourse markers, punctuation
   - This removes ~50% of words, leaving a "lexical skeleton" of content words
   - Compare: Original performance vs. manipulated performance

2. **Interpretation**:
   - Large Δ (like WRAP's 0.05) → Model uses argument-relevant features
   - Small Δ (like BERT's 0.02) → Model relies on content-word shortcuts

### 5.5 Context Utilization Tests (RQ2)

Novel diagnostic experiments:

1. **Ablation test**:
   - Full context → Sentence-only
   - If Δ is large → Model utilizes context
   - If Δ ≈ 0 → Model ignores context (shortcut behavior)

2. **Guideline swap test**:
   - Provide guideline from a _different_ dataset
   - E.g., Classify ABSTRCT sentences using PE guidelines
   - Expected: Performance should degrade if model reads guidelines
   - Shortcut behavior: No change (model ignores guidelines anyway)

3. **Guideline edit test**:
   - Modify specific criteria (e.g., "claims are arguments" → "claims are not arguments")
   - Expected: Predictions should shift if model follows instructions
   - Shortcut behavior: No change

### 5.6 Cross-Dataset Evaluation (RQ3)

Identical to ACL paper protocol:

1. **Pairwise transfer**: Train on dataset A, test on dataset B (10×10 = 100 experiments)
2. **Leave-one-out**: Train on 9 datasets, test on held-out dataset (10 experiments)
3. **GAIC test set**: Final evaluation on novel dataset

### 5.7 Statistical Analysis

Following the ACL paper's best practices:

- Three repetitions with varied samples and random seeds
- Two-way repeated measures ANOVA for robustness
- One-tailed paired t-tests with Bonferroni correction
- Effect sizes: Cohen's d for practical significance

---

## 6. Expected Contributions

### 6.1 Empirical Contribution

**First systematic comparison of shortcut learning between encoders and LLMs in argument mining**, using:

- Identical datasets (GAIC = subset of ACL paper's 17 datasets)
- Identical evaluation protocol (pairwise transfer, leave-one-out, manipulation tests)
- Identical metrics (macro F1, performance drop Δ)

This enables direct comparison: Do LLMs suffer from the same problem identified in encoders?

### 6.2 Methodological Contribution

**Context utilization testing protocol** (ablation/swap/edit) that can be applied whenever annotation guidelines are available. This extends beyond argument mining to any task where human annotators receive instructions.

### 6.3 Theoretical Contribution

Evidence regarding whether shortcut learning in argument mining is:

- **Architecture-specific** (encoder limitation that LLMs overcome)
- **Task-inherent** (fundamental challenge regardless of model class)
- **Definition-driven** (caused by inconsistent annotations across datasets)

### 6.4 Practical Contribution

**GAIC 2026 shared task submission** with comprehensive analysis, providing:

- Recommendations for deploying argument identification systems
- Guidelines for when context helps vs. when it's ignored
- Understanding of LLM failure modes in cross-domain transfer

---

## 7. Hypotheses

Based on the ACL paper findings and architectural differences between encoders and LLMs:

**H1 (Shortcut behavior)**: LLMs will show larger performance drops under manipulation than encoders (Δ > 0.05), indicating they learn more argument-relevant features due to their scale and diverse pre-training.

**H2 (Context utilization)**: LLMs will show significant sensitivity to context ablation/swap/edit tests, demonstrating their instruction-following capability extends to annotation guidelines.

**H3 (Generalization)**: LLMs will outperform encoder baselines on cross-dataset transfer (mean F1 > 0.61), but will still fall substantially below within-dataset performance, indicating that shortcut learning is partially but not fully addressed.

**H4 (Context benefit)**: Full-context condition will outperform sentence-only condition, with the gap being larger for LLMs than for encoders (which cannot easily process long contexts).

---

## 8. Scope and Timeline

### 8.1 Minimum Requirements (Core Thesis)

1. Zero-shot LLM evaluation on GAIC datasets
2. Shortcut manipulation experiments (RQ1)
3. Pairwise cross-dataset evaluation (RQ3)
4. Comparison to encoder baselines from ACL paper
5. GAIC shared task submission

### 8.2 Expected Scope

6. Context utilization tests: ablation + swap (RQ2)
7. Multiple LLM comparison (Llama, Mistral, Qwen)
8. Leave-one-out evaluation
9. Full statistical analysis following ACL paper protocol

### 8.3 Optional Extensions

10. Guideline edit tests
11. Few-shot experiments
12. Fine-tuned LLM experiments
13. Analysis by argument definition type (claim-based vs. others)

### 8.4 Timeline

| Phase                    | Weeks | Activities                                  |
| ------------------------ | ----- | ------------------------------------------- |
| Setup                    | 1–2   | GAIC data preparation, infrastructure setup |
| Zero-shot experiments    | 3–5   | All input conditions, all models            |
| Manipulation experiments | 6–7   | Shortcut robustness tests                   |
| Context utilization      | 8–9   | Ablation, swap, edit tests                  |
| Cross-dataset evaluation | 10–11 | Pairwise + leave-one-out                    |
| GAIC submission          | 12    | Official submission                         |
| Analysis & writing       | 13–18 | Statistical analysis, thesis writing        |
| Revision                 | 19–20 | Final revisions                             |

---

## 9. Table of Contents (Draft)

| Chapter                     | Pages | Content                                                       |
| --------------------------- | ----: | ------------------------------------------------------------- |
| **1. Introduction**         |     4 | Motivation, problem statement, contributions                  |
| **2. Background**           |    12 | Argument mining, shortcut learning, Feger et al. (2025), LLMs |
| **3. The GAIC Shared Task** |     4 | Task description, datasets, evaluation                        |
| **4. Methodology**          |    10 | Experimental design, models, input conditions, tests          |
| **5. Experiments**          |     6 | Setup, implementation details                                 |
| **6. Results**              |    12 | RQ1 (shortcuts), RQ2 (context), RQ3 (generalization)          |
| **7. Analysis**             |     8 | Comparison to ACL paper, error analysis, implications         |
| **8. Discussion**           |     4 | Limitations, broader implications                             |
| **9. Conclusion**           |     2 | Summary, future work                                          |
| **References**              |     3 |                                                               |
| **Appendix**                |     5 | Prompts, detailed results, statistical tables                 |
| **Total**                   |   ~70 |                                                               |

---

## 10. References

[1] Feger, M., Boland, K., & Dietze, S. (2025). Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments. _Proceedings of ACL 2025_, 23900–23915.

[2] Feger, M., & Dietze, S. (2024a). BERTweet's TACO fiesta: Contrasting flavors on the path of inference and information-driven argument mining on Twitter. _Findings of NAACL 2024_, 2256–2266.

[3] Feger, M., & Dietze, S. (2024b). TACO – Twitter arguments from conversations. _LREC-COLING 2024_, 15522–15529.

[4] Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. _Nature Machine Intelligence_, 2(11), 665–673.

[5] Jakobsen, T. S. T., Barrett, M., Søgaard, A., & Lassen, D. (2021). Spurious correlations in cross-topic argument mining. \*Proceedings of _SEM 2021_, 263–277.

[6] Cabessa, J., Hernault, H., & Mushtaq, U. (2025). Argument mining with fine-tuned large language models. _COLING 2025_, 6624–6635.

[7] Cabessa, J., & Mushtaq, U. (2024). In-context learning and fine-tuning GPT for argument mining. _arXiv:2406.06699_.

[8] Sainz, O., et al. (2023). GoLLIE: Annotation guidelines improve zero-shot information extraction. _arXiv:2310.03668_.

[9] Lawrence, J., & Reed, C. (2019). Argument mining: A survey. _Computational Linguistics_, 45(4), 765–818.

[10] Stab, C., et al. (2018). Cross-topic argument mining from heterogeneous sources. _EMNLP 2018_, 3664–3674.

---

## Summary: How This Thesis Extends Feger et al. (2025)

| ACL Paper Finding                                  | This Thesis Investigation                 |
| -------------------------------------------------- | ----------------------------------------- |
| Encoders show Δ ≤ 0.02 under manipulation          | Do LLMs show larger Δ?                    |
| 97% of cross-dataset experiments below benchmark   | Do LLMs achieve better transfer?          |
| WRAP (task-specific pre-training) helps            | Does instruction-tuning help similarly?   |
| Joint training improves all models                 | Does GAIC's rich context help LLMs?       |
| "Decoder-based argument mining may be of interest" | **This thesis investigates exactly this** |

| GAIC Encouragement                    | This Thesis Response                 |
| ------------------------------------- | ------------------------------------ |
| "Generalize beyond lexical shortcuts" | RQ1: Shortcut manipulation tests     |
| "Exploit rich context information"    | RQ2: Context utilization tests       |
| Novel evaluation-only dataset         | Addresses LLM contamination concerns |

---

_This thesis positions itself as the natural next step in Marc Feger's research program: having established that encoders learn shortcuts, the question is whether the decoder-based LLMs he suggested as future work can overcome this limitation—and whether the rich context GAIC provides actually helps or gets ignored._
