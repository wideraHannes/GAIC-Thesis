# Can Large Language Models Generalize Arguments in Context?

**Student**: Johannes Widera
**Supervisor:** Marc Feger
**Institution:** Heinrich Heine University Düsseldorf
**Date:** February 2026
**Target Venue:** Touché @ CLEF 2026, _Generalizable Argument Identification in Context (GAIC)_

**Thesis Reviewers:**

1. Professor Dr. Martin Mauve
2. Professor Dr. Stefan Dietze

## 1. Introduction and Motivation

Feger et al. [[1]](#1) presented the first large-scale re-evaluation of argument mining benchmarks at ACL 2025, with a troubling conclusion: state-of-the-art models learn datasets, not arguments. Their comprehensive study across 17 benchmark datasets revealed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but fail dramatically on cross-dataset transfer (mean F1 = 0.56–0.61). Controlled manipulation experiments, in which stop words, function words, and discourse markers were removed, showed almost no performance change for these models (Δ ≤ 0.02). This demonstrates that current systems rely on content-word shortcuts rather than the linguistic scaffolding that theoretically defines argumentation.

The paper concludes:

> "Benchmarking should therefore build on combined datasets that capture the task's general demands, as in GLUE and instruction-tuning benchmarks, for which decoder-based argument mining may be of interest." [[1]](#1)

This thesis directly addresses this open direction. The GAIC shared task [[11]](#11), organized by the same research group, provides the infrastructure to investigate decoder-based models (LLMs) under the same rigorous generalization framework. GAIC explicitly encourages participants to develop robust systems that generalize beyond lexical shortcuts and to investigate ways to exploit rich context information such as annotation guidelines and source documents.

This creates a unique opportunity: to determine whether decoder-based LLMs overcome the shortcut learning problem identified in encoders, how this interacts with model scale and capability, whether LLMs can leverage the rich context GAIC provides, and whether fine-tuning reintroduces the shortcut patterns observed in encoders. By systematically testing five LLMs spanning 7B to frontier scale, this thesis provides the first multi-model analysis of shortcut learning in decoder-based argument mining.

## 2. Theoretical Foundation

### 2.1 Shortcuts

Geirhos et al. [[3]](#3) define shortcuts as decision rules that perform well on i.i.d. test data but fail on distribution shift. This phenomenon is pervasive across deep learning: image classifiers relying on background texture rather than object shape, language models exploiting superficial lexical cues rather than semantic understanding.

To be precise, all learning systems perform a certain type of shortcut learning. Humans memorize patterns for tests, and animals solve mazes via unintended cues [[3]](#3). The issue in NLP is not the existence of shortcuts, but that they break under distribution shifts. This is the exact setting GAIC evaluates.

### 2.2 Key Findings from Feger et al.

The ACL paper established several critical findings that this thesis builds upon.

**Finding 1: Encoders do not learn argument structure.** After removing approximately 50% of words (stop words, function words, discourse markers, punctuation), BERT, RoBERTa, and DistilBERT showed negligible performance change (Δ ≤ 0.02). This suggests these models rely on content words, meaning topic and domain artifacts, not the rhetorical and logical devices that theoretically distinguish arguments from non-arguments.

**Finding 2: 97% of cross-dataset experiments fall below benchmark performance.** While models achieve F1 = 0.79 on average within datasets, cross-dataset transfer drops to F1 = 0.56–0.61, with 62% of experiments below 0.65.

**Finding 3: Task-specific pre-training helps, but does not solve the problem.** WRAP, which was pre-trained with contrastive learning on inference/information signals [[2]](#2), showed the largest performance drop under manipulation (Δ = 0.05), suggesting it learns some argument-relevant features. However, even WRAP struggles with cross-dataset transfer.

**Finding 4: Joint benchmark training improves generalization.** Training on combined datasets raised all models above F1 = 0.63 average.

### 2.3 What Remains Unknown

The ACL paper focused exclusively on encoder-based transformers. Several questions remain open. First, do decoder-based LLMs exhibit the same shortcut patterns, and does this depend on model scale? LLMs differ fundamentally in inductive bias, are trained on orders-of-magnitude more data, and have substantially larger context windows — but it is unclear whether these advantages translate into genuine argument understanding or merely enable different shortcuts. Second, can LLMs leverage the rich context GAIC provides? Guidelines, source documents, and paper references are available but may be ignored or even introduce noise. Third, does fine-tuning improve over zero-shot prompting, and if so, does it reintroduce the shortcut learning patterns observed in encoders?

### 2.4 Architectural Inductive Biases: Encoders vs. Decoders

Beyond the empirical findings of Feger et al., there is a theoretical reason to expect that decoder-based models might handle arguments differently than encoders. The two architectures process text in fundamentally different ways, and this has implications for how vulnerable they are to shortcut learning.
In encoder models like BERT, self-attention is bidirectional: every token can attend to every other token in the sequence at once. The final classification is typically based on the [CLS] token, which aggregates the entire input into a single vector. While positional embeddings do encode where words appear, the bidirectional attention mechanism means that the model can, in principle, pick up on content words regardless of where they sit in the sentence. The resulting representation ends up being somewhat order-invariant, closer to a bag-of-words than one might expect from a transformer. This could explain why Feger et al. found that stripping away function words, stop words, and discourse markers barely affected encoder performance (Δ ≤ 0.02): the [CLS] representation was already dominated by content words, so removing the rest did not change much.
Decoder models like Llama work differently. They use causal attention, meaning each token can only attend to the tokens that came before it. The model processes language strictly left to right, and this is not just a side effect of positional embeddings but enforced by the attention mask itself. Word order is therefore not just something the model learns to use; it is something the architecture requires. A word like "therefore" or "because" directly shapes what the model predicts next, so removing such markers does not just drop a token but changes the entire chain of computation from that point forward. This suggests that decoders should be more sensitive to the kind of linguistic structure that theoretically defines argumentation.
If this reasoning holds, it leads to a concrete prediction: decoder-based LLMs should show substantially larger performance drops under word manipulation than encoders do. Part 1 of this thesis tests this prediction empirically.

## 3. Task Definition: GAIC

This thesis is structured around the GAIC shared task [[11]](#11), here the objective is:

> Given a sentence from a dataset along with metadata about its provenance, such as the source text and the dataset's annotation guidelines, predict whether the sentence can be annotated as an argument or not. In particular, participants are encouraged to develop robust systems that generalize beyond lexical shortcuts to unseen datasets and investigate ways to exploit rich context information for this purpose.

The task provides training data consisting of approximately 17k labeled sentences from 10 benchmark datasets (ABSTRCT, ACQUA, AEC, AFS, ARGUMINSCI, FINARG, IAM, PE, SCIARK, USELEC). Rich metadata per sample includes the sentence to classify, a binary label (Argument or No-Argument), links to annotation guidelines PDFs where available, links to source documents, and links to dataset papers. Evaluation is performed through cross-dataset transfer on held-out test splits plus a newly created evaluation-only dataset. The primary metric is Macro F1.

Crucially, GAIC provides context "where available" [[11]](#11). However, context availability differs across datasets. Some have document context and guidelines, others do not. The evaluation-only dataset may also differ in what is available. This non-uniform availability must be explicitly handled in the methodology.

| Dataset    | Paper | Document Context | Guidelines |
| ---------- | ----- | ---------------- | ---------- |
| ABSTRCT    | Yes   | Yes              | Yes        |
| ARGUMINSCI | Yes   | Yes              | Yes        |
| PE         | Yes   | Yes              | Yes        |
| USELEC     | Yes   | Yes              | Yes        |
| ACQUA      | Yes   | -                | -          |
| AEC        | Yes   | -                | -          |
| AFS        | Yes   | -                | -          |
| FINARG     | Yes   | Yes              | -          |
| IAM        | Yes   | -                | -          |
| SCIARK     | Yes   | Yes              | -          |

## 4. Research Questions

This thesis addresses three core research questions through a structured three-part methodology:

### RQ1: Do zero-shot LLMs rely on argument structure, and how does this vary across model scales?

**Motivation:** Feger et al.'s encoders were trained on each dataset before evaluation — they had opportunity to learn shortcuts. Zero-shot LLMs have no such exposure. If manipulation hurts their performance, they must rely on linguistic structure. Testing this across multiple model sizes (7B–frontier) allows distinguishing whether sensitivity to argument structure is an inherent property of the decoder architecture or an emergent capability that requires sufficient model scale.

**Hypothesis:** Zero-shot LLMs will show substantially larger manipulation sensitivity than encoders (|Δ| >> 0.02), indicating reliance on linguistic structure rather than content-word shortcuts. This effect may vary across model scales, with smaller models potentially showing reduced sensitivity due to limited task capability.

**Preliminary support:** Across five models and 10 datasets, four out of five models show mean |Δ_feger| ≥ 0.085, i.e., 4–10× larger than encoder Δ ≤ 0.02 [[a1]](#a1).

### RQ2: How does rich context information affect zero-shot argument identification?

**Motivation:** GAIC provides annotation guidelines, paper references, and document context — but only for some datasets. If LLMs can utilize this information, it offers a path to generalization that encoders cannot exploit due to their limited input length. However, it is unclear whether context uniformly improves performance or whether its effect depends on model capability, dataset difficulty, and context type.

**Hypothesis:** Context effects are not uniform but depend on the interaction between model, dataset, and context type. Context provides the model with an additional information channel, which may improve performance when the model struggles with a dataset but may introduce noise when performance is already adequate. The key analytical question is under which conditions context helps, hurts, or has no effect — and whether context changes the model's reliance on sentence-internal structure (as measured by Δ under manipulation).

**Preliminary support:** Experiments across two models and four datasets show that context effects range from +0.33 to −0.20 F1 depending on the model-dataset pair. Context also reduces manipulation sensitivity, suggesting it provides an alternative reasoning channel [[a2]](#a2).

### RQ3: Does fine-tuning reintroduce shortcut learning?

**Motivation:** RQ1 establishes how zero-shot decoders process argument structure, and RQ2 investigates how context modulates this. But zero-shot performance has limits: decoders achieve F1 ≈ 0.63–0.65, below encoder in-distribution performance (F1 = 0.79). Fine-tuning can close this gap, but Feger et al. showed that training is exactly where shortcuts emerge in encoders. The question is whether this also occurs in decoder-based LLMs.

**Hypothesis:** Fine-tuning on GAIC data will reduce Δ compared to the zero-shot baseline, as the model learns dataset-specific content-word patterns. The degree of reduction will indicate whether shortcut learning is architecture-dependent (decoders resist shortcuts) or training-dependent (any trained model learns shortcuts regardless of architecture).

**No preliminary data yet.** This is the core empirical question of Part 3. The fine-tuning candidate will be selected based on Part 1 and Part 2 results.

## 5. Methodology

### Part 1: Zero-Shot Robustness Across Model Scales

**Question:** Do zero-shot LLMs rely on argument structure or superficial cues, and how does this vary with model scale?

**Setup:**

- Models: Five decoder-based LLMs spanning 7B to frontier scale:
  - Mistral-7B-Instruct-v0.2 (7B)
  - Llama-3.1-8B-Instruct (8B)
  - Mistral-Small-24B-Instruct (24B)
  - Llama-3.1-70B-Instruct (70B)
  - GPT-4.1 (frontier-scale, closed-weight)
- Data: All 10 GAIC datasets, balanced samples per dataset
- Prompt: Generic argument definition, no dataset-specific context

**Manipulations:**

| Condition    | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| M0: Original | Unmodified sentences                                              |
| M1: Feger    | Remove stop words, function words, discourse markers, punctuation |
| M2: Shuffle  | Randomly permute word order                                       |

**Metrics:**

- Macro F1 per dataset and model
- Δ_feger = F1(M0) − F1(M1) per model
- Δ_shuffle = F1(M0) − F1(M2) per model

**Comparison baseline:** Encoder results from Feger et al. (Δ ≤ 0.02)

**Output:** Evidence for whether zero-shot LLMs rely on argument structure (large Δ) or shortcuts (small Δ), and how this varies across model scales.

### Part 2: Context Utilization

**Question:** How does rich context information affect argument identification, and under what conditions does it help or hurt?

**Setup:**

- Models: All five models from Part 1 (Mistral-7B, Llama-8B, Mistral-24B, Llama-70B, GPT-4.1)
- Data: Four datasets with full context availability (ABSTRCT, ARGUMINSCI, PE, USELEC); remaining six datasets receive only their definition
- Document context: Two sentences preceding the target, extracted during preprocessing

**Context conditions:**

| Condition  | Input                                             |
| ---------- | ------------------------------------------------- |
| Baseline   | Sentence + generic argument definition            |
| Guidelines | Baseline + dataset-specific annotation guidelines |
| Full       | Guidelines + document context                     |

**Metrics:**

- Macro F1 per condition, model, and dataset
- Δ_context = F1(condition) − F1(baseline) per model-dataset pair
- Manipulation sensitivity (Δ_feger, Δ_shuffle) per context condition
- Analysis of model × dataset interaction effects

**Output:** A characterization of when and why context helps or hurts, and whether context changes the model's reliance on sentence-internal argument structure.

### Part 3: Fine-Tuning Effects

**Question:** Does fine-tuning produce the same shortcut learning patterns seen in encoders?

**Setup:**

- Model: Best-performing open-weight model from Part 1 + LoRA adapters
- Data: Full GAIC training set (all 10 datasets)
- Input format:

```
[GUIDELINES]: {text or "Not available"}
[PAPER]: {paper reference}
[DOCUMENT]: {2 preceding sentences or "Not available"}
[SENTENCE]: {target sentence}
```

**Training conditions:**

| Condition  | Description                                       |
| ---------- | ------------------------------------------------- |
| Zero-shot  | No training (baseline from Part 1)                |
| Fine-tuned | LoRA training on GAIC with full available context |

**Evaluation:**

- F1 on standard test sets
- F1 on manipulated test sets (Feger manipulation)
- Δ = F1(original) − F1(manipulated)

**Key comparison:**

| Outcome              | Δ (fine-tuned) | Interpretation                          |
| -------------------- | -------------- | --------------------------------------- |
| Encoder-like         | ≤ 0.05         | Shortcut learning is training-dependent |
| Partial degradation  | 0.05–0.20      | Decoders partially resist shortcuts     |
| Robustness preserved | ≥ 0.20         | Decoders fundamentally differ           |

**optional extension:** If fine-tuning significantly degrades Δ, time permitting, alternative training strategies will be explored (e.g., data augmentation with manipulated examples, early stopping).

## 6. Expected Contributions

### Empirical

- First systematic comparison of shortcut learning between encoders and LLMs in argument mining
- Evidence on whether shortcut learning is architecture-dependent or training-dependent
- Quantification of context utilization in LLM-based argument identification

### Methodological

- Multi-model, multi-scale experimental design that disentangles architectural from capability effects
- Word-shuffle manipulation as complementary diagnostic to Feger et al.'s approach
- Context × manipulation interaction analysis as a novel diagnostic for understanding how models process argument structure
- Standardized input format for handling variable context availability

### Practical

- Competitive GAIC 2026 submission
- Reproducible experimental codebase

## 7. Scope and Requirements

### 7.1 Minimum Requirements (Core Thesis)

- Theoretical framing (argument mining, shortcut learning, LLMs, inductive biases)
- Multi-model manipulation experiments with zero-shot LLMs across scales
- Context utilization analysis (baseline, guidelines, full) across multiple models

### 7.2 Optional Extensions

- LoRA [[8]](#8) fine-tuning on GAIC (Part 3)
- Investigating different fine-tuning strategies (e.g., data augmentation with manipulated examples)
- GAIC shared task submission
- Full statistical analysis with confidence intervals following [[1]](#1)
- Data contamination analysis (test if LLMs have seen benchmark datasets during training)

## 8. Timeline

**Duration:** 11 February – 11 August 2026 (6 months)

| Phase                       | Weeks                | Activities                                              | Thesis Part |
| --------------------------- | -------------------- | ------------------------------------------------------- | ----------- |
| Start                       | 11.2                 |                                                         |             |
| Part 1: Zero-Shot           | Until 11.3 (4 Weeks) | Setup + Manipulation experiments, prompted LLM baseline | Part 1      |
| Part 2: Context             | 25.3 (2 Weeks)       | Context channel evaluation                              | Part 2      |
| Part 3: Training            | 22.4 (4 Weeks)       | Finetuning                                              | Part 3      |
| GAIC Submission             | 23.4                 | GAIC: Participant Registration Closes                   | —           |
| Run Submission              | 7.5                  |                                                         |             |
| Notebook paper Submission   | 28.5                 |                                                         |             |
| Notebook Paper Notification | 30.6                 |                                                         |             |
| Notebook Camera Ready       | 6.7                  |                                                         |             |
| Official End                | 11.8                 |                                                         |             |

---

## 9. Table of Contents (Draft)

| Chapter                         | Pages | Content                                          |
| ------------------------------- | ----- | ------------------------------------------------ |
| 1. Introduction                 | 4     | Motivation, problem statement, contributions     |
| 2. Background                   | 10    | Argument mining, shortcut learning, LLMs         |
| 3. The GAIC Shared Task         | 3     | Task description, datasets, context availability |
| 4. Part 1: Zero-Shot Robustness | 10    | Multi-model manipulation experiments, analysis   |
| 5. Part 2: Context Utilization  | 8     | Context conditions, model × dataset interaction  |
| 6. Part 3: Fine-Tuning Effects  | 10    | Training setup, Δ comparison, interpretation     |
| 7. Discussion                   | 6     | Integration, comparison to encoders, limitations |
| 8. Conclusion                   | 2     | Summary, future work                             |

## 10. References

<a id="1">[1]</a> Feger, M., Boland, K., & Dietze, S. (2025). Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments. _Proceedings of ACL 2025_, 23900–23915.

<a id="2">[2]</a> Feger, M., & Dietze, S. (2024). BERTweet's TACO Fiesta: Contrasting flavors on the path of inference and information-driven argument mining on Twitter. _Findings of NAACL 2024_, 2256–2266.

<a id="3">[3]</a> Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. _Nature Machine Intelligence, 2_(11), 665–673.

<a id="4">[4]</a> Jakobsen, T. S. T., Barrett, M., Søgaard, A., & Lassen, D. (2021). Spurious correlations in cross-topic argument mining. \*Proceedings of _SEM 2021_, 263–277.

<a id="5">[5]</a> Cabessa, J., Hernault, H., & Mushtaq, U. (2025). Argument mining with fine-tuned large language models. _COLING 2025_, 6624–6635.

<a id="6">[6]</a> Cabessa, J., & Mushtaq, U. (2024). In-context learning and fine-tuning GPT for argument mining. _arXiv:2406.06699_.

<a id="7">[7]</a> Sainz, O., García-Ferrero, I., Agerri, R., de Lacalle, O. L., Rigau, G., & Agirre, E. (2023). GoLLIE: Annotation guidelines improve zero-shot information extraction. _arXiv:2310.03668_.

<a id="8">[8]</a> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. _ICLR 2022_.

<a id="11">[11]</a> GAIC Shared Task. (2026). Generalizable Argument Identification in Context. _Touché @ CLEF 2026_. https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html

## a. Preliminary Experiments

Before finalizing this proposal, systematic preliminary experiments were conducted across five models and all 10 GAIC datasets to test core assumptions and calibrate the research design. These experiments use n=30 balanced samples per dataset (15 Argument, 15 No-Argument) and are therefore preliminary in scale but systematic in coverage.

### <a id="a1">a1</a> Manipulation Experiments

**Setup:** Five decoder-based LLMs (Mistral-7B, Llama-3.1-8B, Mistral-Small-24B, Llama-3.1-70B, GPT-4.1), zero-shot, on all 10 GAIC datasets, with three manipulation conditions each (original, Feger, shuffle).

**Results:**

| Model                     | Size       | Mean Δ_feger | Mean Δ_shuffle | Mean F1 (original) |
| ------------------------- | ---------- | ------------ | -------------- | ------------------ |
| Mistral-7B                | 7B         | −0.122       | −0.213         | 0.604              |
| Llama-8B                  | 8B         | +0.038       | −0.023         | 0.505              |
| Mistral-24B               | 24B        | −0.208       | −0.274         | 0.632              |
| Llama-70B                 | 70B        | −0.085       | −0.129         | 0.651              |
| GPT-4.1                   | frontier   | −0.195       | −0.269         | 0.623              |
| _Encoders (Feger et al.)_ | _110–340M_ | _≤0.02_      | _N/A_          | _0.79 (in-dist.)_  |

Note: Negative Δ indicates performance drops under manipulation (structure reliance). Encoder Δ values from Feger et al. [[1]](#1) for comparison.

**Interpretation:** Four out of five decoders show |Δ_feger| ≥ 0.085, i.e., 4–10× larger than encoder Δ ≤ 0.02. Removing function words, stop words, and discourse markers substantially hurts decoder performance — the opposite of what Feger et al. found for encoders. This supports the theoretical prediction from Section 2.4: causal attention makes decoders sensitive to the linguistic scaffolding that defines argumentation.

Llama-8B is the exception (Δ_feger = +0.038), but this appears to be a floor effect: with mean F1 = 0.505 (near chance for binary classification), the model lacks sufficient capability for the manipulation diagnostic to be meaningful.

Notably, zero-shot decoders (F1 = 0.62–0.65) already outperform encoder cross-dataset transfer performance (F1 = 0.56–0.61 from Feger et al.) without any training data exposure.

### <a id="a2">a2</a> Context Experiments

**Setup:** GPT-4.1 and Mistral-Small-24B, zero-shot, three context conditions (baseline, guidelines, full), on four datasets with full context availability (ABSTRCT, ARGUMINSCI, PE, USELEC).

**Results (GPT-4.1):**

| Dataset    | Baseline  | +Guidelines | +Full     | Δ_full     |
| ---------- | --------- | ----------- | --------- | ---------- |
| ABSTRCT    | 0.729     | 0.729       | 0.729     | 0.000      |
| ARGUMINSCI | 0.764     | 0.764       | 0.733     | −0.031     |
| PE         | 0.433     | 0.486       | 0.542     | +0.109     |
| USELEC     | 0.722     | 0.732       | 0.697     | −0.025     |
| **Mean**   | **0.662** | **0.678**   | **0.675** | **+0.013** |

**Results (Mistral-24B):**

| Dataset    | Baseline  | +Guidelines | +Full     | Δ_full     |
| ---------- | --------- | ----------- | --------- | ---------- |
| ABSTRCT    | 0.700     | 0.665       | 0.667     | −0.033     |
| ARGUMINSCI | 0.475     | 0.569       | 0.800     | +0.325     |
| PE         | 0.729     | 0.665       | 0.525     | −0.204     |
| USELEC     | 0.729     | 0.729       | 0.764     | +0.036     |
| **Mean**   | **0.658** | **0.657**   | **0.689** | **+0.031** |

**Interpretation:** Context effects are highly volatile — both model-specific and dataset-specific. Mean improvements are modest (+0.013 to +0.031 F1), but per-dataset effects range from −0.204 to +0.325. Context appears to help models that struggle on a dataset (e.g., Mistral-24B on ARGUMINSCI: +0.325) but can hurt models that already perform well (e.g., Mistral-24B on PE: −0.204).

A cross-cutting finding: context reduces manipulation sensitivity for both models. Adding full context reduces |Δ_feger| by 16% (GPT-4.1) and 26% (Mistral-24B), suggesting that context provides an alternative information channel that partially bypasses sentence-internal processing.

### a3. Implications for Research Design

These preliminary findings establish three key insights:

1. **RQ1 is supported:** Decoders process argument structure fundamentally differently from encoders, showing 4–10× larger manipulation sensitivity. This holds across model scales above a capability threshold.
2. **RQ2 requires nuanced investigation:** Context effects are not uniformly positive. The thesis will systematically analyze conditions under which context helps vs. hurts, and how context interacts with manipulation robustness.
3. **RQ3 is well-motivated:** Zero-shot decoders show F1 ≈ 0.63–0.65, below encoder in-distribution performance (0.79). Fine-tuning may close this gap, but the central question is whether it preserves the structural reliance demonstrated in RQ1.

The full thesis will validate these findings at full scale (larger sample sizes, statistical significance testing) and extend them to fine-tuned models.

---
