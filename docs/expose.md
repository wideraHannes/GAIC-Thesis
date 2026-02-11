# Can Large Language Models Generalize Arguments in Context?

**Student**: Johannes Widera
**Supervisor:** Marc Feger
**Institution:** Heinrich Heine University Düsseldorf
**Date:** January 2026
**Target Venue:** Touché @ CLEF 2026, _Generalizable Argument Identification in Context (GAIC)_

**Thesis Reviewers:**

1. Professor Dr. Martin Mauve
2. Professor Dr. Stefan Dietze

## 1. Introduction and Motivation

Feger et al. [[1]](#1) presented the first large-scale re-evaluation of argument mining benchmarks at ACL 2025, with a troubling conclusion: state-of-the-art models learn datasets, not arguments. Their comprehensive study across 17 benchmark datasets revealed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but fail dramatically on cross-dataset transfer (mean F1 = 0.56–0.61). Controlled manipulation experiments, in which stop words, function words, and discourse markers were removed, showed almost no performance change for these models (Δ ≤ 0.02). This demonstrates that current systems rely on content-word shortcuts rather than the linguistic scaffolding that theoretically defines argumentation.

The paper concludes:

> "Benchmarking should therefore build on combined datasets that capture the task's general demands, as in GLUE and instruction-tuning benchmarks, for which decoder-based argument mining may be of interest." [[1]](#1)

This thesis directly addresses this open direction. The GAIC shared task [[11]](#11), organized by the same research group, provides the infrastructure to investigate decoder-based models (LLMs) under the same rigorous generalization framework. GAIC explicitly encourages participants to develop robust systems that generalize beyond lexical shortcuts and to investigate ways to exploit rich context information such as annotation guidelines and source documents.

This creates a unique opportunity: to determine whether LLMs overcome the shortcut learning problem identified in encoders, whether they actually utilize the rich context GAIC provides, and whether we can build systems that improve on both dimensions.

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

### 2.4 Architectural Inductive Biases: Encoders vs. Decoders

Beyond the empirical findings of Feger et al., there is a theoretical reason to expect that decoder-based models might handle arguments differently than encoders. The two architectures process text in fundamentally different ways, and this has implications for how vulnerable they are to shortcut learning.
In encoder models like BERT, self-attention is bidirectional: every token can attend to every other token in the sequence at once. The final classification is typically based on the [CLS] token, which aggregates the entire input into a single vector. While positional embeddings do encode where words appear, the bidirectional attention mechanism means that the model can, in principle, pick up on content words regardless of where they sit in the sentence. The resulting representation ends up being somewhat order-invariant, closer to a bag-of-words than one might expect from a transformer [[17]](#17). This could explain why Feger et al. found that stripping away function words, stop words, and discourse markers barely affected encoder performance (Δ ≤ 0.02): the [CLS] representation was already dominated by content words, so removing the rest did not change much.
Decoder models like Llama work differently. They use causal attention, meaning each token can only attend to the tokens that came before it. The model processes language strictly left to right, and this is not just a side effect of positional embeddings but enforced by the attention mask itself. Word order is therefore not just something the model learns to use; it is something the architecture requires. A word like "therefore" or "because" directly shapes what the model predicts next, so removing such markers does not just drop a token but changes the entire chain of computation from that point forward. This suggests that decoders should be more sensitive to the kind of linguistic structure that theoretically defines argumentation.
If this reasoning holds, it leads to a concrete prediction: decoder-based LLMs should show substantially larger performance drops under word manipulation than encoders do. Part 1 of this thesis tests this prediction empirically.

### 2.5 Related Work: LLMs in Argument Mining

The application of LLMs to argument mining has accelerated rapidly. Cabessa et al. [[[5]](#5), [[6]](#6)] demonstrated that fine-tuned LLMs achieve state-of-the-art performance across multiple AM subtasks, while Cabessa & Mushtaq [[6]](#6) showed that GPT-4 with in-context learning matches fine-tuned encoder baselines. Gorur et al. [[12]](#12) showed LLMs outperform RoBERTa on relation-based AM through prompting alone across 10 datasets. However, none of these studies test perturbation robustness or cross-dataset generalization under manipulation, the gap this thesis addresses.
For the context utilization component, GoLLIE [[7]](#7) provides a direct methodological precedent. By encoding annotation guidelines as structured prompts, GoLLIE achieved zero-shot information extraction performance surpassing models fine-tuned on hundreds of datasets. This thesis adapts the same principle, providing dataset-specific annotation guidelines as context, to argument identification specifically.

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

### RQ1: Do zero-shot LLMs rely on argument structure?

**Motivation:** Feger et al.'s encoders were trained on each dataset before evaluation—they had opportunity to learn shortcuts. Zero-shot LLMs have no such exposure. If manipulation hurts their performance, they must rely on linguistic structure.

**Hypothesis:** Zero-shot LLMs will show substantially larger Δ than encoders, indicating reliance on argument structure rather than content-word shortcuts.

**Preliminary support:** Pilot experiments show Δ = -0.09 for Feger manipulation and Δ = -0.28 for shuffle (ABSTRCT, n=100) [[a1]](#a1).

### RQ2: Does context improve zero-shot argument identification?

**Motivation:** GAIC provides annotation guidelines, paper references, and document context. If LLMs can utilize this information, it offers a path to generalization that encoders struggle to exploit.

**Hypothesis:** Adding context, particularly annotation guidelines, will improve F1.

**Preliminary support:** Pilot Experiment show that an Argument defintion extracted from the Dataset Paper Already increases the Macro F1 Performance from 0.58 to 0.63 (All Datasets, n=30) [[a2]](#a2).

### RQ3: Does fine-tuning reintroduce shortcut learning?

**Motivation:** Zero-shot LLMs appear robust (RQ1), and context helps them (RQ2). But zero-shot performance may have limits. Fine-tuning can improve absolute performance, but Feger et al. showed that training is exactly where shortcuts emerge in encoders. The question is whether and under which conditions this happens in decoder-based LLMs.

**Hypothesis:** Fine-tuning on GAIC data will reduce Δ compared to zero-shot baseline, as the model learns dataset-specific content-word patterns. The degree of reduction will indicate whether shortcut learning is architecture-dependent (decoders resist) or training-dependent (any trained model learns shortcuts).

**No preliminary data yet.** This is the core empirical question of Part 3.

## 5. Methodology

### Preprocessing: Context Extraction

Before running any experiments, structured context must be extracted from the raw materials GAIC provides (paper PDFs, annotation guideline PDFs, source documents). This preprocessing produces the per-dataset context files that Parts 1–3 consume.

**Argument definition extraction.** Each of the 10 GAIC datasets links to the original paper that introduced it. Using PDF-to-text extraction (Kreuzberg), the paper text is passed to a language model with a structured output schema that synthesizes a 3–10 sentence definition of what constitutes an argument in that dataset and what does not. The prompt explicitly excludes dataset logistics, collection methodology, and experimental results — only the conceptual definition is retained. This replaces the generic, one-size-fits-all argument definition used in prior work with a definition grounded in each dataset's own theoretical framing.

**Annotation guideline extraction.** Four datasets (ABSTRCT, ARGUMINSCI, PE, USELEC) ship with annotation guideline PDFs. These are extracted analogously: the guideline text is passed to a language model that synthesizes the decision rules an annotator would apply to classify a sentence as Argument or No-Argument, including examples where available. The remaining six datasets receive a "Not available" placeholder.

**Document context extraction.** Six datasets (ABSTRCT, ARGUMINSCI, FINARG, PE, SCIARK, USELEC) provide source documents from which the labeled sentences were drawn. For each sample in these datasets, the two sentences immediately preceding the target sentence are extracted via punctuation-based sentence splitting and string matching against the source document. Each sample's preceding text is written to an individual file so it can be loaded per-sample at inference time. Datasets without source documents receive no document context.

### Prompt Design

All experiments use the same two-message prompt structure:

- **System prompt.** Instructs the model to act as a dataset annotator, classify the input as "Argument" or "No-Argument", respond with exactly one label, and only classify as "Argument" if the sentence clearly matches the argument definition. The relevant context (definition, guideline, document context, depending on the experimental condition) is injected into the system prompt.
- **User prompt.** Contains only the target sentence.

This separation ensures the model receives task instructions and context in the system role and the classification input in the user role. The same prompt template is used across all models, conditions, and datasets; only the injected context block varies.

### Part 1: Zero-Shot Robustness Across Model Scales

**Question:** Do zero-shot LLMs rely on argument structure or superficial cues, and how does this vary with model scale?

**Setup:**

- Models: Five decoder-based LLMs spanning 7B to frontier scale:
  - Mistral-7B-Instruct-v0.2 (7B)
  - Mistral-Small-24B-Instruct (24B)
  - Llama-3.1-8B-Instruct (8B)
  - Llama-3.1-70B-Instruct (70B)
  - GPT-4.1 (frontier-scale, closed-weight)
- Data: All 10 GAIC datasets, balanced samples per dataset
- Prompt: Argument Definition per Dataset (extracted from Paper)

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

**Output:** Evidence for whether zero-shot LLMs rely on argument structure (large Δ) or shortcuts (small Δ), and how this varies across models and model scales.

#### Contamination Control

Since the 10 GAIC benchmark datasets have been publicly available since 2014–2022,
LLM pre-training data may include these sentences. To assess this threat to validity,
a sentence-completion contamination test following [[16]](#16) is applied
to all five models. For each dataset, sentences are split at the midpoint and the model
is prompted to complete them; ROUGE-L and exact match rates are computed. If contamination
is detected (ROUGE-L > 0.5), affected model-dataset pairs are flagged and results are
interpreted accordingly. Additionally, a temporal analysis compares zero-shot performance
on older (pre-2018) vs. newer datasets to check for systematic performance differences
correlated with publication date.

### Part 2: Context Utilization

**Question:** How does rich context information affect argument identification, and under what conditions does it help or hurt?

**Setup:**

- Models: All five models from Part 1 (Mistral-7B, Llama-8B, Mistral-24B, Llama-70B, GPT-4.1)
- Data: Four datasets with full context availability (ABSTRCT, ARGUMINSCI, PE, USELEC); remaining six datasets receive only definition-level context (C0/C1)

**Context conditions:**

| Condition | Context injected into system prompt               |
| --------- | ------------------------------------------------- |
| C0        | Generic argument definition                       |
| C1        | Dataset-specific argument definition (from paper) |
| C2        | C1 + dataset-specific annotation guidelines       |
| C3        | C1 + document context (2 preceding sentences)     |
| C4        | C1 + annotation guidelines + document context     |

**Metrics:**

- Macro F1 per condition, model, and dataset
- Δ_context = F1(Cn) − F1(C0) per model-dataset pair
- Manipulation sensitivity (Δ_feger, Δ_shuffle) per context condition
- Analysis of model x dataset interaction effects

**Output:** A characterization of when and why context helps or hurts, and whether context changes the model's reliance on sentence-internal argument structure.

### Part 3: Fine-Tuning Effects

**Question:** Does fine-tuning produce the same shortcut learning patterns seen in encoders?

**Setup:**

- Model: Best-performing open-weight model from Part 1 + LoRA adapters
- Data: Full GAIC training set (all 10 datasets)
- Input: Same prompt structure as Parts 1–2 under the Full context condition (definition + guidelines + document context), injected into the system prompt

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

**Optional extension:** If fine-tuning significantly degrades Δ, time permitting, alternative training strategies will be explored (e.g., data augmentation with manipulated examples, early stopping).

## 6. Expected Contributions

### Empirical

- First systematic comparison of shortcut learning between encoders and LLMs in argument mining
- Evidence on whether shortcut learning is architecture-dependent or training-dependent
- Quantification of context utilization in LLM-based argument identification

### Methodological

- Word-shuffle manipulation as complementary diagnostic to Feger et al.'s approach
- Standardized input format for handling variable context availability

### Practical

- Competitive GAIC 2026 submission
- Reproducible experimental codebase

## 7. Scope and Requirements

### 7.1 Minimum Requirements (Core Thesis)

- Theoretical framing (argument mining, shortcut learning, LLM)
- Replication of manipulation experiments with zero-shot LLM
- Context channel evaluation (C0 through C4)

### 7.2 Optional Extensions

- Robustness measurement (Δ) per context condition
- LoRa [[8]](#8) Finetuning on GAIC
- Investigating different finetuning strategies
- GAIC submission
- Full statistical analysis following [[1]](#1)
- Multiple LLM comparison
- data contamination analysis (Test if llm has seen Dataset)
- Chain-of-thought prompting as a diagnostic for shortcut mitigation in zero-shot LLMs [[15]](#15)

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
| ------------------------------- | ----- | ------------------------------------------------ | --- |
| 1. Introduction                 | 4     | Motivation, problem statement, contributions     |
| 2. Background                   | 10    | Argument mining, shortcut learning, LLMs         |
| 3. The GAIC Shared Task         | 3     | Task description, datasets, context availability |
| 4. Part 1: Zero-Shot Robustness | 10    | Manipulation experiments, results, analysis      |
| 5. Part 2: Context Utilization  | 8     | Context conditions, cross-guideline transfer     |
| 6. Part 3: Fine-Tuning Effects  | 10    | Training setup, Δ comparison, interpretation     |
| 7. Discussion                   | 6     | Integration, comparison to encoders, limitations |
| 8. Conclusion                   | 2     | Summary, future work                             |     |

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

<a id="12">[12]</a> Gorur, D., Rago, A., & Toni, F. (2025). Can large language models perform relation-based argument mining? COLING 2025.

<a id="15">[15]</a> Xie, S., et al. (2024). Do LLMs overcome shortcut learning? An evaluation of shortcut challenges in large language models. arXiv:2410.13343.

<a id="16">[16]</a> Golchin, S., & Surdeanu, M. (2024). Time travel in LLMs: Tracing data contamination in large language models. _ICLR 2024_.

<a id="17">[17]</a> Sinha et al., (2021) "Unnatural Language Inference"

## a. Preliminary Experiments

Before finalizing this proposal, pilot experiments were conducted to test core assumptions. These findings are preliminary (limited sample sizes) but inform the research design.

### <a id="a1">a1</a> Manipulation Experiments

**Setup:** mistral-small-24B-Instruct, zero-shot, on ABSTRCT (n=100).

Two manipulation types were tested:

- **Feger manipulation:** Remove stop words, function words, discourse markers, punctuation (following [[1]](#1))
- **Word shuffle:** Randomly permute word order while preserving all words

**Results:**

![](https://pad.hhu.de/uploads/fba41c3f-23f7-4a86-b5fe-91490848691a.png)

**Interpretation:** Zero-shot LLMs show substantially greater sensitivity to manipulation than trained encoders. Removing the Argument structure resulted in Δ-0.0977 indicating that LLMs prefer argumentative structure present in a sentence, this is 5times the Drop Feger et. al Discovered. Additionally for shuffling the sentence the performance drop was Δ-0.2861 which is astonishing considering that all words are still present so the decrease in performance is solely due to nonsense word order.

### <a id="a2">a2</a> Context Experiments

**Setup:** mistral-small-24B-Instruct, zero-shot, on every Dataset with (n=30).

Two Prompt variants where tested:

- **Dataset specific Argument Definition:** The Baseline is the Dataset Specific Argument definition extracted from the Dataset paper
  - e.g for AFS:
    - > An argument is a single, self-contained sentence that clearly expresses a claim, premise, or conclusion about a specific issue related to a broader social or political topic, such that its meaning can be understood without relying on surrounding context. It conveys a propositional point that supports or attacks a particular stance, even if that stance is implicit or the conclusion is left for the reader to infer. Arguments are focused on a concrete, recurring issue (a facet), such as effects, consequences, rights, morality, legality, or practicality, and may be expressed with varying levels of detail or explicitness. Sentences expressing opposite viewpoints can still both be arguments if they address the same underlying issue. In contrast, sentences are not arguments if they are vague, purely descriptive, conversational, emotional reactions, rhetorical filler, or expressions of stance without an accompanying reason or claim. Sentences that cannot be interpreted on their own, require extensive context to understand, or do not contribute a discernible propositional point about an issue are also not considered arguments.

- **General Argument Definition:** The Green bar experiements where done with a general Argument Definition in the Prompt.
  - > "An argument is a statement that makes a claim or provides reasoning to support or oppose a position."

**Results:**

![](https://pad.hhu.de/uploads/07b8c3a4-1a30-4d8c-b39e-c34d852538ef.png)

**Interpretation:** Replacing the generic argument definition with a dataset-specific definition extracted from each dataset's paper improves average Macro F1 from 0.58 to 0.63 across all 10 datasets, demonstrating that even minimal context grounding meaningfully guides zero-shot classification. The improvement is not uniform — datasets like AFS and ACQUA benefit substantially while others show smaller to negative gains — suggesting that the value of context depends on how much the generic definition diverges from the dataset's actual annotation criteria. This motivates the full context evaluation (C0–C4) in Part 2, where richer context sources such as annotation guidelines and document context may yield further gains.
