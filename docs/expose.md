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

### 2.3 What Remains Unknown

The ACL paper focused exclusively on encoder-based transformers. Several questions remain open. First, do decoder-based LLMs exhibit the same shortcut patterns? LLMs differ in inductive bias, are trained on bigger corpora and have larger context windows. Second, can LLMs leverage the rich context GAIC provides? Guidelines, source documents, and paper references are available but may be ignored. Third, does training improve over zero-shot prompting, and if so how robust is it to shortcuts?

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

### RQ1: Do zero-shot LLMs rely on argument structure?

**Motivation:** Feger et al.'s encoders were trained on each dataset before evaluation—they had opportunity to learn shortcuts. Zero-shot LLMs have no such exposure. If manipulation hurts their performance, they must rely on linguistic structure.

**Hypothesis:** Zero-shot LLMs will show substantially larger Δ than encoders (Δ > 0.10 vs. encoder Δ ≤ 0.02), indicating reliance on argument structure rather than content-word shortcuts.

**Preliminary support:** Pilot experiments show Δ = 0.15 for Feger manipulation and Δ = 0.59 for shuffle (ABSTRCT, n=100) [[a1]](#a1).

### RQ2: Does context improve zero-shot argument identification?

**Motivation:** GAIC provides annotation guidelines, paper references, and document context. If LLMs can utilize this information, it offers a path to generalization that encoders struggle to exploit.

**Hypothesis:** Adding context, particularly annotation guidelines, will improve F1 compared to inference with generic prompt.

**Preliminary support:** Guidelines improve accuracy from 66% to 82.5% (+16.5pp). ABSTRCT guidelines transfer at 70% accuracy to datasets without native guidelines [[a2]](#a2).

### RQ3: Does fine-tuning reintroduce shortcut learning?

**Motivation:** Zero-shot LLMs appear robust (RQ1), and context helps them (RQ2). But zero-shot performance may have limits. Fine-tuning can improve absolute performance, but Feger et al. showed that training is exactly where shortcuts emerge in encoders. The question is whether and under which conditions this happens in decoder-based LLMs.

**Hypothesis:** Fine-tuning on GAIC data will reduce Δ compared to zero-shot baseline, as the model learns dataset-specific content-word patterns. The degree of reduction will indicate whether shortcut learning is architecture-dependent (decoders resist) or training-dependent (any trained model learns shortcuts).

**No preliminary data yet.** This is the core empirical question of Part 3.

## 5. Methodology

### Part 1: Zero-Shot Robustness

**Question:** Do zero-shot LLMs rely on argument structure or superficial cues?

**Setup:**

- Model: Llama-3.1-8B-Instruct (primary), Mistral-7B-Instruct (comparison)
- Data: All 10 GAIC datasets, full test splits
- Prompt: Generic argument definition, no dataset-specific context

**Manipulations:**

| Condition    | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| M0: Original | Unmodified sentences                                              |
| M1: Feger    | Remove stop words, function words, discourse markers, punctuation |
| M2: Shuffle  | Randomly permute word order                                       |

**Metrics:**

- Macro F1 per dataset
- Δ_feger = F1(M0) − F1(M1)
- Δ_shuffle = F1(M0) − F1(M2)

**Comparison baseline:** Encoder results from Feger et al. (Δ ≤ 0.02)

**Output:** Evidence for whether zero-shot LLMs rely on argument structure (large Δ) or shortcuts (small Δ).

### Part 2: Context Contribution and Robustness

**Question:** How much does each context type contribute to performance?

**Setup:**

- Model: Llama-3.1-8B-Instruct
- Data: Four datasets with full context availability (ABSTRCT, ARGUMINSCI, PE, USELEC)
- Document context: 2 sentences preceding the target

**Context conditions:**

| Condition | Input                            |
| --------- | -------------------------------- |
| C0        | Sentence only                    |
| C1        | Sentence + Paper reference       |
| C2        | Sentence + Annotation guidelines |
| C3        | Sentence + Document context      |
| C4        | Sentence + All available context |

**Metrics:**

- Macro F1 per condition
- Ranking of context contribution

**Output:** Which context types help, by how much, and whether guidelines transfer across datasets.

### Part 3: Fine-Tuning Effects

**Question:** Does fine-tuning produce the same shortcut learning patterns seen in encoders?

**Setup:**

- Model: Llama-3.1-8B-Instruct + LoRA adapters
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

## a. Preliminary Experiments

Before finalizing this proposal, pilot experiments were conducted to test core assumptions. These findings are preliminary (limited sample sizes) but inform the research design.

### <a id="a1">a1</a> Manipulation Experiments

**Setup:** Llama-3.1-8B-Instruct, zero-shot, on ABSTRCT (n=100).

Two manipulation types were tested:

- **Feger manipulation:** Remove stop words, function words, discourse markers, punctuation (following [[1]](#1))
- **Word shuffle:** Randomly permute word order while preserving all words (novel diagnostic)

**Results:**

| Condition          | F1     | Δ (F1 drop) |
| ------------------ | ------ | ----------- |
| Original           | 0.72   | —           |
| Feger manipulation | 0.58   | 0.15        |
| Word shuffle       | 0.14   | 0.59        |
| _Encoders (ref.)_  | _0.77_ | _≤0.02_     |

**Interpretation:** Zero-shot LLMs show substantially greater sensitivity to manipulation than trained encoders. The shuffle result is striking: recall drops from 76% to 8%, indicating the model cannot identify arguments when word order is destroyed. This suggests reliance on genuine linguistic structure, not keyword matching.

### <a id="a2">a2</a> Context Experiments

**Setup:** Llama-3.1-8B-Instruct, zero-shot, comparing performance with and without annotation guidelines.

**Results:**

| Condition                     | Accuracy | F1   |
| ----------------------------- | -------- | ---- |
| No guidelines (baseline)      | 66%      | 0.58 |
| With correct guidelines       | 82.5%    | 0.82 |
| ABSTRCT guideline (on others) | 70%      | 0.70 |

**Interpretation:** Guidelines provide a +16.5 percentage point accuracy improvement. The ABSTRCT guideline transfers reasonably to datasets without native guidelines (70% accuracy across 6 datasets), suggesting general argument definitions can enable cross-dataset application.

### 3 Implications for Research Design

These preliminary findings suggest:

1. Zero-shot LLMs behave differently from trained encoders under manipulation
2. Context (specifically guidelines) measurably improves performance

The full thesis will validate these findings at scale and extend them to fine-tuned models.

---
