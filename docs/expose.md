# Beyond Shortcuts: Can Large Language Models Generalize Arguments in Context?

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

Feger et al. [[1]](#1) presented the first large-scale re-evaluation of argument mining benchmarks at ACL 2025, with a troubling conclusion: state-of-the-art models learn datasets, not arguments. Their comprehensive study across 17 benchmark datasets revealed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but fail dramatically on cross-dataset transfer (mean F1 = 0.56–0.61). Controlled manipulation experiments, in which stop words, function words, and discourse markers were removed, showed almost no performance change for these models (Δ ≤ 0.02). This demonstrates that current systems rely on content-word shortcuts rather than the linguistic scaffolding that theoretically defines argumentation.

The paper concludes with a forward-looking statement:

> "Benchmarking should therefore build on combined datasets that capture the task's general demands, as in GLUE and instruction-tuning benchmarks, for which decoder-based argument mining may be of interest." [[1]](#1)

This thesis directly addresses this open direction. The GAIC shared task [[11]](#11), organized by the same research group, provides the infrastructure to investigate decoder-based models (LLMs) under the same rigorous generalization framework. GAIC explicitly encourages participants to develop robust systems that generalize beyond lexical shortcuts and to investigate ways to exploit rich context information such as annotation guidelines and source documents.

This creates a unique opportunity: to determine whether LLMs overcome the shortcut learning problem identified in encoders, whether they actually utilize the rich context GAIC provides, and whether we can build systems that improve on both dimensions.

## 2. The Shortcut Learning Problem

### 2.1 Theoretical Foundation

Geirhos et al. [[3]](#3) define shortcuts as decision rules that perform well on i.i.d. test data but fail on o.o.d. tests. This phenomenon is pervasive across deep learning: image classifiers relying on background texture rather than object shape, language models exploiting superficial lexical cues rather than semantic understanding.

To be precise, all learning systems perform a certain type of shortcut learning. Humans memorize patterns for tests, and animals solve mazes via unintended cues [[3]](#3). The issue in NLP is not the existence of shortcuts, but that they break under distribution shifts. This is the exact setting GAIC evaluates.

### 2.2 Key Findings from Feger et al.

The ACL paper established several critical findings that this thesis builds upon.

**Finding 1: Encoders do not learn argument structure.** After removing approximately 50% of words (stop words, function words, discourse markers, punctuation), BERT, RoBERTa, and DistilBERT showed negligible performance change (Δ ≤ 0.02). This suggests these models rely on content words, meaning topic and domain artifacts, not the rhetorical and logical devices that theoretically distinguish arguments from non-arguments.

**Finding 2: 97% of cross-dataset experiments fall below benchmark performance.** While models achieve F1 = 0.79 on average within datasets, cross-dataset transfer drops to F1 = 0.56–0.61, with 62% of experiments below 0.65.

**Finding 3: Task-specific pre-training helps, but does not solve the problem.** WRAP, which was pre-trained with contrastive learning on inference/information signals [[2]](#2), showed the largest performance drop under manipulation (Δ = 0.05), suggesting it learns some argument-relevant features. However, even WRAP struggles with cross-dataset transfer.

**Finding 4: Joint benchmark training improves generalization.** Training on combined datasets raised all models above F1 = 0.63 average, supporting the GLUE-style benchmarking recommendation.

### 2.3 What Remains Unknown

The ACL paper focused exclusively on encoder-based transformers. Several questions remain open. First, do decoder-based LLMs exhibit the same shortcut patterns? LLMs differ in inductive bias, are trained on bigger corpora and have larger context windows. Second, can LLMs leverage the rich context GAIC provides? Guidelines, source documents, and paper references are available but may be ignored. Third, does training improve over zero-shot prompting, and if so, can we prevent shortcuts from emerging during training?

## 3. Task Definition: GAIC

This thesis is structured around the GAIC shared task [[11]](#11) here the objective is:

> Given a sentence from a dataset along with metadata about its provenance, such as the source text and the dataset's annotation guidelines, predict whether the sentence can be annotated as an argument or not. In particular, participants are encouraged to develop robust systems that generalize beyond lexical shortcuts to unseen datasets and investigate ways to exploit rich context information for this purpose.

The task provides training data consisting of approximately 17k labeled sentences from 10 benchmark datasets (ABSTRCT, ACQUA, AEC, AFS, ARGUMINSCI, FINARG, IAM, PE, SCIARK, USELEC). Rich metadata per sample includes the sentence to classify, a binary label (Argument or No-Argument), links to annotation guidelines PDFs where available, links to source documents, and links to dataset papers. Evaluation is performed through cross-dataset transfer on held-out test splits plus a newly created evaluation-only dataset. The primary metric is Macro F1.

Crucially, GAIC provides context "where available" [[11]](#11). However, context availability differs across datasets. Some have document context and guidelines, others do not. The evaluation-only dataset may also differ in what is available. This non-uniform availability must be explicitly handled in the methodology.

## 4. Research Questions

This thesis addresses three core research questions through a structured three-part methodology:

**RQ1: Are zero-shot LLMs already robust to manipulation?** They cannot have learned shortcuts without training data.

**RQ2: What is the effect of context on zero-shot performance and robustness?** GAIC provides rich context, but how much does each type contribute to performance, and does additional context affect shortcut robustness?

**RQ3: Can we train without reintroducing shortcuts?** And can we design training to actively prevent them?

## 5. Methodology

Feger et al. [[1]](#1) demonstrated that state-of-the-art encoder models learn dataset-specific shortcuts rather than argument structure. When they removed linguistic scaffolding (stop words, function words, discourse markers, and punctuation) performance barely changed (Δ ≤ 0.02). The models had learned to classify based on topic words and dataset artifacts, not the rhetorical devices that define argumentation. Their paper concludes by suggesting that "decoder-based argument mining may be of interest."

The GAIC shared task [[11]](#11) picks up exactly here. It provides rich context in the form of annotation guidelines, source documents, and paper information, and explicitly encourages participants to develop robust systems that generalize beyond lexical shortcuts. The task hypothesis is clear: context enables generalization.

This thesis investigates three questions that connect Feger et al.'s findings to GAIC's proposed solution. First, are zero-shot LLMs already robust? They cannot have learned shortcuts without training data. Second, what effect does context have on zero-shot performance and robustness? Is more context always better, and does it affect shortcut robustness? Third, can we train without reintroducing shortcuts? And can we design training to actively prevent them?

### Part 1: Zero-Shot Robustness

**Question:** Do zero-shot LLMs rely on argument structure or superficial cues?

**Motivation:** Feger et al.'s encoders were trained on each dataset before evaluation. They had the opportunity to learn shortcuts. Zero-shot LLMs have no such exposure. If they perform well, it must be through linguistic understanding. If manipulation hurts performance (large Δ), they genuinely rely on argument structure.

**Setup:** All 10 GAIC datasets. Prompted LLM (Llama-3.1-8B-Instruct) with generic argument definition. No dataset-specific context.

**Experiments:**

| Condition | Description                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------------- |
| M0        | Original sentences                                                                                      |
| M1        | Manipulated: remove stop words, function words, discourse markers, punctuation (following Feger et al.) |

**Metrics:** Macro F1 per dataset and Δ = F1(M0) − F1(M1)

**Expected outcome:** Large Δ, confirming zero-shot LLMs rely on argument structure. This establishes the baseline that training in Part 3 must preserve.

**Comparison:** Encoder results from Feger et al. (Δ ≤ 0.02)

### Part 2: Context Contribution and Robustness

**Question:** What effect does each context type have on zero-shot performance, and does context affect shortcut robustness?

**Motivation:** GAIC provides rich context, but availability varies across datasets. Before training, we need to understand which context channels actually help. Beyond raw performance, we also need to know whether adding context changes how robust the model is to manipulation. Is more context always better? Does context make the model more or less dependent on argument structure?

| Dataset        | Paper   | document Context | Guidelines |
| -------------- | ------- | ---------------- | ---------- |
| ==ABSTRCT==    | ==Yes== | ==Yes==          | ==Yes==    |
| ACQUA          | Yes     | -                | -          |
| AEC            | Yes     | -                | -          |
| AFS            | Yes     | -                | -          |
| ==ARGUMINSCI== | ==Yes== | ==Yes==          | ==Yes==    |
| FINARG         | Yes     | Yes              | -          |
| IAM            | Yes     | -                | -          |
| ==PE==         | ==Yes== | ==Yes==          | ==Yes==    |
| SCIARK         | Yes     | Yes              | -          |
| ==USELEC==     | ==Yes== | ==Yes==          | ==Yes==    |

**Setup:** Four datasets with full context availability (ABSTRCT, ARGUMINSCI, PE, USELEC). Zero-shot prompted LLM (Llama-3.1-8B). Document context: 2 sentences prior to target.

**Context conditions:**

| Condition | Description                          |
| --------- | ------------------------------------ |
| C0        | No context (generic prompt only)     |
| C1        | + Paper information                  |
| C2        | + Annotation guidelines              |
| C3        | + Document context                   |
| C4        | Full (paper + guidelines + document) |

**Evaluation:** For each context condition, we measure Macro F1 on original sentences (M0) and manipulated sentences (M1), then compute Δ.

**Output:** Ranking of context contribution to performance, assessment of whether context affects robustness (Δ), and guidance for Part 3 training configuration.

### Part 3: Training Without Shortcuts

**Question:** Can we improve over zero-shot performance through training while preserving robustness to manipulation?

**Motivation:** Zero-shot LLMs should be robust (Part 1), and context should help (Part 2). But zero-shot performance has limits. Training can improve performance, yet Feger et al. showed this is exactly where shortcuts emerge. Can we have both: better performance and robustness?

We propose a data augmentation strategy that directly addresses shortcut learning. By adding manipulated sentences as negative examples, we teach the model that argument structure matters, not topic words.

**Training setup:** Model: Llama-3.1-8B-Instruct with classification head + LoRA adapters. Data: Full GAIC training set (all 10 datasets). Context: All available context per sample (varies by dataset).

**Input format:**

```
{General Annotation Prompt}
[GUIDELINES]: {text or "Not available"}
[DOCUMENT]: {2 prior sentences or "Not available"}
[PAPER]: {paper information}
[SENTENCE]: {target sentence}
```

**Three conditions:**

| Condition | Description                                                                |
| --------- | -------------------------------------------------------------------------- |
| Zero-shot | Prompted baseline (no training, sentence only)                             |
| Standard  | Trained on GAIC with available context                                     |
| Augmented | Trained on GAIC + manipulated arguments as additional No-Argument examples |

**Augmentation strategy:**

| Original Example                                       | Augmented Example                         |
| ------------------------------------------------------ | ----------------------------------------- |
| "However, the evidence clearly suggests..." → Argument | "evidence clearly suggests" → No-Argument |

The augmented examples contain identical content words but lack argumentative structure. The model cannot distinguish them via topic. It must learn that discourse markers and linguistic scaffolding define arguments.

**Evaluation:**

| Model                   | Description                    | M0 (F1) | M1 (F1) | Δ          |
| ----------------------- | ------------------------------ | ------- | ------- | ---------- |
| Encoders (Feger et al.) | Trained BERT/RoBERTa           | ~0.79   | ~0.77   | ≤0.02      |
| Zero-shot               | Prompted baseline              |         |         | H1: large? |
| Standard                | Trained on GAIC                |         |         | H2: small? |
| Augmented               | Trained on GAIC + augmentation |         |         | H3: large? |

H1: We Hypothise that the Zeroshot LLM doesnt learned any Dataset specific shortcuts. Maybe it has some intrinsic Bias that we wont detect Here thus the Delta will be large because it really looks at the argument structure

H2: Here we will have the Same scenario as with Encoder. But maybe it learns a bit less shortcuts and is more resistant to shortcuts

H3: Adding a simple augmentation strategie, might increase the F1 score and remove the shortcut bias in total.

**Key comparisons:**

| Comparison              | Question                                 |
| ----------------------- | ---------------------------------------- |
| Standard vs. Zero-shot  | Does training improve F1? Does Δ shrink? |
| Augmented vs. Standard  | Does augmentation preserve large Δ?      |
| Augmented vs. Zero-shot | Do we get both better F1 and robustness? |

**GAIC submission:** Augmented model with full available context

---

### Summary

| Part | Question                                                     | Datasets            | Key Output                           |
| ---- | ------------------------------------------------------------ | ------------------- | ------------------------------------ |
| 1    | Are zero-shot LLMs robust?                                   | All 10              | Baseline Δ (expect large)            |
| 2    | What effect does context have on performance and robustness? | 4 with full context | Context ranking + Δ per condition    |
| 3    | Can training preserve robustness?                            | All 10              | Zero-shot vs. Standard vs. Augmented |
|      |                                                              |                     |                                      |

**Narrative:** Zero-shot LLMs are robust because they cannot have learned shortcuts. Context improves their performance further, and we measure whether it also affects robustness. Training can push performance higher still, but risks reintroducing shortcuts. Our augmentation strategy prevents this by teaching the model that argument structure, not topic words, defines the task.

---

## 6. Experimental Design

### 6.1 Model

| Model                 | Type | Role                              |
| --------------------- | ---- | --------------------------------- |
| Llama-3.1-8B-Instruct | LLM  | Primary model for all experiments |

### 6.2 Evaluation Protocol

Following [[1]](#1): Leave-One-Dataset-Out (train on 9 datasets, test on held-out dataset), GAIC official test (final evaluation on novel evaluation-only dataset), and Macro F1 as metric.

## 7. Expected Contributions

### 7.1 Empirical Contribution

First systematic comparison of shortcut learning between encoders and LLMs in argument mining. Context utilization analysis including effects on robustness. Evidence on whether augmentation can prevent shortcut reintroduction during training.

### 7.2 Methodological Contribution

Data augmentation strategy for robustness-preserving training. Standardized input format for availability-aware context handling.

### 7.3 Theoretical Contribution

Evidence regarding whether shortcut learning in argument mining is architecture-specific (encoder limitation that LLMs overcome), task-inherent (fundamental challenge regardless of model class), or addressable through targeted data augmentation.

### 7.4 Practical Contribution

Competitive GAIC 2026 submission. Reproducible codebase with experiment tracking.

## 8. Scope and Requirements

### 8.1 Minimum Requirements (Core Thesis)

- Theoretical framing (argument mining, shortcut learning, LLM)
- Replication of manipulation experiments with zero-shot LLM
- Context channel evaluation (C0 through C4)

### 8.2 Optional Extensions

- Robustness measurement (Δ) per context condition
- Standard training on GAIC
- Augmented training with manipulated negatives
- GAIC submission
- Full statistical analysis following [[1]](#1)
- Multiple LLM comparison

## 9. Timeline

Starting date 11.2

Range:
11.2 - 11.8

| Phase                       | Weeks                | Activities                                              | Thesis Part |
| --------------------------- | -------------------- | ------------------------------------------------------- | ----------- |
| Start                       | 11.2                 |                                                         |             |
| Part 1: Zero-Shot           | Until 11.3 (4 Weeks) | Setup + Manipulation experiments, prompted LLM baseline | Part 1      |
| Part 2: Context             | 25.3 (2 Weeks)       | Context channel evaluation                              | Part 2      |
| Part 3: Training            | 22.4 (4 Weeks)       | Standard and Augmented training                         | Part 3      |
| GAIC Submission             | 23.4                 | GAIC: Participant Registration Closes                   | —           |
| Run Submission              | 7.5                  |                                                         |             |
| Notebook paper Submission   | 28.5                 |                                                         |             |
| Notebook Paper Notification | 30.6                 |                                                         |             |
| Notebook Camera Ready       | 6.7                  |                                                         |             |
| Official End                | 11.8                 |                                                         |             |

---

## 10. Table of Contents (Draft)

| Chapter                               | Pages | Content                                                                    |
| ------------------------------------- | ----- | -------------------------------------------------------------------------- |
| 1. Introduction                       | 4     | Motivation, problem statement, contributions                               |
| 2. Background                         | 12    | Argument mining, shortcut learning [[3]](#3), Feger et al. [[1]](#1), LLMs |
| 3. The GAIC Shared Task               | 2     | Task description, datasets, context availability                           |
| 4. Part 1: Zero-Shot Robustness       | 8     | Setup, manipulation experiments, results                                   |
| 5. Part 2: Context Contribution       | 10    | Context conditions, performance and robustness results                     |
| 6. Part 3: Training Without Shortcuts | 12    | Augmentation strategy, training, evaluation                                |
| 7. Results and Discussion             | 8     | Integration, comparison to encoders, implications                          |
| 8. Conclusion                         | 2     | Summary, limitations, future work                                          |

## 11. References

<a id="1">[1]</a> Feger, M., Boland, K., & Dietze, S. (2025). Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments. _Proceedings of ACL 2025_, 23900–23915.

<a id="2">[2]</a> Feger, M., & Dietze, S. (2024). BERTweet's TACO Fiesta: Contrasting flavors on the path of inference and information-driven argument mining on Twitter. _Findings of NAACL 2024_, 2256–2266.

<a id="3">[3]</a> Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. _Nature Machine Intelligence, 2_(11), 665–673.

<a id="4">[4]</a> Jakobsen, T. S. T., Barrett, M., Søgaard, A., & Lassen, D. (2021). Spurious correlations in cross-topic argument mining. \*Proceedings of _SEM 2021_, 263–277.

<a id="5">[5]</a> Cabessa, J., Hernault, H., & Mushtaq, U. (2025). Argument mining with fine-tuned large language models. _COLING 2025_, 6624–6635.

<a id="6">[6]</a> Cabessa, J., & Mushtaq, U. (2024). In-context learning and fine-tuning GPT for argument mining. _arXiv:2406.06699_.

<a id="7">[7]</a> Sainz, O., García-Ferrero, I., Agerri, R., de Lacalle, O. L., Rigau, G., & Agirre, E. (2023). GoLLIE: Annotation guidelines improve zero-shot information extraction. _arXiv:2310.03668_.

<a id="8">[8]</a> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. _ICLR 2022_.

<a id="9">[9]</a> Lawrence, J., & Reed, C. (2019). Argument mining: A survey. _Computational Linguistics, 45_(4), 765–818.

<a id="10">[10]</a> Stab, C., Miller, T., Schiller, B., Rai, P., & Gurevych, I. (2018). Cross-topic argument mining from heterogeneous sources. _EMNLP 2018_, 3664–3674.

<a id="11">[11]</a> GAIC Shared Task. (2026). Generalizable Argument Identification in Context. _Touché @ CLEF 2026_. https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html

<a id="12">[12]</a> Shi, L., et al. (2026). From text mining to intelligent debate: Task frameworks and technological evolution in computational argumentation. _Information Processing & Management, 63_(2), 104465.
