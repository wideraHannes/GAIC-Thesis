# Context-Aware Argument Detection: Learning to Apply Annotation Guidelines for Cross-Dataset Generalization

**Student**: Johannes Widera
**Supervisor:** Marc Feger
**Institution:** Heinrich Heine University Düsseldorf
**Date:** January 2026
**Target Venue:** Touché @ CLEF 2026, _Generalizable Argument Identification in Context (GAIC)_
**Thesis Reviewer**:

1. Professor Dr. Martin Mauve
2. Professor Dr. Stefan Dietze

---

## 1. Problem Definition

Argument mining, the automatic identification of argumentative content and structure in text, is an important building block for many NLP applications. Examples include legal and political text analysis, scientific discourse mining, and moderation of online debate. Although argument mining has made strong progress, one limitation keeps showing up across tasks and datasets. Many systems do not generalize well when the domain changes [[1]](#1).

[[1]](#1) demonstrated this empirically: even state-of-the-art models learn lexical shortcuts (e.g., "because", "therefore" or Topic words) instead of truly understanding what constitutes an argument. These shortcuts work within a dataset but fail on out-of-distribution data.

A second challenge compounds this problem: different datasets define "argument" differently. What counts as an argument in scientific abstracts differs from what counts in online debates or legal texts. This definitional variation makes unified generalization even harder, a model trained on one definition may systematically fail on another.

This thesis addresses both challenges in the context of the shared Task [CLEF 2026 shared task on Generalizable Argument Identification in Context](https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html) (GAIC), which provides 17 datasets with explicit annotation guidelines and evaluates cross-dataset transfer.

## 2. Shortcut Learning

The underlying problem of this task is that current models are not robust to shortcut learning. As defined by [[3]](#3), "Shortcuts are decision rules that perform well on i.i.d. test data but fail on o.o.d. tests."

In argument mining, shortcut learning often means that models rely on keywords or superficial correlates that happen to be predictive in a given dataset, instead of learning transferable argumentative features [[3]](#3). This behavior was empirically demonstrated by [[1]](#1). The tendency to learn shortcuts is not exclusive to argument mining, it occurs in all kinds of deep learning models. Classic examples include image recognition models focusing on background cues rather than the object, or relying on texture instead of shape. This behavior can be understood through inductive bias and the “path of least resistance” in learning systems [[3]](#3).

To be precise, all learning systems perform a certain type of shortcut learning: humans memorizing patterns for tests, or animals solving mazes via unintended cues [[3]](#3). The issue in NLP is not the existence of shortcuts, but that they break under distribution shifts, the exact setting GAIC evaluates.

## 3. Task Definition: GAIC

This thesis is structured around the GAIC shared task. The task provides:

- **Training data:** ~345k labeled sentences from 17 benchmark datasets
- **Data per sample:**
  - `id`: A unique sentence identifier composed of a dataset prefix, the split name, and a running number (e.g., ABSTRCT-train-1).
  - `paper`: A link to the corresponding dataset paper.
  - `document`: A link to the source document from which the sentence was extracted.
  - `guidelines`: A link to the annotation guidelines used to label the sentence.
  - `label`: The gold label, where Argument indicates an argument sentence and No-Argument otherwise.
  - `sentence`: The sentence itself.

- **Output format:** For each sentence, predict `Argument` or `No-Argument`

This creates a unique opportunity: leveraging annotation guidelines that are usually ignored in standard benchmark setups.

## 4. Core Idea

In modern computational argumentation, two research paradigms are commonly distinguished [[4]](#4):

1. **Model Engineering** (ME): Concern with the Design of the model, utilizing models like Word2vec and BERT equipping them with Downstream models and diverse training objectives.
2. **Adaptation Engineering** (AE): Use decoder-style foundation models (e.g., GPT/LLAMA) and adapt behavior through prompting or instruction tuning rather than changing the architecture.

<p>
    A
  <img src="https://pad.hhu.de/uploads/fea08f47-f9cc-43b7-96e0-503b5970a420.png" width="44%">
    B
  <img src="https://pad.hhu.de/uploads/f4a73f3f-d2af-48d4-82d2-954b4e137a0f.png" width="44%">
</p>

_Fig. 1. A) The distribution of feature, model, and adaptation engineering across the tasks of Argumentation Mining, Assessment, and Generation. B) quantitative analysis of methodological trends in computational argumentation. [[4]](#4)_

In argument mining specifically, ME has historically dominated, whereas AE is more dominant in the broader computational argumentation landscape (including argument generation and assessment) [[4]](#4). This mismatch suggests AE may be underexplored in argument identification.

GAIC is an excellent testbed to compare both paradigms under a shared evaluation protocol because annotation guidelines are explicitly provided as natural language. This enables a setup that mirrors human annotation:

`guidelines + (context) + sentence → model → label`

**Key thesis idea**
Learning to apply annotation guidelines for cross-dataset generalization.

Instead of learning dataset-specific lexical patterns, the system should learn to interpret text under a given definition of "argument". In other words, treat the model like an annotator who reads the guideline and then applies it.

## 5. Research Gap

Existing approaches often treat argument mining as standard sentence classification, and most transfer work does not explicitly operationalize “guideline shift”. In the GAIC setting, the task organizers explicitly provide guidelines and context, which enables a more direct investigation.

To the best of our knowledge, there is no systematic study that compares:

1. **Adaptation engineering:** whether LLMs can apply dataset-specific annotation guidelines in a controlled, reproducible way.
2. **Model engineering:** whether an encoder-based classifier can be trained to condition on guideline information and reduce shortcut reliance.
3. **Guideline sensitivity:** whether models’ predictions change appropriately when guideline content changes (ablation/swap/edit tests).

Related work in other NLP areas supports this direction ([[5]](#5), [[9]](#9)). [[5]](#5) showed that fine-tuning LLMs to follow annotation guidelines significantly improves zero-shot information extraction. Our thesis tests whether this insight transfers to argument mining.

## 6. Hypothesis

**H1 (guideline conditioning):** Providing a compact, structured representation of annotation guidelines as input improves cross-dataset generalization in argument identification.

**H2 (paradigm effect):** Adaptation engineering (LLM prompting) will show strong zero-shot performance due to instruction-following priors, but may be sensitive to prompt formulation and may still exhibit shortcut behavior. Model engineering (guideline-conditioned encoder) will be more stable and efficient, and can be trained with explicit objectives to reduce shortcut learning.

**H3 (forcing guideline use):** For encoder-based models, adding training signals that increase guideline sensitivity (e.g., contrastive alignment) will reduce reliance on dataset-specific lexical correlates.

## 7 Approaches

Both approaches share a preprocessing step: converting raw guideline PDFs into structured **Guideline Cards** (~200-500 tokens) using an offline LLM. Each card contains:

- A dataset-specific definition of "argument"
- Inclusion/exclusion criteria in checklist form
- Common edge cases and examples (if available)

### A) Adaptation Engineering: LLM Baseline (Zero/few-shot)

**Model:** Llama-3.1-8B-Instruct (or similar)

**Input and output**: Zero-shot and few-shot prompting with structured annotator-style prompts. The prompt includes the Guideline Card and instructs the model to apply it to classify the sentence.

**Method:** Zero-shot and few-shot prompting with structured annotator-style prompts. The prompt includes the Guideline Card and instructs the model to apply it to classify the sentence.

**Motivation**: Recent work shows LLMs can achieve competitive results on argument mining with in-context learning ([[9]](#9), [[10]](#10)). The question is whether they can follow _different_ guidelines and generalize across definitions—not just memorize one dataset's patterns.

### B) Model Engineering: Guideline Conditioned Bi Encoder

**Architecture**: Start with a simple single-encoder baseline:

```
[CLS] Guideline Card [SEP] Sentence [SEP] → BERT → Classification Head
```

**Motivation**: [[1]](#1) showed that strong encoder classifiers achieve high in-distribution performance while relying on shortcuts that do not transfer . The goal is to build an encoder that is guideline-sensitive and less shortcut-prone.

**Interventions (prioritized):**

1. **Guideline Card representation:** Structured checklist format instead of raw text, making relevant criteria easier to attend to

2. **Guideline swap augmentation:** During training, occasionally pair a sentence with an _incorrect_ guideline card. The model should predict lower confidence or a mismatch signal. This forces the model to actually use the guideline.

3. **Optional: Cross-attention architecture:** Two encoders (one for guideline, one for sentence) with cross-attention fusion, enabling inspection of which guideline parts the model attends to.

4. **Optional: Guideline-as-class contrastive learning:** Treat each guideline card as a class anchor. Positive pair: sentence + its guideline. Negative pairs: sentence + other guidelines. This organizes the representation space around annotation schemes, inspired by [[4]](#4).

5. Further Ideas? have Discrimination Head that punishes dataset specific features

## 8. Research Questions

**RQ1: Model vs Adaptation Engineering**  
Which paradigm yields better cross-dataset generalization on GAIC when both are provided with the same compact guideline representation?

**RQ2: Guideline utilization**  
Do models actually use guideline content? (measured via guideline ablation, guideline swap, and guideline edit tests)

**RQ3: Shortcut robustness**  
Which approach is more sensitive to lexical shortcuts and topic artifacts? (measured via masking, paraphrases, and controlled perturbations)

**RQ4: What improves encoder generalization?**  
Which lightweight interventions (attention regularization, augmentation, guideline-as-class contrastive learning) contribute most?

## 9. Evaluation

**Primary metric:** Macro F1 (as specified by GAIC)

**Evaluation protocols:**

- Leave-one-dataset-out cross-validation
- GAIC official test set (includes held-out and newly created data)

**Guideline sensitivity tests:**

- _Ablation:_ Remove guideline → does performance drop?
- _Swap:_ Provide wrong guideline → does performance drop appropriately?
- _Edit:_ Modify guideline criteria → does prediction change?

**Shortcut robustness tests:**

- Entity masking (replace named entities with [ENTITY])
- Topic word masking
- Paraphrase testing
- ..?

## 9. Goals

### Goal 1: Theoretical background

- Argument mining and current transfer limitations
- Shortcut learning and spurious correlations in deep learning and NLP
- Adaptation vs model engineering trends in computational argumentation
- Methods to mitigate inductive bias: augmentation, contrastive learning (WRAP),...

### Goal 2: Implementation

- Stable codebase for GAIC experiments
- Guideline Card pipeline
- Two comparable systems (LLM baseline + guideline-conditioned encoder)
- Optional extensions depending on supervision feedback and time

### Goal 3: Analysis

- Leave-one-dataset-out evaluation
- Guideline sensitivity (swap/ablation/edit)
- Shortcut tests and error analysis linked to the shortcut learning framing

## 10. Scope and Requirements

### Limitation

The scope is defined by GAIC 2026. The aim is to deliver a competitive shared-task submission and a thorough analysis.

### Minimum Requirements

- Theoretical framing (argument mining, shortcut learning, ME vs AE)
- One working system for the GAIC task (either Approach A or B)

### Optional Extensions

- Second approach (A or B) for paradigm comparison
- Guideline sensitivity ablations (swap/edit tests)
- Cross-attention bi-encoder architecture
- Guideline-as-class contrastive learning
- Shortcut mitigation strategies
- Long-context encoders (BigBird [[6]](#6), Longformer [[7]](#7))

## 11. Table of Contents (Draft)

| Chapter                   | Pages | Key Content                                                               |
| ------------------------- | ----: | ------------------------------------------------------------------------- |
| **Ch 1: Introduction**    |     5 | Motivation, problem, contributions                                        |
| **Ch 2: Background**      |    12 | Argument mining, shortcut learning, LLMs, adaptation vs model engineering |
| **Ch 3: Task & Datasets** |     6 | GAIC task, datasets, guideline cards                                      |
| **Ch 4: Methodology**     |    10 | LLM baseline, guideline-conditioned encoder, optional extensions          |
| **Ch 5: Experiments**     |     6 | Setup, baselines, metrics, evaluation protocol                            |
| **Ch 6: Results**         |    10 | Main results + shared task results                                        |
| **Ch 7: Analysis**        |     6 | Guideline sensitivity, shortcut tests, error analysis                     |
| **Ch 8: Conclusion**      |     2 | RQ answers, limitations, future work                                      |

## References

<a id="1">[1]</a> Feger, M., Boland, K., & Dietze, S. (2025). Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments. _Proceedings of ACL 2025_.

<a id="2">[2]</a> Feger, M., & Dietze, S. (2024). BERTweet’s TACO Fiesta: Contrasting flavors on the path of inference and information-driven argument mining on Twitter. _Findings of NAACL 2024_.

<a id="3">[3]</a> Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. _Nature Machine Intelligence, 2_(11), 665–673.

<a id="4">[4]</a> Shi, L., et al. (2026). From text mining to intelligent debate: Task frameworks and technological evolution in computational argumentation. _Information Processing & Management, 63_(2), 104465.

<a id="5">[5]</a> Sainz, O., García-Ferrero, I., Agerri, R., de Lacalle, O. L., Rigau, G., & Agirre, E. (2023). GoLLIE: Annotation guidelines improve zero-shot information extraction. _arXiv:2310.03668_.

<a id="6">[6]</a> Zaheer, M., et al. (2020). Big Bird: Transformers for longer sequences. _NeurIPS 2020_. _arXiv:2007.14062_.

<a id="7">[7]</a> Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. _arXiv:2004.05150_.

<a id="8">[8]</a> Jakobsen, T. S. T., Barrett, M., Søgaard, A., & Lassen, D. (2021). Spurious correlations in cross-topic argument mining. \*Proceedings of _SEM 2021_.

<a id="9">[9]</a> Cabessa, J., & Mushtaq, U. (2024). In-context learning and fine-tuning GPT for argument mining. _arXiv:2406.06699_.

<a id="10">[10]</a> Cabessa, J., Hernault, H., & Mushtaq, U. (2025). Argument mining with fine-tuned large language models. _COLING 2025_.
