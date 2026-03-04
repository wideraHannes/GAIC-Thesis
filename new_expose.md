# Do Decoder-Based LLMs Overcome Shortcut Learning in Argument Identification?

**Investigating Robustness, Context Utilization, and Fine-Tuning Effects in the GAIC @ CLEF 2026 Shared Task**

---

**Student:** Johannes Widera  
**Supervisor:** Marc Feger  
**First Reviewer:** Professor Dr. Martin Mauve  
**Second Reviewer:** Professor Dr. Stefan Dietze  
**Institution:** Heinrich Heine University Düsseldorf  
**Target Venue:** Touché @ CLEF 2026 — Generalizable Argument Identification in Context (GAIC)  
**Date:** March 2026

---

## 1. Introduction and Motivation

Do neural language models understand what an argument is, or do they merely memorize which words tend to appear in argumentative text? Feger et al. [1] presented the first large-scale re-evaluation of this question at ACL 2025. Their study across 17 benchmark datasets revealed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but fail on cross-dataset transfer (mean F1 = 0.56–0.61). Controlled manipulation experiments — removing stop words, function words, and discourse markers — showed almost no performance change (Δ ≤ 0.02). The conclusion: _state-of-the-art models learn datasets, not arguments._

This thesis investigates whether decoder-based LLMs overcome this shortcut learning problem. The work is motivated by an architectural argument: encoder models use bidirectional attention that produces largely order-invariant representations [17], while decoder models enforce causal (left-to-right) attention that makes word order a hard architectural constraint. Discourse markers like "therefore" and "because" directly shape the conditional distribution over subsequent tokens — removing them disrupts the entire computation chain. Sinha et al. [17] showed 98.7% of MNLI examples survived word-order permutation in RoBERTa, while Li et al. [20] found only 6.7% order sensitivity in PrefixLMs versus 58.4% in CausalLMs — an 8.7× difference. This generates a testable prediction: decoders should be more sensitive to argumentative structure than encoders.

The work is structured around the GAIC shared task [11], organized by the same research group behind the ACL 2025 findings. It proceeds in three parts: (1) zero-shot evaluation with a cumulative context ladder to test whether prompt-based context can replace dataset-specific training; (2) LoRA fine-tuning to directly replicate Feger et al.'s experimental design with a decoder; and (3) layer-wise probing before and after fine-tuning to mechanistically explain where shortcut learning emerges.

---

## 2. Background

### 2.1 Shortcut Learning in Decoder LLMs

Geirhos et al. [3] define shortcuts as decision rules that perform well on i.i.d. test data but fail under distribution shift. Yuan et al. [15] (EMNLP 2024) showed that decoder LLMs still exploit shortcuts — Gemini-Pro suffered a 29% accuracy drop on constituent shortcuts, and larger models showed inverse scaling under standard prompting. Sun et al. [18] revealed that decoder shortcuts are qualitatively different from encoder shortcuts: while encoders learn simple feature→label correlations, decoders learn more complex task→feature→label three-way correlations. Tang et al. [19] confirmed inverse scaling for shortcuts even during in-context learning without parameter updates.

### 2.2 Key Findings from Feger et al. (ACL 2025)

The predecessor study [1] established four findings: (1) Encoders do not learn argument structure (Δ ≤ 0.02 under manipulation). (2) Cross-dataset transfer fails systematically (97% of off-diagonal results below benchmark). (3) WRAP's task-specific contrastive pre-training helps partially. (4) Joint benchmark training improves generalization (all models above F1 = 0.63). The paper concludes by calling for decoder-based argument mining — this thesis answers that call.

Feger's experimental design is central to Part 3 of this thesis. He conducted: pairwise transfer experiments (train on dataset A, test on B → 17×17 matrix), joint training experiments (train on all-but-one, test on held-out → Table 4), and manipulation experiments (repeat both with discourse markers removed). The key finding: the manipulation tables are nearly identical to the non-manipulated tables (Δ ≤ 0.02), proving that the models did not rely on discourse structure. The manipulation diagnostic only makes sense for trained models — it tests what training taught the model to rely on.

### 2.3 Encoder vs. Decoder: The Inductive Bias Argument

Sinha et al. [17] showed near-complete order invariance in BERT-like encoders. Pham et al. [22] extended this to GLUE broadly, finding 75–90% of correct predictions survived random word shuffling. In contrast, Li et al. [20] demonstrated that CausalLMs are 8.7× more order-sensitive than PrefixLMs. LLM2Vec [23] showed that causal attention "inherently limits" decoder-only LLMs for text representation, requiring explicit conversion to bidirectional attention for good embeddings. Papadimitriou et al. [27] added nuance: word-order insensitivity primarily occurs on prototypical inputs, but for non-prototypical instances word order becomes critical — argumentation, where the distinction between premise and claim depends on discourse structure, is precisely such a non-prototypical domain.

### 2.4 Related Work: LLMs in Argument Mining

Cabessa et al. [5, 6] achieved SOTA with fine-tuned LLMs on argument mining subtasks but only evaluated within-dataset. Gorur et al. [12] showed LLMs outperform RoBERTa on relation-based AM through prompting. GoLLIE [7] demonstrated that encoding annotation guidelines as structured prompts enables zero-shot information extraction surpassing models fine-tuned on hundreds of datasets — the methodological inspiration for using annotation guidelines as prompts in this thesis. Li et al. [21] ("Echoes of BERT," 2025) showed layer-wise probing on modern decoder LLMs reveals hierarchical linguistic organization persisting across architectures. Belinkov [24] established methodological consensus for probing: linear probes on frozen representations with selectivity controls. No published work combines behavioral manipulation with mechanistic probing for argument mining.

---

## 3. The GAIC Shared Task

The GAIC shared task [11] provides 10 benchmark datasets, each with ~1.7k labeled sentences in a 60/20/20 train/dev/test split. Rich metadata includes links to annotation guideline PDFs, source documents, and dataset papers. Primary evaluation is on a newly created evaluation-only dataset (addressing LLM contamination), with secondary evaluation on held-out test splits. The primary metric is Macro F1.

Context availability differs across datasets:

| Dataset    | Paper | Document Context | Guidelines |
| ---------- | ----- | ---------------- | ---------- |
| ABSTRCT    | Yes   | Yes              | Yes        |
| ARGUMINSCI | Yes   | Yes              | Yes        |
| PE         | Yes   | Yes              | Yes        |
| USELEC     | Yes   | Yes              | Yes        |
| FINARG     | Yes   | Yes              | —          |
| SCIARK     | Yes   | Yes              | —          |
| ACQUA      | Yes   | —                | —          |
| AEC        | Yes   | —                | —          |
| AFS        | Yes   | —                | —          |
| IAM        | Yes   | —                | —          |

**Data split strategy:** The dev split is used for all Parts 1+2 experiments (~340 sentences/dataset, 170 per class). The test split is reserved for Part 3 evaluation and the GAIC submission. The train split is used for Part 3 LoRA fine-tuning. This ensures clean methodology: evaluation data is never used for tuning decisions. Contamination testing runs across all splits as it measures a property of the model, not the experimental design.

---

## 4. Research Questions

This thesis asks one overarching question — _do decoder-based LLMs overcome the shortcut learning problem identified in encoder-based models for argument identification?_ — decomposed into three sub-questions:

### RQ1: Can zero-shot LLMs with rich context match or exceed trained encoders for cross-dataset argument identification?

**Rationale:** Feger et al. trained encoders on 850 labeled examples per dataset. We replace training with a cumulative context ladder (generic definition → dataset-specific definition → annotation guidelines → document context). If context-grounded prompting matches joint-trained encoders (Feger's Table 4, F1 ≈ 0.63–0.74), it demonstrates that rich metadata is a viable alternative to multi-dataset training for generalization.

**Hypothesis:** Dataset-specific context will improve F1 over generic definitions, with annotation guidelines providing the largest gains, consistent with the GoLLIE finding [7] that guideline-grounded prompts improve zero-shot performance.

**Preliminary evidence:** Pilot experiments show C0→C1 improves average Macro F1 from 0.58 to 0.63 across all 10 datasets (n=30/dataset). Zero-shot manipulation sensitivity: Δ_feger = −0.10 and Δ_shuffle = −0.29 on ABSTRCT (n=100), representing 5× and 14× the encoder effect respectively.

### RQ2: Does fine-tuning a decoder introduce encoder-like shortcut learning?

**Rationale:** Feger's manipulation diagnostic (Δ ≤ 0.02) only makes sense for trained models — it tests what training taught the model to rely on. By LoRA fine-tuning a decoder on GAIC data and re-running the manipulation, we directly compare: does a trained decoder show the same shortcut patterns as trained encoders? This is the apple-to-apple comparison that connects zero-shot findings (RQ1) to Feger's encoder results.

**Hypothesis:** Fine-tuning will reduce Δ compared to zero-shot, as the model learns dataset-specific content-word patterns. The magnitude of this reduction indicates whether shortcut learning is architecture-dependent or training-dependent.

### RQ3: Where in the network does shortcut learning emerge during fine-tuning?

**Rationale:** Layer-wise probing before and after LoRA fine-tuning reveals how internal representations change. If probing accuracy on manipulated sentences converges with original sentences after training (i.e., the model stops distinguishing them internally), that is mechanistic evidence of shortcut adoption — visible in the representations before it manifests in output behavior.

**Hypothesis:** Fine-tuning will shift the representational profile: probing curves for manipulated vs. original sentences will diverge less after training, particularly in later layers where task-specific features are encoded.

---

## 5. Methodology

### 5.1 Model Selection

All models are standard instruction-tuned (non-reasoning) to ensure a clean comparison with Feger et al.'s standard fine-tuned encoders. Reasoning models (e.g., chain-of-thought variants) are explicitly excluded because (a) they introduce an additional variable not present in Feger's design, (b) their hidden states encode reasoning processes rather than classification features, confounding probing results, and (c) they are harder to fine-tune due to output format conflicts with simple classification labels.

We focus on the Mistral family to enable controlled scale comparisons with identical tokenization and training methodology, complemented by GPT-4.1 as an independent cross-provider validation.

| #   | Model                  | Parameters | API String              | Role                                                                         |
| --- | ---------------------- | ---------- | ----------------------- | ---------------------------------------------------------------------------- |
| 1   | Ministral 3B Instruct  | 3B         | `ministral-3b-latest`   | Small anchor. Tests whether even a 3B decoder shows structural sensitivity.  |
| 2   | Ministral 14B Instruct | 14B        | `ministral-14b-latest`  | **Protagonist.** Zero-shot (Parts 1+2), LoRA fine-tuning + probing (Part 3). |
| 3   | Mistral Small 24B      | 24B        | `mistral-small-latest`  | Mid-scale Mistral. Pilot model — continuity with preliminary results.        |
| 4   | Mistral Medium 3       | ~70B       | `mistral-medium-latest` | Large-scale Mistral. Tests whether scale improves robustness.                |
| 5   | GPT-4.1                | Frontier   | `gpt-4.1`               | Cross-provider frontier validation. Non-reasoning by design.                 |

**Scale story (within Mistral):** 3B → 14B → 24B → ~70B. Same tokenizer, same training philosophy, same prompt format. Any performance difference is attributable to scale alone. GPT-4.1 serves as an independent check — if it shows the same Δ pattern as the Mistral models, the finding generalizes beyond one model family.

**Infrastructure:** All Parts 1+2 experiments run through the Mistral API (api.mistral.ai) and OpenAI API, both accessed via the OpenAI Python SDK with different base URLs. This enables a single codebase for all five models. Part 3 (LoRA fine-tuning and probing) runs on Lightning AI with dedicated GPU (A10G 24GB). The protagonist model (Ministral-3-14B-Instruct) has open weights on HuggingFace (`mistralai/Ministral-3-14B-Instruct-2512`) under Apache 2.0, enabling full access to hidden states for probing. At FP8 precision it requires ~14GB VRAM; with QLoRA (4-bit base) fine-tuning requires ~8GB.

### 5.2 Preprocessing: Context Extraction

Before running experiments, structured context is extracted from GAIC's raw materials:

**Argument definition extraction.** Each dataset's original paper is processed via PDF-to-text extraction, then passed to a language model with a structured output schema requesting: (1) a positive definition (what IS an argument in this dataset), (2) a negative definition (what is NOT an argument), and (3) 1–2 examples per class. Only the conceptual definition is retained — no dataset statistics or methodological details.

**Annotation guideline extraction.** Four datasets (ABSTRCT, ARGUMINSCI, PE, USELEC) include annotation guideline PDFs. These are processed to extract the operational decision rules an annotator would apply.

**Document context extraction.** Six datasets provide source documents. For each sample, the two sentences preceding the target sentence are extracted to provide local discourse context.

### 5.3 Prompt Design

All experiments use a two-message prompt structure:

- **System prompt:** Instructs the model to act as a dataset annotator and classify the input as "Argument" or "No-Argument." Context (definition, guidelines, document context) is injected here. The system prompt ends with an explicit classification constraint: "Respond with exactly one word: 'Argument' or 'No-Argument'."
- **User prompt:** Contains only the target sentence.

The same template is used across all models, conditions, and datasets — only the injected context block varies. This placement follows Liu et al. [26] ("Lost in the Middle"), which showed that decoder models use information at the beginning and end of the context window most effectively.

### 5.4 Parts 1+2: Zero-Shot Evaluation with Context Ladder and Manipulation

Parts 1 and 2 form a single unified experiment, separated only in the thesis writing. Part 1 corresponds to the baseline condition (C0); Part 2 adds the context ladder (C1–C3). All are zero-shot — no training is involved.

**Context conditions (cumulative ladder):**

| Condition                       | Context in System Prompt                                                              | What it isolates                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| C0: Generic                     | "An argument is a statement that makes a claim supported by reasoning or evidence..." | Baseline: model's pre-trained notion of "argument"                            |
| C1: Dataset-specific definition | 3–10 sentence definition extracted from the dataset's paper                           | Definition gap: does aligning to the dataset's theoretical framing help?      |
| C2: C1 + annotation guidelines  | C1 + operational decision rules from annotation guidelines                            | Operationalization gap: do concrete annotation rules add value beyond theory? |
| C3: C2 + document context       | C2 + 2 preceding sentences from source document                                       | Discourse context gap: does local context disambiguate borderline cases?      |

C0–C1 are evaluated on all 10 GAIC datasets. C2 is additionally evaluated on the 4 datasets with guidelines (ABSTRCT, ARGUMINSCI, PE, USELEC). C3 is evaluated on datasets with both guidelines and document context.

**Manipulation conditions (applied to the target sentence in the user prompt only):**

| Condition              | Description                                                       | What it tests                                                                    |
| ---------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| M0: Original           | Unmodified sentence                                               | Baseline performance                                                             |
| M1: Feger manipulation | Remove stop words, function words, discourse markers, punctuation | Sensitivity to argumentative scaffolding (direct comparison to Feger's Δ ≤ 0.02) |
| M2: Word shuffle       | Randomly permute word order (all words preserved)                 | Sensitivity to word order (tests causal attention hypothesis)                    |

**Data:** Dev split of all 10 GAIC datasets (~170 sentences per class per dataset).

**Metrics:** Macro F1 per dataset/model/condition. Δ_feger = F1(M0) − F1(M1). Δ_shuffle = F1(M0) − F1(M2). Direct comparison to Feger et al.'s Table 4 (joint-trained encoder results) and manipulation tables.

**Contamination control:** Sentence-completion contamination test applied to all four models across all splits. The first half of each sentence is provided as a prompt; ROUGE-L similarity between the model's completion and the actual second half is measured. Affected model-dataset pairs are flagged in results tables.

### 5.5 Part 3: Fine-Tuning and Manipulation Replication

**Question:** Does fine-tuning a decoder produce the same shortcut patterns Feger found in encoders?

**Design (mirroring Feger et al.'s Table 4):**

- **Model:** Ministral-3-14B-Instruct with LoRA adapters
- **Training data:** GAIC train splits (all 10 datasets jointly, ~17k sentences)
- **Prompt during training:** Same template as Parts 1+2, using the best-performing context condition (likely C1, as it is available for all 10 datasets)
- **Evaluation:** Test splits of all 10 GAIC datasets + new evaluation-only dataset
- **LoRA configuration:** QLoRA (4-bit base model + FP16 adapters), r=16, α=32, target modules: q_proj, v_proj. Training: 3 epochs, batch size 32, learning rate grid [2e-5, 5e-5] (matching Feger's GLUE hyperparameter grid). Tuned on dev split.

**Experimental conditions:**

1. **Joint training → test on each dataset's test split** (mirrors Feger's Table 4)
2. **Manipulation on test sets:** Re-run with Feger manipulation (M1) on all test splits
3. **Direct Δ comparison:** Feger's encoder Δ (≤ 0.02) vs. zero-shot decoder Δ (~0.10 from Parts 1+2) vs. fine-tuned decoder Δ (?)

**Interpretation framework:**

| Outcome            | Δ (fine-tuned decoder) | Interpretation                                                      |
| ------------------ | ---------------------- | ------------------------------------------------------------------- |
| Encoder-like       | ≤ 0.05                 | Shortcut learning is training-dependent, not architecture-dependent |
| Partial resistance | 0.05–0.15              | Causal attention provides partial protection against shortcuts      |
| Full robustness    | ≥ 0.15                 | Decoder architecture fundamentally resists shortcut learning        |

**Infrastructure:** All fine-tuning and evaluation runs on Lightning AI (A10G 24GB GPU). Ministral-3-14B at FP8 ≈ 14GB, at 4-bit QLoRA ≈ 8GB base + adapters — well within GPU memory. Probing with `output_hidden_states=True` in small batches fits within 24GB at FP8.

### 5.6 Mechanistic Ablation: Layer-Wise Probing

**Question:** Where in the network does the model encode argument-relevant features, and how does fine-tuning change this?

**Model:** Ministral-3-14B-Instruct (open weights, Apache 2.0, `mistralai/Ministral-3-14B-Instruct-2512` on HuggingFace)

**Method:**

1. Run forward pass with `output_hidden_states=True`
2. Extract hidden states at the final token position at each transformer layer
3. Train a linear probe (sklearn LogisticRegression) per layer on frozen representations to predict Argument/No-Argument
4. 5-fold cross-validation on dev split
5. Report accuracy per layer
6. Repeat with randomized labels → selectivity = real accuracy − randomized accuracy (following Hewitt & Liang [25])

**Conditions:**

- **Before fine-tuning:** Probe on M0 (original), M1 (Feger manipulation), M2 (shuffle)
- **After LoRA fine-tuning:** Repeat probing on same sentences with same conditions

**Metrics:** Per-layer probing accuracy, Expected Layer / Center of Gravity (following Tenney et al. [28]), selectivity per layer.

**Expected output:** Layer-wise accuracy curves showing (a) where argument features form, (b) where manipulation breaks them, and (c) whether fine-tuning causes the model to stop distinguishing manipulated from original sentences internally — the mechanistic signature of shortcut adoption.

**Infrastructure:** FP8 inference on Lightning AI (A10G 24GB). Hidden state extraction in batches of ~8 sentences to manage memory. All probing computation (logistic regression) runs on CPU.

---

## 6. Expected Contributions

**Empirical:** First systematic comparison of shortcut learning between encoder and decoder architectures in argument mining. Evidence on whether shortcut learning is architecture-dependent or training-dependent. Quantification of how cumulative context (definitions, guidelines, document context) affects zero-shot argument identification across heterogeneous datasets.

**Methodological:** Cumulative context ladder as a standardized framework for zero-shot argument identification with variable metadata availability. Layer-wise probing before/after fine-tuning as a diagnostic for shortcut emergence — first application to computational argumentation.

**Practical:** Competitive GAIC 2026 submission with reproducible codebase.

---

## 7. Scope and Requirements

### 7.1 Core Requirements

- Theoretical framing connecting argument mining, shortcut learning, and decoder LLM architectures
- Zero-shot context ladder evaluation across multiple models and datasets (Parts 1+2)
- Contamination analysis for all models

### 7.2 Extensions (in priority order)

1. LoRA fine-tuning on GAIC with manipulation re-evaluation (Part 3, RQ2)
2. Layer-wise probing before/after fine-tuning (Part 3, RQ3)
3. GAIC shared task submission (deadline: 7 May 2026)
4. LoRA rank ablation (r=4, 8, 16, 32, 64) testing whether low rank preferentially captures shortcuts

---

## 8. Timeline

**Duration:** 11 February – 11 August 2026 (6 months)

| Phase                            | Weeks | Activities                                                                                                                                      |
| -------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Parts 1+2: Zero-shot experiments | 1–4   | Context extraction finalization, context ladder + manipulation experiments across 4 models × 10 datasets via Portkey API, contamination testing |
| Analysis + writing Ch. 4         | 5–6   | Context ladder plots, manipulation Δ tables, first draft of experimental chapter                                                                |
| Part 3: Fine-tuning + probing    | 7–10  | LoRA training on Lightning AI, test evaluation, manipulation re-evaluation, layer-wise probing before/after                                     |
| GAIC submission                  | 11    | Best system packaged for TIRA submission (deadline: 7 May)                                                                                      |
| Notebook paper                   | 12    | GAIC notebook paper (deadline: 28 May)                                                                                                          |
| Thesis writing                   | 13–24 | Full draft, revision cycles with supervisor, defense preparation                                                                                |
| **Submission**                   | **—** | **11 August 2026**                                                                                                                              |

---

## 9. Draft Table of Contents

| Chapter                 | Est. Pages | Content                                                                                                        |
| ----------------------- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| 1. Introduction         | 4          | Motivation, problem statement, contributions                                                                   |
| 2. Background           | 10         | Argument mining, shortcut learning, encoder vs. decoder theory, probing methodology                            |
| 3. The GAIC Shared Task | 3          | Task description, 10 datasets, context availability, data split strategy                                       |
| 4. Zero-Shot Evaluation | 12         | Context ladder design, manipulation experiments, contamination analysis, results across 4 models × 10 datasets |
| 5. Fine-Tuning Effects  | 8          | LoRA training, Feger Table 4 replication, Δ comparison (encoder vs. zero-shot decoder vs. fine-tuned decoder)  |
| 6. Mechanistic Analysis | 8          | Layer-wise probing before/after fine-tuning, visualization of shortcut emergence                               |
| 7. Discussion           | 5          | Integration across parts, comparison to Feger, limitations                                                     |
| 8. Conclusion           | 2          | Summary, implications for argument mining, future work                                                         |

---

## 10. References

[1] Feger, M., Boland, K., & Dietze, S. (2025). Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments. _ACL 2025_, 23900–23915.
[2] Feger, M., & Dietze, S. (2024). BERTweet's TACO Fiesta. _Findings of NAACL 2024_, 2256–2266.
[3] Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. _Nature Machine Intelligence_, 2(11), 665–673.
[4] Jakobsen, T. S. T., et al. (2021). Spurious correlations in cross-topic argument mining. _\*SEM 2021_, 263–277.
[5] Cabessa, J., Hernault, H., & Mushtaq, U. (2025). Argument mining with fine-tuned LLMs. _COLING 2025_.
[6] Cabessa, J., & Mushtaq, U. (2024). In-context learning and fine-tuning GPT for argument mining. arXiv:2406.06699.
[7] Sainz, O., et al. (2024). GoLLIE: Annotation guidelines improve zero-shot information extraction. _ICLR 2024_.
[8] Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. _ICLR 2022_.
[11] GAIC Shared Task. (2026). Touché @ CLEF 2026. https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html
[12] Gorur, D., Rago, A., & Toni, F. (2025). Can LLMs perform relation-based argument mining? _COLING 2025_.
[15] Yuan, Y., et al. (2024). Do LLMs overcome shortcut learning? _EMNLP 2024_, 12188–12200.
[16] Golchin, S., & Surdeanu, M. (2024). Time travel in LLMs: Tracing data contamination. _ICLR 2024_.
[17] Sinha, K., et al. (2021). Unnatural language inference. _ACL 2021_ (Best Paper Award).
[18] Sun, Y., et al. (2024). Exploring and mitigating shortcut learning for generative LLMs. _LREC-COLING 2024_.
[19] Tang, S., et al. (2023). Large language models can be lazy learners. _Findings of ACL 2023_.
[20] Li, Y., et al. (2024). Addressing order sensitivity of in-context demonstration examples in causal language models. arXiv:2402.15637.
[21] Li, Z., et al. (2025). Echoes of BERT: Do modern language models rediscover the classical NLP pipeline? arXiv:2506.02132.
[22] Pham, T., et al. (2021). Out of order. _Findings of ACL 2021_.
[23] BehnamGhader, P., et al. (2024). LLM2Vec. _COLM 2024_.
[24] Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_, 48(1), 207–219.
[25] Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. _EMNLP 2019_.
[26] Liu, N. F., et al. (2024). Lost in the middle: How language models use long contexts. _TACL 2024_.
[27] Papadimitriou, I., et al. (2022). When classifying grammatical role, BERT doesn't care about word order... except when it matters. _ACL 2022_.
[28] Tenney, I., et al. (2019). BERT rediscovers the classical NLP pipeline. _ACL 2019_.
[29] Shuttleworth, B., et al. (2024). LoRA vs full fine-tuning: An illusion of equivalence. _ICML 2025_.
[30] Merchant, A., et al. (2020). What happens to BERT embeddings during fine-tuning? _BlackboxNLP 2020_.

---

## Appendix: Preliminary Experiments

### A.1 Manipulation Sensitivity (Zero-Shot)

**Setup:** Mistral-Small-24B-Instruct (via Portkey), zero-shot with dataset-specific definition (C1), ABSTRCT dataset (n=100 balanced sentences from dev split).

| Manipulation                                          | Macro F1 | Δ from M0 |
| ----------------------------------------------------- | -------- | --------- |
| M0: Original                                          | 0.72     | —         |
| M1: Feger (remove function words + discourse markers) | 0.62     | −0.10     |
| M2: Word shuffle (preserve all words, permute order)  | 0.43     | −0.29     |

The Feger manipulation Δ (−0.10) is 5× the encoder effect (Δ ≤ 0.02). The shuffle Δ (−0.29) is 14× the encoder effect, consistent with the causal attention hypothesis: word order matters fundamentally for decoder models.

### A.2 Context Ladder (Zero-Shot)

**Setup:** Mistral-Small-24B-Instruct (via Portkey), zero-shot, all 10 GAIC datasets (n=30/dataset balanced from dev split).

| Condition                       | Mean Macro F1 (10 datasets) |
| ------------------------------- | --------------------------- |
| C0: Generic definition          | 0.58                        |
| C1: Dataset-specific definition | 0.63                        |

Dataset-specific definitions improve performance by +0.05 on average. Gains are non-uniform: AFS and ACQUA benefit substantially; others show smaller or negative effects, suggesting context value depends on how much the generic definition diverges from the dataset's actual annotation criteria.
