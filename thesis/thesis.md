# Shortcut Robustness and Context Utilization

# in LLM-Based Argument Mining

```
Master’s Thesis
by
```

## Johannes Widera

```
born in
Düsseldorf
```

```
submitted to
```

```
Professorship for Computer Networks
Prof. Dr. Martin Mauve
Heinrich-Heine-University Düsseldorf
```

```
August 2026
```

```
Supervisor:
Marc Feger, M. Sc.
```

# Abstract

Hier beschreibe ich nun auf einer Seite, welches Problem ich betrachtet, wie ich es gelöst
habe und was die Hauptergebnisse meiner Arbeit sind.

Laut Prüfungsordnung muss es eine Zusammenfassung geben, die nicht länger als eine Seite
ist.

```
ii
```

# Acknowledgments

Viele Personen haben mich während meiner Arbeit an dieser Thesis unterstützt und ich
möchte ihnen danken.

Bei Abschlussarbeiten ist in der Regel keine Danksagung erforderlich. Wenn man keine
Danksagung haben möchte, kann man \input{acknowledgements} in der master-
Datei auskommentieren.

```
iii
```

## Contents

**List of Figures vii**

**List of Tables viii**

- 1. Introduction List of Listings ix
  - 1.1. Motivation
  - 1.2. Problem Statement
  - 1.3. Research Questions
  - 1.4. Contributions
  - 1.5. Thesis Structure
- 2. Background
  - 2.1. Argument Mining
  - 2.2. Encoder vs. Decoder Architectures
  - 2.3. Large Language Models for Argument Mining
  - 2.4. Shortcut Learning
  - 2.5. Data Contamination in LLMs
  - 2.6. In-Context Learning
- 3. The GAIC Shared Task
  - 3.1. Task Definition
  - 3.2. Datasets
  - 3.3. Context Availability
  - 3.4. Annotation Scheme Heterogeneity
- 4. Methodology
  - 4.1. Models
  - 4.2. Context Extraction and Prompt Design Contents
    - 4.2.1. Context Extraction Pipeline
    - 4.2.2. The Context Ladder
    - 4.2.3. Prompt Structure
  - 4.3. Text Manipulation Experiments (RQ2)
    - 4.3.1. M1: Content-Only Reduction
    - 4.3.2. M2: Word Shuffle
    - 4.3.3. Delta Metrics
    - 4.3.4. Context–Manipulation Interaction
  - 4.4. Contamination Detection (RQ3)
    - 4.4.1. Motivation
    - 4.4.2. The Data Contamination Quiz Method
    - 4.4.3. Contamination Thresholds and Interpretation
  - 4.5. Experimental Design
    - 4.5.1. Data Splits
    - 4.5.2. Sample Sizes
    - 4.5.3. Evaluation Metric
    - 4.5.4. Reproducibility
    - 4.5.5. Comparison Baselines
  - 4.6. Summary
- 5. Results
  - 5.1. RQ1: Can Zero-shot Decoders match trained Encoders?
  - 5.2. RQ2: Are Decoders More Sensitive to Argumentative Structure?
    - 5.2.1. Does Context increase Manipulation Sensitivity?
    - surements? 5.3. RQ3: Does Contamination Confound Performance and Manipulation Mea-
    - 5.3.1. Contamination Levels by Model and Dataset
- 6. Discussion
  - 6.1. Integration of Results
  - 6.2. Comparison to Encoder Baselines
  - 6.3. The Role of Context
  - 6.4. Implications for Argument Mining
  - 6.5. Limitations
- 7. Conclusion Contents
  - 7.1. Summary
  - 7.2. Contributions
  - 7.3. Future Work
- A. Weitere Abbildungen
  - A.1. Datensatzeigenschaften
- Bibliography

## List of Figures

```
4.1. The three-stage context extraction pipeline................... 19
4.2. Prompt structure by context level. C0 omits all colored blocks and has a
simplified task "Classify whether the sentence is an argument or not."; each
subsequent level adds context cumulatively................... 21
4.3. Pydantic schema for Structured output in Phase 1............... 25
```

```
5.1. TBD....................................... 32
5.2. TBD....................................... 32
5.3. TBD....................................... 34
5.4. DCQ contamination levels across datasets and models, showing systematic
variation by model and dataset. Error bars indicate the range across BCQ
positions..................................... 35
```

```
A.1. Nochmal das Logo............................... 39
```

```
vii
```

## List of Tables

```
3.1. Overview of the 10 GAIC benchmark datasets with their annotation schemes
and formal argument structure where applicable. A = ⟨C, P⟩ denotes the
canonical argument kernel (Equation 2.1); datasets marked with A =⟨s⟩ use a
flat single-component scheme where no claim–premise distinction is annotated. 13
3.2. Context availability across the 10 GAIC datasets. A checkmark (✓) indicates
that the context type is provided; a dash (–) indicates it is unavailable..... 14
```

```
4.1. Models evaluated in this thesis. All are instruction-tuned, non-reasoning vari-
ants........................................ 18
4.2. Context ladder conditions. Each level is cumulative: C3 includes all content
from C0 through C2............................... 20
4.3. Text manipulation conditions. M0 serves as the baseline against which deltas
are computed................................... 22
4.4. Summary of the experimental methodology by research question........ 28
```

```
5.1. Macro-F1 scores by model, context level, and dataset. Best per column in bold. 31
5.2. Performance degradation after text manipulation (C1 context). Format:∆content
/∆shu f f le. Bold indicates largest drop per manipulation type.......... 34
5.3. Comparison of argument identification approaches on the GAIC benchmark.
SOTA represents in-distribution supervised performance (upper bound). En-
coder shows the best cross-dataset transfer result from Feger2025-kz. KNN
uses k-nearest neighbor retrieval (k = 50). LLM is our zero-shot decoder-
based approach evaluated on the dev set. Best zero-shot method per row in
bold........................................ 36
```

```
viii
```

## List of Listings

```
ix
```

## Chapter 1.

## Introduction

### 1.1. Motivation

- Argument mining – the automatic identification of argumentative structures in text – is
  a foundational task for discourse analysis across domains like political debate, scientific
  reasoning, and online deliberation - Over the past years, encoder-based language models
  (BERT, RoBERTa, DistilBERT) have been the dominant approach, achieving strong bench-
  mark scores on individual datasets - But this line of research is hitting a wall: Feger, Boland,
  and Dietze (ACL 2025) showed that reported SOTA performance does not reflect genuine
  understanding of arguments – models learn datasets, not arguments - Cross-dataset transfer
  performance drops sharply, and controlled manipulations reveal that models rely on content-
  word shortcuts rather than argumentative structure - The field of encoder-based argument
  mining appears saturated – incremental improvements on isolated benchmarks no longer
  translate into real progress
- At the same time, these encoder systems ignore a resource that human annotators rely on
  heavily: context - When humans decide whether a sentence is argumentative, they draw
  on the surrounding text, the definition of what counts as an argument in a given setting,
  and the annotation guidelines that operationalize that definition - Standard encoder pipelines
  receive isolated sentences with no access to any of this – and are then expected to generalize
  across datasets with fundamentally different annotation schemes - The question this thesis
  asks is: can we design systems that utilize this additional context and that understand what
  an argument is without relying on shortcuts?

```
Chapter 1. Introduction
```

- Large language models offer a natural way to test this – they can receive rich context di-
  rectly through prompting, and they can be applied zero-shot, without any task-specific train-
  ing - The GAIC shared task (Touché @ CLEF 2026), which grew out of the generalization
  problems identified by Feger, Boland, and Dietze (2025), provides an ideal testbed: 10 argu-
  ment mining datasets with unified splits, each accompanied by annotation guidelines, dataset
  papers, and source documents - This thesis uses GAIC as a playground to investigate these
  questions from an LLM perspective - Starting from the most naive zero-shot classification,
  context is added iteratively – first a task definition, then annotation guidelines, then the source
  document – to test whether and how context helps, and how this compares to trained encoder
  systems

### 1.2. Problem Statement

- Encoder-based AM is saturated: strong in-distribution results but poor generalization and
  reliance on shortcuts (Feger, Boland, and Dietze, 2025) - Existing approaches ignore avail-
  able context (guidelines, definitions, source documents) that humans rely on - It is unknown
  whether LLMs, with their ability to process rich context and their different architecture,
  can overcome these limitations - Zero-shot application avoids training-induced shortcuts but
  introduces new threats: data contamination and potential insensitivity to actual argument
  structure - Fine-tuning LLMs on AM data might reintroduce the shortcut patterns seen in
  encoders

### 1.3. Research Questions

RQ1: Can zero-shot LLMs match trained encoder baselines on argument identification, and
does adding context improve performance?

RQ2: Are LLMs more sensitive to argumentative structure than encoders, as measured by
performance drops under text manipulation?

RQ3: Does data contamination confound performance and manipulation measurements?

```
Chapter 1. Introduction
```

### 1.4. Contributions

- First shortcut robustness comparison between trained encoders and zero-shot LLMs in argu-
  ment mining, directly extending Feger, Boland, and Dietze (2025) - A context ladder (c0–c3)
  that iteratively adds task definition, annotation guidelines, and source documents to zero-shot
  prompts – testing whether the context that encoder pipelines ignore actually helps - Sentence
  shuffling as a new manipulation diagnostic complementing Feger et al.’s content-word abla-
  tions - Contamination audit (DCQ/BCQ) across 3 models and 10 GAIC datasets - Evidence
  on whether fine-tuning reintroduces encoder-like shortcut patterns

### 1.5. Thesis Structure

- Chapter 2: Background (argument mining, shortcut learning, encoder vs. decoder architec-
  tures, LLMs for AM, data contamination) - Chapter 3: The GAIC shared task and datasets -
  Chapter 4: Methodology - Chapter 5: Results (organized by RQ) - Chapter 6: Discussion -
  Chapter 7: Conclusion

## Chapter 2.

## Background

### 2.1. Argument Mining

Argument mining refers to the automated process of detecting and extracting the structure of
reasoning and inference as expressed in natural language [1]. The field draws on computa-
tional linguistics, argumentation theory, and rhetoric, and has been applied across domains
including law, science, politics, and online deliberation.

#### What is an Argument?

Following the flat or simple argument model [2, 3], arguments can be represented in terms of
their structural composition, sometimes referred to as the argument kernel [4]. Formally, an
argument is defined as a canonical structure

```
A =⟨C, P⟩, P ={P 1 ,..., Pk}, k≥ 1 , (2.1)
```

where C denotes the conclusion and P the set of premises [5]. The conclusion states the
position being argued for, while the premises provide the supporting evidence or reasoning
that justifies it. This structural view is intentionally minimal: it abstracts away from surface
form, domain, and rhetorical strategy, capturing only the inferential relationship between

```
Chapter 2. Background
```

premises and conclusion.

#### Core Subtasks

Building on this definition, argument mining decomposes into three core subtasks of in-
creasing complexity. Argument identification is the binary task of deciding whether a given
sentence or span is argumentative at all, which is the focus of this thesis. Component classifi-
cation goes further by labeling identified argumentative units as claims, premises, evidence,
or other roles depending on the annotation scheme. Relation extraction then determines the
logical connections between components, such as support, attack, or rebuttal.

In practice, what counts as a conclusion or premise varies considerably across annotation
schemes. As discussed in Section ??, the 10 GAIC datasets operationalize the argument
kernel differently, leading to fundamentally incompatible notions of what constitutes an ar-
gument in context. This heterogeneity is one of the central challenges the field has not yet re-
solved: inter-annotator agreement is often only moderate, and models trained on one dataset
generalize poorly to others [6].

#### Methodological Progression

[7] organize the methodological history of argument mining into three paradigms that closely
follow the broader evolution of NLP, which provides a useful lens for situating the approach
taken in this thesis.

The first paradigm is Feature Engineering, where the prediction function takes the form
y = f(ψ(D)): a hand-crafted feature function ψ(·) transforms raw argumentative text D into
a numerical representation, which is then fed to a statistical classifier f(·) such as an SVM
[7]. Early work relied on discourse markers and hand-crafted linguistic features [8, 9], fol-
lowed by feature-engineered models trained on part-of-speech tags and sentiment features
[10, 11]. While this approach provided interpretable models and valuable linguistic insights,
the feature sets were domain-specific and laborious to construct, which severely limited gen-
eralization.

```
Chapter 2. Background
```

The second paradigm is Model Engineering, formalized as y =M (φ(D)), where φ(·) de-
notes a pre-trained model andM (·) a task-specific neural architecture designed on top of it
[7]. With the introduction of BERT [12] and RoBERTa [13], fine-tuning pre-trained encoder
models per dataset became the dominant approach and achieved state-of-the-art results across
most benchmarks. However, as [6] showed, these gains do not reflect genuine understand-
ing of argumentative structure but rather dataset-specific shortcut learning, and cross-dataset
transfer performance drops sharply as a result.

The third and current paradigm is Adaptation Engineering, where the prediction takes the
form y = F(γ(D)): a large language model F(·) receives a task-specific prompt γ(·) con-
structed from the input [7]. Unlike the previous paradigms, the model architecture remains
fixed; researchers instead focus on adapting the model through prompt engineering or in-
struction tuning. [7] note that this shift has been rapid: adaptation engineering grew from
virtually no presence before 2022 to 86% of publications in the field by 2024. The rise of
decoder-based large language models offers a fundamentally different perspective on argu-
ment mining: instead of training on labeled data, these models can be prompted zero-shot
with task definitions and annotation guidelines, as explored in this thesis (see Chapter 4).

### 2.2. Encoder vs. Decoder Architectures

Encoders (BERT, RoBERTa, DeBERTa): - Bidirectional self-attention, [CLS] token aggrega-
tion - Representation is somewhat order-invariant - Sinha et al. 2021 ("Unnatural Language
Inference"): shuffled word order barely degrades encoder NLI performance – confirms bag-
of-words-like behavior - Explains why Feger’s encoder baselines show small deltas under
manipulation

Decoders (GPT, Llama, Mistral): - Causal attention: each token only sees previous tokens -
Word order enforced by attention mask, not just positional embeddings - Discourse markers
like "therefore" shape all downstream computation – removing them disrupts the causal chain

- Classification via generation, not [CLS] pooling

Prediction: decoders should show larger∆shuffleand∆contentthan encoders if they genuinely
process sequential structure. If they also take shortcuts, drops should be small. This is the
core test of RQ2.

```
Chapter 2. Background
```

Fine-tuning might push decoders toward encoder-like shortcuts – motivates RQ4.

TODO: read Sinha et al. 2021 more carefully for methodology TODO: check Pham et al.
2021 ("Out of Order") for more on word order sensitivity

### 2.3. Large Language Models for Argument Mining

Recent work applying LLMs to AM: - Cabessa et al. (COLING 2025): fine-tuned LLMs
achieve SOTA across AM subtasks - Cabessa & Mushtaq (2024): GPT-4 with ICL matches
fine-tuned encoder baselines - Gorur et al. (COLING 2025): LLMs outperform RoBERTa on
relation-based AM via prompting - R. Schäfer: on integrating LLMs into argument annota-
tion workflows

Guidelines as prompts: - Sainz et al. 2023 (GoLLIE): annotation guidelines as structured
prompts for zero-shot IE - Surpassed models fine-tuned on hundreds of datasets - Direct
precedent for my context ladder (c0 -> c3): more context = better zero-shot performance -
Open question: does more context also increase manipulation sensitivity?

General LLM shortcut robustness: - Xie et al. 2024: "Do LLMs overcome shortcut learning?"

- evaluated shortcuts in LLMs generally but not for AM

The gap this thesis fills: - Nobody has combined (a) zero-shot LLM evaluation on AM, (b)
manipulation testing, (c) contamination control, and (d) fine-tuning robustness in one study -
10 datasets, 3 models, 4 context levels, 2 manipulation types

TODO: read Xie et al. 2024 closely – their manipulation methodology might inform mine
TODO: check if there’s more recent GoLLIE follow-up work

### 2.4. Shortcut Learning

[14] define shortcuts as decision rules that perform well on i.i.d. test data but fail under distri-
bution shift. This is pervasive across deep learning: image classifiers relying on background

```
Chapter 2. Background
```

texture rather than object shape, language models exploiting superficial lexical cues rather
than semantic understanding.

To be precise, all learning systems perform some shortcut learning. Humans memorize pat-
terns for tests, animals solve mazes via unintended cues [14]. The issue is not shortcuts per
se, but that they break under distribution shifts.

Shortcuts in NLP specifically: - NLI: hypothesis-only baselines achieve high accuracy with-
out reading the premise (cite Poliak et al. 2018, Gururangan et al. 2018) - Sentiment analysis:
models latch onto negation words rather than understanding composition - QA: lexical over-
lap between question and passage suffices for many benchmarks

Shortcut learning in argument mining: - Feger et al.’s encoder experiments: text manip-
ulations (content removal, shuffling) show encoders are surprisingly robust to destroying
argumentative structure - Drops of∆≤ 0 .02 after removing function words – suggests bag-
of-words-like behavior - This is the direct motivation for this thesis: do decoders behave
differently?

TODO: look into McCoy et al. 2019 (HANS dataset) for more NLI shortcut examples
TODO: check if Niven Kao 2019 (argument comprehension shortcuts) is relevant here

### 2.5. Data Contamination in LLMs

Benchmark contamination in the training data of LLMs is a common thing and dilutes the
expressiveness of comparisons based on benchmarks thus in every domain its nowadays a
topic.

Thus when utilizing large language models we must investigate such contamination

- LLM training corpora (Common Crawl, arXiv, GitHub) likely include public benchmarks -
  If a model has seen test data during pretraining, "zero-shot" performance is inflated - Espe-
  cially concerning for older, widely-cited datasets

Detection: - Golchin & Surdeanu (ICLR 2024): Data Contamination Quiz (DCQ) – prompt
the model to complete a dataset sentence; if it can, it has likely seen the data - BCQ variant:

```
Chapter 2. Background
```

vary completion position to test robustness of signal - ROUGE-L and exact match to quantify
similarity - Model-agnostic: works without access to training data

Why it matters for this thesis: - If models memorized GAIC sentences, RQ1 results are
inflated - Contamination can mask shortcuts: a model that "remembers" the answer doesn’t
need to reason - Temporal hypothesis: pre-2018 datasets more likely contaminated - My
mitigation: DCQ/BCQ applied to all 3 models across all 10 datasets

TODO: check Oren et al. 2024 for alternative contamination detection approaches TODO:
look into whether any GAIC datasets have known contamination issues

### 2.6. In-Context Learning

- In-context learning (ICL) is an inference paradigm in which a large language model
  adapts to a task specification provided in its input, without updating its parameters [brown2020language].
- Task adaptation occurs at inference time via the prompt, distinguishing ICL from fine-
  tuning, which optimizes parameters on a training distribution.
- ICL is studied in three regimes by demonstration count:
  - Zero-shot: task description only [wei2022finetuned, sanh2022multitask].
  - Few-shot: task description plus a small number of demonstrations [brown2020language].
  - Many-shot: hundreds to thousands of demonstrations, enabled by long-context
    models [agarwal2024manyshot].
- This thesis studies zero-shot ICL with varying descriptive context (Chapters 4–??)
  and extends to few-shot demonstrations as a continuation of the same paradigm (Sec-
  tion ??).
- Prior work has applied ICL to argument mining: cabessa2024llms show that GPT-
  with ICL matches fine-tuned encoder baselines, and sainz2024gollie demonstrate that
  including annotation guidelines as structured context in the prompt enables zero-shot

```
Chapter 2. Background
```

```
generalization across schemas.
```

- Known failure modes relevant to this work:
  - Label bias [zhao2021calibrate]: predictions are systematically biased by label-
    token frequencies in demonstrations.
  - Demonstration-selection shortcuts [liu2021makes, min2022rethinking]: similarity-
    based retrieval can reinforce surface features shared with the query, reintroducing
    at inference the kind of shortcut that feger2025limited document in fine-tuned
    encoders.

## Chapter 3.

## The GAIC Shared Task

This chapter describes the Generalizable Argument Identification in Context (GAIC) shared
task at Touché @ CLEF 2026, which serves as the experimental testbed for this thesis.
Having established in Chapter 2 that encoder-based argument mining systems learn dataset-
specific artifacts rather than transferable notions of argumentation, this chapter introduces
the shared task that was designed in direct response to those findings. The task definition is
presented in Section 3.1, followed by a description of the benchmark datasets in Section 3.2.
Section 3.3 analyzes the non-uniform availability of contextual metadata, and Section 3.
discusses the annotation scheme heterogeneity that makes context exploitation not merely
beneficial but necessary.

### 3.1. Task Definition

The GAIC shared task was organized at Touché @ CLEF 2026 by the same research group
that identified the generalization failures discussed in Chapter 2 [6]. It reformulates argument
identification as a context-grounded classification problem, departing from the established
paradigm in which models are trained on isolated sentences without access to the metadata
that human annotators rely on. The official task description reads:

```
“Given a sentence from a dataset along with metadata about its provenance, such
as the source text and the dataset’s annotation guidelines, predict whether the
sentence can be annotated as an argument or not. In particular, participants are
```

```
Chapter 3. The GAIC Shared Task
```

```
encouraged to develop robust systems that generalize beyond lexical shortcuts to
unseen datasets and investigate ways to exploit rich context information for this
purpose.”[15]
```

Formally, the input consists of a sentence accompanied by optional metadata: the dataset’s
source paper, annotation guidelines, and the source document from which the sentence was
drawn. The output is a binary label, Argument or No-Argument. Each input instance is
provided as a JSON object containing a unique identifier that encodes the source dataset and
split, the sentence text, and URLs pointing to the paper, guidelines, and source document
where available. A dash indicates that a particular metadata field is not provided for the
given dataset.

The task departs from traditional argument mining evaluation ([6],..) in one important as-
pect. It explicitly provides rich contextual metadata and encourages participants to exploit
it, thereby shifting the paradigm from training-based generalization, where task knowledge
must be encoded into model weights, to context-grounded classification, where task knowl-
edge is injected at inference time through the prompt.

Systems are evaluated using the macro F 1 score, computed for each test split and averaged
across all datasets, ensuring balanced treatment of both classes regardless of their frequency.
Each team may submit up to three systems via TIRA^1 , producing predictions in JSONL
format with sentence identifiers and predicted labels.

### 3.2. Datasets

GAIC unifies 10 established, publicly available argument mining datasets spanning diverse
domains, text genres, and argumentation frameworks. Each dataset contributes 1,700 labeled
sentences, partitioned into training (60%), development (20%), and test (20%) splits. All
sentences carry binary labels, Argument or No-Argument, derived from the original dataset
annotations through a mapping that collapses fine-grained argument component labels into a
single positive class. This mapping follows the same binarization scheme described in [6],
which identified these 10 datasets as the most relevant for studying argument identification.
Table 3.1 provides an overview.

(^1) https://www.tira.io/

```
Chapter 3. The GAIC Shared Task
```

Table 3.1.: Overview of the 10 GAIC benchmark datasets with their annotation schemes and
formal argument structure where applicable. A =⟨C, P⟩ denotes the canonical
argument kernel (Equation 2.1); datasets marked with A =⟨s⟩ use a flat single-
component scheme where no claim–premise distinction is annotated.
Dataset Domain Source Genre Argument Focus Formal Structure
ABSTRCT Biomedical RCT abstracts Claims vs. evidence in clinical trials A =⟨C, P⟩, k≥ 1
ACQUA Comparative Web comparisons Preference statements A =⟨s⟩, s expresses x≻ y
AEC Online debate CreateDebate Extractable argument facets A =⟨s⟩, s self-contained
AFS Debate forums ProCon, iDebate Self-contained debate positions A =⟨s⟩, s self-contained
ARGUMINSCI Scientific Research papers Own claims vs. background claims A =⟨C, P⟩, k≥ 1
FINARG Financial Earnings calls Claims with premise support A =⟨C, P⟩, k≥ 1
IAM Web Mixed sources Topic-dependent claims A =⟨C⟩, P implicit
PE Educational Persuasive essays Major claims, claims, premises A =⟨C, P⟩, k≥ 1, hierarchical
SCIARK Scientific Abstracts Claims and supporting evidence A =⟨C, P⟩, k≥ 1
USELEC Political Election debates Policy stances and judgments A =⟨C⟩, stance-grounded

The datasets span four broad domain clusters. The scientific cluster includes ABSTRCT
mayer2020abstrct, which consists of randomized controlled trial abstracts, ARGUMINSCI lauscher2018arguminsci,
which draws from full research papers, and SCIARK sciark2021, which covers scientific ab-
stracts from multiple disciplines. The political and debate cluster comprises AEC misra2015aec,
a collection of online debate forum posts from CreateDebate, AFS swanson2016afs, which
contains structured debate positions from ProCon and iDebate, and USELEC haddadan2019uselec,
which is based on transcripts of U.S. presidential and vice-presidential election debates. The
professional cluster includes FINARG alhamzeh2022finarg, which draws from earnings call
transcripts, and ACQUA toledo2019acqua, which consists of web-based comparative pref-
erence statements. The educational cluster contains PE stab2017pe, a dataset of persuasive
student essays, and IAM toledo2022iam, which aggregates mixed web content with topic-
dependent claims.

This diversity is deliberate and forces systems to generalize along multiple axes simultane-
ously: across domains (medical vs. financial vs. political), across registers (formal academic
writing vs. informal online debate), across text lengths (full documents vs. short forum posts),
and across argumentation models (claim-evidence vs. stance-based vs. hierarchical). The to-
tal corpus comprises approximately 17,000 sentences, balanced at roughly 50/50 between
Argument and No-Argument within each dataset.

```
Chapter 3. The GAIC Shared Task
```

### 3.3. Context Availability

A distinctive feature of GAIC is the provision of contextual metadata alongside each sen-
tence. However, not all context types are available for all datasets. This non-uniform avail-
ability, summarized in Table 3.2, creates a natural experimental condition for studying which
types of context contribute most to generalization.

Table 3.2.: Context availability across the 10 GAIC datasets. A checkmark (✓) indicates that
the context type is provided; a dash (–) indicates it is unavailable.
Dataset Definition Guidelines Document Context
ABSTRCT✓✓✓
ARGUMINSCI✓✓✓
PE✓✓✓
USELEC✓✓✓
FINARG✓ – ✓
SCIARK✓ – ✓
ACQUA✓ – –
AEC✓ – –
AFS✓ – –
IAM✓ – –

Three types of context are distributed across the datasets:

Argument definitions are available for all 10 datasets. Each definition describes the theoret-
ical framework for what constitutes an argument in the respective dataset, extracted from the
original dataset paper. These specify the conceptual criteria for classification, for instance,
whether claims and evidence are both considered argumentative or only claims.

Annotation guidelines are available for 4 of the 10 datasets (ABSTRCT, ARGUMINSCI,
PE, and USELEC). Guidelines provide the operational decision rules that human annotators
followed during labeling, including boundary cases, signal phrases, and concrete examples.
Where available, they resolve ambiguities left open by the theoretical definition. This distinc-
tion between definitions and guidelines mirrors the broader observation in AM that the theo-
retical framework alone is often insufficient for consistent annotation stab2018argumentation,
feger2025limited.

Document context is available for 6 of the 10 datasets (ABSTRCT, ARGUMINSCI, PE,

```
Chapter 3. The GAIC Shared Task
```

USELEC, FINARG, and SCIARK). This consists of the source document from which each
sentence was drawn, enabling the resolution of anaphoric references and understanding of
local argumentative flow.

The practical consequence of this non-uniform availability is that the full context ladder—
from generic classification to definition-augmented, to guideline-augmented, to document-
augmented—can only be tested on the four datasets with complete context (ABSTRCT, AR-
GUMINSCI, PE, USELEC). The remaining datasets serve as controls for assessing the rel-
ative contribution of each context type. This design is further exploited in the methodology
described in Chapter 4.

### 3.4. Annotation Scheme Heterogeneity

A defining characteristic of the GAIC benchmark, and one that sets it apart from prior evalu-
ation setups, is that its constituent datasets operationalize the concept of “argument” in fun-
damentally different ways. The same sentence could receive different labels depending on
which annotation scheme is applied. This heterogeneity is not a defect of the benchmark but
a deliberate design choice: it is precisely what makes context exploitation necessary rather
than optional.

As discussed in Section ?? and extensively documented by [6], the lack of a universally ac-
cepted definition of “argument” is a persistent challenge in the field. The 10 GAIC datasets
reflect this challenge concretely. They can be grouped into four families of annotation mod-
els, each reflecting a distinct theoretical commitment about what constitutes an argument:

Claim-evidence models (ABSTRCT, SCIARK, IAM) distinguish between concluding state-
ments and supporting observations. Under these schemes, a statistical finding such as “The
mortality rate was 23%” is typically classified as evidence—a factual observation support-
ing a claim—rather than as an argument in its own right. In ABSTRCT specifically, claims
are conclusions about treatment outcomes, while evidence consists of the measurements and
observations on which those conclusions rest.

Hierarchical models (PE, ARGUMINSCI) define nested argumentative structures in which
major claims are supported by claims, which are in turn supported by premises. All com-

```
Chapter 3. The GAIC Shared Task
```

ponents that participate in this hierarchical structure are labeled as arguments. Under such a
scheme, the same statistical finding that would be classified as non-argumentative evidence
in a claim-evidence model may be labeled Argument if it serves as a premise supporting a
higher-level claim.

Comparative and stance models (ACQUA, USELEC) require explicit expressions of pref-
erence or policy position. ACQUA demands comparative judgment, that is, statements of the
form X is better than Y. USELEC requires policy stances or evaluative judgments that call
for justification. Mere descriptions or factual comparisons without expressed preference are
non-argumentative under both schemes.

Extractability models (AEC, AFS) focus on whether a sentence can stand alone as an in-
terpretable argumentative unit without requiring surrounding context. A sentence that makes
a clear, self-contained argumentative point qualifies as an argument; one that depends on its
context for interpretability does not.

This heterogeneity has a direct consequence for system design: no universal set of lexical or
structural cues can identify arguments reliably across all 10 datasets. A system must adapt
its classification criteria to each dataset’s notion of “argument.” The provided context—
definitions and, where available, guidelines—is designed to make this adaptation possible.
Without context, a model can only rely on generic features, which, as established in Chap-
ter 2, tend to be dominated by domain-specific content words rather than structural patterns
of argumentation.

The GAIC task thus creates a controlled setting for the central hypothesis of this thesis: that
context-grounded classification, where the model receives an explicit specification of what
counts as an argument, can overcome the generalization failures of context-free approaches.
The methodology for testing this hypothesis is presented in the following chapter.

## Chapter 4.

## Methodology

This chapter presents the experimental methodology for investigating whether zero-shot large
language models can solve the GAIC task through context exploitation and whether they ex-
hibit fundamentally different sensitivity to argumentative structure than the trained encoders
examined by [6]. The chapter is organized as follows. Section 4.1 describes the models un-
der evaluation. Section 4.2 details the context extraction pipeline and prompt design that set
up the GAIC metadata for zero-shot classification. Section 4.3 introduces the text manipu-
lation experiments that probe structural sensitivity. Section 4.4 presents the contamination
detection protocol that serves as a validity check for the main findings. Finally, Section 4.5
specifies the data splits, sample sizes, evaluation metric, and reproducibility measures that
govern all experiments.

### 4.1. Models

We evaluate three decoder-based large language models spanning different scales and providers,
summarized in Table 4.1. The selection is guided by two considerations: covering a range
of model capacities to test whether structural sensitivity and context exploitation vary with
scale. We started with more than these three model but due to high cost and redundancy
destilled the models to these three.

All three models are instruction-tuned variants, chosen deliberately to ensure a clean com-
parison with the encoder experiments of [6].

```
Chapter 4. Methodology
```

Table 4.1.: Models evaluated in this thesis. All are instruction-tuned, non-reasoning variants.

```
Model Parameters Provider Role
Ministral 8B 8B Mistral Small-scale, low contamination baseline
Mistral Medium ∼70B? Mistral Large-scale Mistral
GPT-5.2 Frontier OpenAI Frontier reasoning model
```

The inclusion of multiple model scales serves a dual purpose. First, it enables analysis of
whether structural sensitivity is a general property of decoder architectures or emerges only
at sufficient capacity. Second, it provides natural variation in contamination levels, since
larger models trained on more data are expected to have encountered more benchmark sen-
tences during pre-training. This variation is exploited in the contamination analysis of RQ3
(Section 4.4).

### 4.2. Context Extraction and Prompt Design

#### 4.2.1. Context Extraction Pipeline

As described in Chapter 3, GAIC provides URLs pointing to dataset papers, annotation
guidelines, and source documents. These raw sources must be transformed into structured,
LLM-consumable context before they can be injected into classification prompts. To this
end, we developed a three-stage extraction pipeline.

```
Chapter 4. Methodology
```

```
Figure 4.1.: The three-stage context extraction pipeline.
```

In the first stage, we extract the metadata provided by GAIC. For each dataset, we have a
corresponding paper, and for 4 out of 10 Datasets we have the Annotation Guidelines and
Document context. For the document context, we extract the two sentences immediately pre-
ceding each target sentence. Sentence boundaries are detected using a regex-based splitter
that segments on sentence-terminal punctuation ([.!?]) followed by whitespace and an up-
percase letter. The choice of two preceding sentences balances providing sufficient discourse
context for resolving anaphora and understanding local argumentative flow against the risk
of introducing noise from distant text.

In the second stage, raw PDF files, the dataset papers, and annotation guidelines are con-
verted to Markdown using Kreuzberg^1 , an open-source PDF parser that handles multi-column
layouts, embedded figures, and complex table structures typical of academic publications.
This stage produces unstructured text that preserves content while discarding layout infor-
mation.

In the third stage, the extracted text is processed by GPT-5.2 with Pydantic schemas that
enforce structured output. Two extraction tasks are performed per dataset. For argument
definitions, the schema requires 3 to 10 sentences that specify what constitutes an argument

(^1) https://github.com/DS4SD/kreuzberg

```
Chapter 4. Methodology
```

and what does not, consistent with the specific dataset’s theoretical framework. For annota-
tion guidelines, the schema requires 3 to 10 sentences summarizing the operational decision
rules for determining whether a sentence is an argument. In both cases, the system prompt
instructs the model to extract only dataset-specific criteria, excluding generic argumentation
theory, dataset statistics, or methodological details unrelated to the concept of argumentative-
ness. The use of schema enforcement ensures that extracted context is consistently structured
across all 10 datasets, facilitating systematic insertion into classification prompts.

#### 4.2.2. The Context Ladder

The extracted context enables a cumulative ablation study, referred to as the context ladder, in
which each level includes all content from the previous levels, thereby isolating the marginal
contribution of each context type. Table 4.2 specifies the four conditions.

Table 4.2.: Context ladder conditions. Each level is cumulative: C3 includes all content from
C0 through C2.
Level Content in Prompt Purpose
C0 Generic instruction only Baseline LLM knowledge of “argument”
C1 + Dataset-specific definition Does aligning to the dataset’s theory help?
C2 + Annotation guidelines Do operational rules add value beyond the-
ory?
C3 + 2 preceding sentences Does local discourse context help disam-
biguation?

At the C0 level, the model receives only a minimal, dataset-agnostic instruction: it is told to
classify whether a sentence is an argument, relying solely on its pre-trained understanding
of the concept. This condition measures baseline capacity for argument identification with-
out any task-specific guidance. At C1, the dataset-specific definition extracted in Stage 3 is
added, aligning the model’s classification criteria with the theoretical framework of the tar-
get dataset. For example, when classifying sentences from ABSTRCT, the model receives
a definition that distinguishes claims from evidence in the context of clinical trials. At C2,
annotation guidelines are added on top of the definition, providing operational decision rules
that specify how annotators resolved boundary cases, which linguistic signals indicated ar-
gumentativeness, and what exceptions applied. At C3, the two sentences preceding the target
sentence in the source document are appended, providing local discourse context.

```
Chapter 4. Methodology
```

As noted in Section 3.3, C2 and C3 are only testable on datasets where the required con-
text is available: C2 on the four datasets with guidelines (ABSTRCT, ARGUMINSCI, PE,
USELEC), and the full ladder on those same four datasets, which also provide document
context.

#### 4.2.3. Prompt Structure

All experiments use a consistent two-message format to minimize confounds from prompt
engineering variation. The system message contains the assembled context and classification
instruction; the user message contains the target sentence:

System Message

```
You are an expert in argumentation analysis.
C1+ ## Argument Definition
[dataset-specific definition]
C2+ ## Annotation Guideline
[operational rules; overrides definition]
C3 ## Document Context
[preceding sentences, per-sample]
## Task
Classify whether the following sentence is an argument
based on the criteria above.
```

User Message

```
{sentence}
```

Figure 4.2.: Prompt structure by context level. C0 omits all colored blocks and has a simpli-
fied task "Classify whether the sentence is an argument or not."; each subsequent
level adds context cumulatively.

Context is assembled in a fixed order. Definition, then guidelines, then document context,
with explicit section headers separating each component. When both a definition and guide-
lines are present, a note instructs that guidelines override definitions where they conflict,
reflecting the principle that operational annotation rules take precedence over theoretical
frameworks in cases of ambiguity. Output validation is enforced via Structured Output^2
through Pydantic schemas that accept only the two valid labels.

(^2) https://developers.openai.com/api/docs/guides/structured-outputs

```
Chapter 4. Methodology
```

### 4.3. Text Manipulation Experiments (RQ2)

To probe whether LLMs rely on argumentative structure or surface-level patterns, we apply
two systematic text manipulations following and extending the methodology of [6]. The
manipulations are summarized in Table 4.3 and illustrated with an example sentence.

Table 4.3.: Text manipulation conditions. M0 serves as the baseline against which deltas are
computed.
Condition Transformation What It Tests
M0: Original Unmodified sentence Baseline performance
M1: Content-only Remove stopwords, function words, punctuation Reliance on discourse markers and syntax
M2: Shuffle Random word permutation (seed = 42) Reliance on word order

#### 4.3.1. M1: Content-Only Reduction

The content-only manipulation removes all linguistic elements that serve structural or func-
tional roles in argumentation while retaining semantic content words. Implementation uses
the spaCy^3 library with the en_core_web_sm model for part-of-speech tagging. The re-
moved elements include stopwords from spaCy’s default list (which covers many common
discourse markers such as therefore, however, and because), all punctuation tokens, and to-
kens tagged with function-word part-of-speech categories: adpositions (ADP), auxiliaries
(AUX), coordinating and subordinating conjunctions (CCONJ, SCONJ), determiners (DET),
particles (PART), pronouns (PRON), and interjections (INTJ). The retained elements are con-
tent words, including nouns, verbs, adjectives, adverbs, proper nouns, and numerals, lower-
cased and joined by single spaces.

This manipulation directly replicates the content-word reduction used by [6], in which the
elimination of approximately every second word from a sentence, removing the structural
scaffolding while preserving the semantic content, produced negligible performance change
(∆≤ 0 .02) across all four encoder models tested. If a decoder model similarly maintains
its performance under M1, this would indicate that it, too, classifies based on content words
alone, the same shortcut behavior identified in encoders.

To illustrate, consider the sentence “Therefore, human cloning should be prohibited.” Under

(^3) https://spacy.io/

```
Chapter 4. Methodology
```

M1, this reduces to “human cloning prohibited”, stripping the discourse marker therefore,
the auxiliary should be, and the punctuation.

#### 4.3.2. M2: Word Shuffle

The shuffle manipulation randomly permutes all words in the sentence using a fixed random
seed (42) for reproducibility. Terminal punctuation is preserved but repositioned according to
the permutation. This disrupts syntactic structure, phrasal boundaries, grammatical relations,
and the sequential ordering patterns that discourse markers create.

This manipulation is new relative to [6] and specifically targets the architectural difference
between encoders and decoders. Encoders use bidirectional self-attention, producing repre-
sentations that Sinha2020-yh showed to be largely order-invariant, shuffled word order barely
degrades encoder performance on natural language inference. Decoders, by contrast, use
causal attention: each token’s representation depends only on preceding tokens. Word order
is therefore a hard architectural constraint, not merely a learned pattern. A discourse marker
such as therefore at position k shapes the computation of all tokens at positions k+ 1 , k+ 2 ,...
through the causal attention mask. Disrupting word order should therefore produce sub-
stantially larger performance degradation in decoders than the near-zero deltas observed in
encoders.

Returning to the example, the shuffled form becomes “prohibited should human Therefore
cloning be.”, which destroys the argumentative structure while preserving the vocabulary.

#### 4.3.3. Delta Metrics

For each manipulation, the performance delta relative to the original is computed as:

∆content= F (^1) M0− F (^1) M1 (4.1)
∆shuffle= F (^1) M0− F (^1) M2 (4.2)
Positive deltas indicate that the model relied on the removed or disrupted features; larger

```
Chapter 4. Methodology
```

deltas imply greater sensitivity to argumentative structure. The comparison baseline is the
encoder result reported by Feger2025-kz:∆≤ 0 .02 across all manipulation conditions for
BERT, RoBERTa, and DistilBERT, with only WRAP showing a slightly larger drop of∆ =
0 .05 due to its task-specific pre-training. We hypothesize that decoder LLMs will show sub-
stantially larger deltas, indicating that they process argumentative structure during generation
rather than relying on content-word shortcuts.

#### 4.3.4. Context–Manipulation Interaction

An important secondary analysis examines whether context level modulates manipulation
sensitivity. Both manipulations (M1, M2) are applied at each context level (C0 through
C3, where available). Two competing outcomes are possible. If context enables genuinely
structure-aware classification, then manipulations that destroy structure should cause larger
performance drops when context is present, context increases sensitivity. Alternatively, if
context itself provides sufficient cues for classification regardless of the sentence’s structural
integrity, manipulations should have smaller effects in the presence of context, context buffers
against manipulation. Comparing deltas across context levels distinguishes between these
hypotheses.

### 4.4. Contamination Detection (RQ3)

#### 4.4.1. Motivation

The 10 GAIC benchmark datasets were published between 2014 and 2022. The pre-training
corpora of the evaluated models, which draw from Common Crawl, academic papers, and
publicly available datasets, likely include some or all of these sentences. If models achieve
high Macro F 1 scores through memorization rather than genuine classification, the perfor-
mance results of RQ1 are inflated. If contamination also masks manipulation effects, because
the model retrieves a memorized label regardless of input perturbation, the structural sensi-
tivity findings of RQ2 are confounded. Contamination detection therefore serves not as an
end in itself but as a validity check for the main experimental findings.

```
Chapter 4. Methodology
```

#### 4.4.2. The Data Contamination Quiz Method

We adopt the Data Contamination Quiz (DCQ) methodology from [16], which detects con-
tamination while controlling for position biases inherent in multiple-choice evaluation. The
method proceeds in three phases.

Phase 1: Perturbation Generation. For each dataset, 50 sentences are sampled (25 per
class, balanced between Argument and No-Argument). For each sampled sentence, GPT-4.1
generates four synonym-based perturbations at temperature 1.0. Constraints are enforced via
a Pydantic schema: all four perturbations must differ from the original and from each other,
only word-level substitutions are permitted, and overall meaning and syntactic structure must
be preserved. The result is a set of sentences that are semantically equivalent to the original
but lexically distinct. A model can distinguish the original from its perturbations only if it
has memorized the exact wording.

```
Figure 4.3.: Pydantic schema for Structured output in Phase 1
```

Phase 2: Bias Detector Quiz (BDQ). Large language models often exhibit position biases
in multiple-choice settings, preferring certain answer positions regardless of content. The
BDQ detects these biases before measuring contamination, preventing false positives. A five-
choice quiz is presented using only the four perturbations; the original is absent, with option
E as “None of the provided options.” Since the correct answer is not among the choices, any
systematic preference for specific positions reflects bias rather than recognition. Positions
selected at rates below random chance (20% for five options) are marked as non-preferred.

```
Chapter 4. Methodology
```

Phase 3: Bias Compensator Quiz (BCQ). The actual contamination measurement places
the original sentence at non-preferred positions, positions where the model is least likely to
select an answer by chance, alongside three perturbations and option E. If the model still
selects the original at a non-preferred position, this constitutes stronger evidence of genuine
recognition. The raw recognition rate is adjusted using Cohen’s kappa:

```
κ =Pobserved− Pexpected
1 − Pexpected
```

##### (4.3)

where Pexpectedis the bias rate from Phase 2. This adjustment discounts recognition that
could be attributed to residual position preference. The output is a contamination range
[min%, max%] per model per dataset, accounting for measurement uncertainty across posi-
tions.

#### 4.4.3. Contamination Thresholds and Interpretation

Contamination levels are categorized as low (0–20%), moderate (20–40%), or high (above
40%). These thresholds determine how results are interpreted rather than whether they are
reported:

For RQ1, high contamination on a given dataset means that improved F 1 with context may
partly reflect better retrieval of memorized labels rather than genuine context exploitation.
Such results are flagged, and context effects are examined separately on low-contamination
datasets to test whether the pattern persists.

For RQ2, the critical test is whether manipulation sensitivity persists on contaminated data. If
a model has memorized sentences but manipulation still causes large∆, the model is not per-
forming verbatim retrieval but still processing structure during generation. Persistent deltas
on contaminated data would strengthen rather than weaken the structural sensitivity finding.
This interpretation transforms contamination from a pure validity threat into an additional
source of evidence.

```
Chapter 4. Methodology
```

### 4.5. Experimental Design

#### 4.5.1. Data Splits

All context ladder and manipulation experiments (RQ1, RQ2) are conducted on the GAIC
development split, which contains approximately 340 sentences per dataset (170 per class).
The test split is reserved for the final evaluation and the GAIC shared task submission. The
training split is not used, as all experiments are zero-shot.

#### 4.5.2. Sample Sizes

For the context ladder and manipulation experiments, 60 sentences per dataset are used (30
per class), sampled deterministically by taking the first k balanced sentences from the devel-
opment split. This sample size was chosen to keep the inference runs as sparse as possible
while maintaining enough statistical power to produce meaningful results. To be precise, a
single full experimental pass requires 60× 10 × 3 × 3 × 4 = 21 ,600 inference calls across
samples, datasets, manipulation conditions (M0–M2), models, and context levels (C0–C3).
Using the full development split of 340 sentences per dataset would increase this figure by
roughly sixfold, at considerable API cost, without proportional gain.

Deterministic sampling avoids dependence on random seed behavior and ensures exact re-
producibility. For contamination testing, 50 sentences per dataset (25 per class) are used
[16].

#### 4.5.3. Evaluation Metric

All experiments report the macro F 1 score, computed as the unweighted mean of the F 1 scores
for the Argument and No-Argument classes. This metric treats both classes equally regardless
of their frequency, aligns with the official GAIC evaluation metric, and ensures that models
cannot achieve inflated scores by trivially predicting the majority class.

```
Chapter 4. Methodology
```

#### 4.5.4. Reproducibility

Several measures ensure full reproducibility. A fixed random seed of 42 governs all stochastic
operations, including word shuffling (M2) and any sampling procedures. All LLM inferences
are performed at temperature 0 to produce deterministic outputs. Structured outputs are
enforced through Pydantic schemas that constrain model responses to valid labels.

#### 4.5.5. Comparison Baselines

Results are interpreted relative to two baselines established by [6]:

The performance baseline consists of encoder models (BERT, RoBERTa, DistilBERT, WRAP)
trained on jointly combined benchmark data and evaluated on held-out datasets. These
achieve macro F 1 scores between 0.63 and 0.74, depending on the model and target dataset. In
contrast, single-dataset SOTA encoders reach higher in-distribution scores (mean F 1 = 0 .79)
but substantially lower cross-dataset transfer scores (mean F 1 = 0 .56–0.61).

The manipulation baseline consists of the encoder deltas under the content-only manipu-
lation:∆≤ 0 .02 for BERT, RoBERTa, and DistilBERT, and∆ = 0 .05 for WRAP. These
near-zero values indicate that the standard encoder models are largely insensitive to the re-
moval of discourse markers, function words, and syntactic structure, they classify based on
content words alone.

### 4.6. Summary

Table 4.4 provides a consolidated overview of the experimental methodology across the three
research questions.

```
Table 4.4.: Summary of the experimental methodology by research question.
RQ Method Key Metric Comparison Baseline
RQ1 Context ladder (C0→C3) Macro F 1 Feger et al. encoders (F 1 = 0 .63–0.74)
```

RQ2 Text manipulation (M0/M1/M2)∆ = F (^1) M0− F (^1) Mx Feger et al. encoder∆≤ 0. 02
RQ3 DCQ three-phase protocol Contamination [min%, max%] Validity threshold: 40%

```
Chapter 4. Methodology
```

The combination of performance evaluation (RQ1), structural sensitivity probing (RQ2), and
contamination control (RQ3) enables a coherent analysis. Specifically, it allows us to deter-
mine whether the rich context provided by GAIC enables generalization that training alone
cannot achieve, whether decoder LLMs process the structural properties of arguments that
encoders provably ignore, and whether these findings hold up under the scrutiny of contami-
nation analysis. The results of these experiments are presented in the following chapter.

## Chapter 5.

## Results

### 5.1. RQ1: Can Zero-shot Decoders match trained Encoders?

The short answer is yes, on most datasets, and context is the key lever.

Looking at Table ??, the single biggest performance driver is the step from C0 to C1, so
adding a precise task definition. This alone gives us between + 0 .06 and + 0 .11 average
macro-F1 across models, and no other context step produces a comparable jump.

The two top performers are ABSTRCT and ARGUMINSCI, where all models reach their
highest scores. What I find remarkable is that even Ministral 8B, our smallest model, per-
forms best on these benchmarks and even surpasses GPT-5.2 on ARGUMINSCI, although
it is roughly a factor of 100 smaller. So whatever signal these datasets carry seems to be
accessible without huge scale, as long as the task is specified properly.

AEC is the most interesting case. In the first version of our C1 prompt we got near-random
performance, basically the same as C0. After checking the benchmark again and integrating
the most important aspect of its design by hand, performance jumped from random to above
0.9 F1, which is almost SOTA. This shows that my approach is most effective when the task
description is not just automatically extracted but handcrafted, especially for datasets whose
annotation logic is not fully captured by the paper alone.

```
Chapter 5. Results
```

The worst performance is on FINARG and PE. For FINARG we can explain it with (cite

... ), namely that there is a misalignment between the argument definition in the paper and
the actual annotation guidelines, so even a faithful extraction of the stated definition does not
match what the labels reflect.

Figure 5.2 compares our best zero-shot scores to the encoder SOTA from Marc Feger’s paper.
What is quite impressive is that our naive decoder approach, without heavily optimizing the
prompts, already surpasses the trained encoders on half of the datasets. This answers our
research question with a clear yes: a model that was not trained on the dataset, given only an
instruction extracted from the paper or annotation guideline, can beat a supervised encoder.

Figure 5.1 shows the effect of the context ladder on the four datasets where the full ladder
is available. The interesting part is that for almost all models we see a clear ladder from
C0 to C2 (e.g. ABSTRCT, USELEC), and with C3 the performance stays the same or drops
only marginally. PE behaves the opposite way: we see an inverted ladder where every added
context level decreases F1, going from medium-good performance across all three models
down to pure random on the last one. So there is a hint that in our benchmarks extra context
usually helps, but sometimes it hurts, and what this means in detail needs to be investigated
per dataset. It also underlines that there is no one-fits-all solution.

```
Table 5.1.: Macro-F1 scores by model, context level, and dataset. Best per column in bold.
Model Ctx ABSTRCT ACQUA AEC AFS ARGUMINSCI FINARG IAM PE SCIARK USELEC Avg
gpt_5_2_openai c0 0.692 0.618 0.562 0.569 0.524 0.405 0.563 0.666 0.601 0.472 0.567
c1 0.663 0.749 0.967 0.767 0.917 0.438 0.764 0.600 0.717 0.630 0.721
c2 0.883 – – – 0.883 – – 0.583 – 0.649 0.750
c3 0.917 – – – 0.847 0.540 – 0.524 0.748 0.738 0.757
ministral_8b c0 0.738 0.748 0.451 0.514 0.496 0.426 0.764 0.676 0.729 0.520 0.606
c1 0.833 0.744 0.950 0.699 0.733 0.542 0.681 0.697 0.633 0.580 0.709
c2 0.950 – – – 0.668 – – 0.603 – 0.679 0.725
c3 0.900 – – – 0.733 0.563 – 0.474 0.619 0.619 0.682
mistral_medium c0 0.764 0.732 0.444 0.451 0.576 0.365 0.661 0.641 0.867 0.562 0.606
c1 0.900 0.764 0.917 0.766 0.757 0.583 0.782 0.603 0.783 0.614 0.747
c2 0.917 – – – 0.753 – – 0.569 – 0.710 0.737
c3 0.933 – – – 0.753 0.611 – 0.547 0.762 0.698 0.733
```

```
Chapter 5. Results
```

```
Figure 5.1.: TBD
```

```
USELEC
ARGUMINSCI ABSTRCT
```

```
SCIARK IAM AEC AFS ACQUA FINARG PE
```

```
0.0
```

```
0.2
```

```
0.4
```

```
0.6
```

```
0.8
```

```
1.0
```

```
Macro-F1
```

```
0.69 0.69
```

```
0.87
0.71
0.60 0.60
```

```
0.76
0.66 0.61
0.57
```

```
0.66
```

```
0.84 0.89 0.83
0.76
```

```
0.96
0.84 0.84
0.69
0.74 0.78
```

```
0.92 0.95 0.87
0.78
```

```
0.97
```

```
0.77 0.76
0.61
```

```
0.70
```

```
Zero-Shot LLMs vs Supervised Encoders vs k-NN Baseline
k-NN Baseline (k=50)Encoder SOTA (Feger et al.)
LLM Best (Zero-Shot)
```

```
Figure 5.2.: TBD
```

### 5.2. RQ2: Are Decoders More Sensitive to Argumentative Structure?

In this section we investigate how robust the models are to the actual content within the sen-
tences via the content manipulation introduced by [6], and additionally whether the models

```
Chapter 5. Results
```

really look at the syntactic structure of the sentences or treat the text like a bag of words, as
encoders do. For the latter we use the shuffle experiment.

The first observation is that our drops under content manipulation are much larger than what
[6] reports for encoders. This is not surprising, since our model never had the chance to learn
dataset-specific keywords as shortcuts in the first place. Still, some datasets seem easy to
identify even without the content words. For example, GPT-5.2 only loses 0.02 compared to
the unmanipulated case, which could partly be due to memorization, or simply because the
sentences without their argumentation syntax can still be identified as arguments. The largest
drop for GPT-5.2 is on ARGUMINSCI, with almost 0.46, going from 0.9 down to random
performance. This means that ARGUMINSCI relies heavily on argument syntax, and much
less on content words from the dataset.

The second manipulation is the shuffle experiment, where we randomly permute the words
in the sentence and check whether the model still classifies it as an argument. Here we see
a clear benefit of using decoders. Because of their robustness to manipulated context, these
models do not treat the text as a bag of words anymore. They go through it autoregressively
and can therefore make more robust inferences instead of relying on a few surface indicators.
All models drop on average between 0.20 and 0.29 F1, and surprisingly Ministral is the
most robust to this kind of manipulation. A possible explanation is that because Ministral
is smaller, it has less world knowledge and follows the task more closely as intended. The
worst robustness is GPT-5.2, possibly because it is, in a sense, too smart and tries to infer
that there is an argument anyway, where in our setting this is exactly the trap.

#### 5.2.1. Does Context increase Manipulation Sensitivity?

This question came up automatically during the experiments. Here we look at the four
datasets that have the full context ladder and ask whether more context increases robust-
ness, in the sense that the model knows better what to look for and stops trying to outsmart
the prompt.

What we see is a clear jump from C0 to C1. So if we do not specify the task, the model
tries to solve it in many different ways, but with a more detailed task description (in our
case the definition of what counts as an argument), robustness almost doubles for content
manipulation, and we gain about + 0 .1 on the shuffle experiment. From C2 to C3 the returns

```
Chapter 5. Results
```

Table 5.2.: Performance degradation after text manipulation (C1 context). Format:∆content/
∆shu f f le. Bold indicates largest drop per manipulation type.
Dataset GPT-5.2 Ministral 8B Mistral Medium
ABSTRCT -0.02 / -0.16 -0.25 / -0.45 -0.18 / -0.35
ACQUA -0.14 / -0.14 -0.23 / -0.22 -0.16 / -0.11
AEC -0.33 / -0.19 -0.32 / -0.10 -0.28 / -0.07
AFS -0.07 / -0.18 -0.11 / -0.37 -0.04 / -0.36
ARGUMINSCI -0.46 / -0.48 -0.31 / -0.40 -0.32 / -0.39
FINARG -0.08 / -0.10 -0.21 / -0.21 -0.25 / -0.25
IAM -0.12 / -0.23 -0.02 / -0.28 -0.10 / -0.27
PE -0.08 / -0.07 -0.26 / -0.36 -0.01 / -0.19
SCIARK -0.21 / -0.30 -0.09 / -0.30 -0.20 / -0.26
USELEC -0.08 / -0.23 -0.09 / -0.17 -0.12 / -0.14
Mean -0.16 / -0.21 -0.19 / -0.29 -0.17 / -0.24

are marginal, which suggests there is a limit to how much context helps for robustness. The
next step would be to try out dedicated prompt-improvement methods that explicitly prevent
this kind of behaviour (cite!!), but for the scope of this thesis we leave this out.

```
Figure 5.3.: TBD
```

```
Chapter 5. Results
```

### 5.3. RQ3: Does Contamination Confound Performance

### and Manipulation Measurements?

#### 5.3.1. Contamination Levels by Model and Dataset

```
ABSTRCT ACQUA AEC
AFS
ARGUMINSCI FINARG
```

```
IAM PE SCIARK USELEC
Dataset
```

```
0
```

```
10
```

```
20
```

```
30
```

```
40
```

```
50
```

```
60
```

```
70
```

```
80
```

```
Contamination Range (%)
```

```
DCQ Contamination Levels by Model and Dataset(bars show range from BCQ positions)
Mistral MediumGPT-5.2
Ministral 8B
```

Figure 5.4.: DCQ contamination levels across datasets and models, showing systematic vari-
ation by model and dataset. Error bars indicate the range across BCQ positions.

This is a tricky question, so first of all, what can we see? GPT-5.2 and Mistral Medium have
the highest contamination percentages, sometimes more on GPT, sometimes more on Mistral.
This can mainly be explained by the size of the models. With an estimated 1.3T parameters
for GPT-5.2 and roughly 200B for Mistral Medium, they have much more memorization
capacity than the much smaller Ministral 8B, which might have forgotten these datasets even
if it saw them during pre-training. As a result, Ministral stays below 30% contamination,
which is clearly lower than the other two models.

If we look at our best-performing dataset, ABSTRCT, one could argue that Mistral Medium
and GPT-5.2 have seen the data and that this contributed to their high scores. But Ministral
8B shows only around 28% contamination on the same dataset and still reaches almost SOTA.
So contamination might be a factor, but on this dataset there are clearly other things at play.

For ACQUA and SCIARK we see that Mistral has higher contamination and also scored
higher on the experiments, which is at least consistent with the contamination story, even if

```
Chapter 5. Results
```

we cannot conclude causality from it.

Further work needs to be done to investigate benchmark contamination, with the BCQ method
we only see if the models know the datasets and have seen the sentences but not if they have
seen the actual label etc. So They are familiar with the domain and this might help them in
the inference.

Table 5.3.: Comparison of argument identification approaches on the GAIC benchmark.
SOTA represents in-distribution supervised performance (upper bound). Encoder
shows the best cross-dataset transfer result from Feger2025-kz. KNN uses k-
nearest neighbor retrieval (k = 50). LLM is our zero-shot decoder-based approach
evaluated on the dev set. Best zero-shot method per row in bold.
Dataset SOTA Encoder KNN LLM
(upper bound) (cross-dataset) (k=50) (zero-shot)
ABSTRCT 0.89 0.74 0.87 0.89
ACQUA 0.84 0.66 0.66 0.82
AEC 0.96 0.57 0.60 0.93
AFS 0.84 0.60 0.76 0.71
ARGUMINSCI 0.84 0.59 0.69 0.77
FINARG 0.68 0.66 0.61 0.55
IAM 0.76 0.73 0.60 0.71
PE 0.78 0.69 0.57 0.55
SCIARK 0.83 0.75 0.71 0.70
USELEC 0.74 0.70 0.70 0.58
Mean 0.82 0.65 0.68 0.73

## Chapter 6.

## Discussion

### 6.1. Integration of Results

### 6.2. Comparison to Encoder Baselines

### 6.3. The Role of Context

### 6.4. Implications for Argument Mining

### 6.5. Limitations

## Chapter 7.

## Conclusion

### 7.1. Summary

### 7.2. Contributions

### 7.3. Future Work

## Appendix A.

## Weitere Abbildungen

Falls kein Anhang benötigt wird, das Einbinden in der master-Datei auskommentieren.

### A.1. Datensatzeigenschaften

Die folgenden Abbildungen zeigen etwas Wichtiges.

```
Figure A.1.: Nochmal das Logo
```

## Bibliography

```
[1] John Lawrence and Chris Reed. “Argument mining: A survey”. en. In: Comput. Lin-
guist. Assoc. Comput. Linguist. 45.4 (Jan. 2020), pp. 765–818.
[2] Ehud Aharoni et al. “A Benchmark Dataset for Automatic Detection of Claims and Ev-
idence in the Context of Controversial Topics”. In: Proceedings of the First Workshop
on Argumentation Mining. Ed. by Nancy Green et al. Baltimore, Maryland: Associa-
tion for Computational Linguistics, June 2014, pp. 64–68. DOI: 10.3115/v1/W14-
2109. URL: https://aclanthology.org/W14-2109/.
[3] Christian Stab et al. “Cross-topic Argument Mining from Heterogeneous Sources”.
In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing. Ed. by Ellen Riloff et al. Brussels, Belgium: Association for Computa-
tional Linguistics, 2018, pp. 3664–3674. DOI: 10.18653/v1/D18-1402. URL:
https://aclanthology.org/D18-1402/.
[4] Aris Fergadis et al. “Argumentation mining in scientific literature for sustainable de-
velopment”. In: Proceedings of the 8th Workshop on Argument Mining. 2021, pp. 100–
111.
[5] James B Freeman. Dialectics and the macrostructure of arguments: A theory of argu-
ment structure. Vol. 10. Walter de Gruyter, 2011.
[6] Marc Feger, Katarina Boland, and Stefan Dietze. “Limited generalizability in argu-
ment mining: State-of-the-art models learn datasets, not arguments”. In: arXiv [cs.CL]
(May 2025).
[7] Lida Shi et al. “From text mining to intelligent debate: Task frameworks and techno-
logical evolution in computational argumentation”. en. In: Inf. Process. Manag. 63.2
(Mar. 2026), p. 104465.
```

```
Bibliography
```

```
[8] Marie-Francine Moens et al. “Automatic detection of arguments in legal texts”. In:
Proceedings of the 11th international conference on Artificial intelligence and law.
2007, pp. 225–230.
[9] Raquel Mochales Palau and Marie-Francine Moens. “Argumentation mining: the de-
tection, classification and structure of arguments in text”. In: Proceedings of the 12th
international conference on artificial intelligence and law. 2009, pp. 98–107.
```

[10] Christian Stab and Iryna Gurevych. “Identifying argumentative discourse structures in
persuasive essays”. In: Proceedings of the 2014 conference on empirical methods in
natural language processing (EMNLP). 2014, pp. 46–56.

[11] Niall Rooney, Hui Wang, and Fiona Browne. “Applying Kernel Methods to Argumen-
tation Mining.” In: FLAIRS. 2012.

[12] Jacob Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Lan-
guage Understanding. 2019. arXiv: 1810.04805 [cs.CL]. URL: https://
arxiv.org/abs/1810.04805.

[13] Yinhan Liu et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. 2019.
arXiv: 1907.11692 [cs.CL]. URL: https://arxiv.org/abs/1907. 11692.

[14] Robert Geirhos et al. “Shortcut learning in deep neural networks”. en. In: Nat. Mach.
Intell. 2.11 (Nov. 2020), pp. 665–673.

[15] Touch&#xE9; at CLEF 2026 - Generalizability of Argument Identification in Context
— touche.webis.de. https://touche.webis.de/clef26/touche26-
web/generalizable-argument-mining.html. [Accessed 17-04-2026].

[16] Shahriar Golchin and Mihai Surdeanu. “Data Contamination Quiz: A tool to detect
and estimate contamination in large language models”. In: arXiv [cs.CL] (Nov. 2023).

## Ehrenwörtliche Erklärung

Hiermit versichere ich, die vorliegende Masterarbeit selbstständig verfasst und keine anderen
als die angegebenen Quellen und Hilfsmittel benutzt zu haben. Alle Stellen, die aus den
Quellen entnommen wurden, sind als solche kenntlich gemacht worden. Diese Arbeit hat in
gleicher oder ähnlicher Form noch keiner Prüfungsbehörde vorgelegen.

Düsseldorf, 11. August 2026 Johannes Widera
