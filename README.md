# Can Large Language Models Generalize Arguments in Context?

**Master's Thesis** | Heinrich Heine University Dusseldorf | February -- August 2026

**Student:** Johannes Widera | **Supervisor:** Marc Feger

**Target Venue:** [Touché @ CLEF 2026 -- Generalizable Argument Identification in Context (GAIC)](https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html)

---

## The Problem

State-of-the-art argument mining models learn **datasets, not arguments**. Feger et al. ([ACL 2025](https://aclanthology.org/2025.acl-long.1280/)) showed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but collapse on cross-dataset transfer (F1 = 0.56--0.61). Removing stop words, function words, and discourse markers -- roughly 50% of all words -- caused *almost no performance change* (Δ ≤ 0.02). These encoder models rely on content-word shortcuts, not the linguistic structure that defines argumentation.

Their paper concludes with an open direction: **decoder-based argument mining**.

## The Idea

Decoder-based LLMs process language fundamentally differently. Causal attention enforces strict left-to-right processing -- word order is not just learned but architecturally required. A discourse marker like *"therefore"* directly shapes the entire chain of computation from that point forward. This leads to a concrete prediction: **LLMs should be sensitive to the linguistic scaffolding that encoders ignore.**

This thesis tests that prediction. Across three models spanning 7B to frontier scale and 10 benchmark datasets from the GAIC shared task, we systematically investigate whether decoder-based LLMs overcome the shortcut learning problem -- and what happens when you give them rich context like annotation guidelines and source documents.

## Modular Experiment Architecture

The experiment framework is built around **composable, TOML-configurable components**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOML Configuration                              │
│  [experiment] name, sample_size, output_dir                             │
│  [llm] provider, model, temperature                                     │
│  [context] sources = ["definition", "guideline", "document_context"]    │
│  [datasets] enabled = ["ABSTRCT", "PE", ...]                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      unified_experiment.py                              │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ Data Loader  │   │   Context    │   │    Text      │                 │
│  │              │   │   Assembler  │   │ Manipulator  │                 │
│  │ • train.jsonl│   │              │   │              │                 │
│  │ • dev.jsonl  │   │ • definition │   │ • M0: orig   │                 │
│  │ • labels     │   │ • guideline  │   │ • M1: content│                 │
│  │ • balanced   │   │ • doc_context│   │ • M2: shuffle│                 │
│  │   sampling   │   │ • few-shot   │   │   (spaCy)    │                 │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    LLM Classification                            │   │
│  │  • Multi-provider: OpenAI, Portkey, Together AI, Mistral, Ollama │   │
│  │  • Structured output (Pydantic schemas)                          │   │
│  │  • Parallel execution via ThreadPoolExecutor                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│                            ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Metrics & Output                              │   │
│  │  • classification_report per manipulation                        │   │
│  │  • Δ_content_only, Δ_shuffle deltas                              │   │
│  │  • Full sample-level predictions + config → JSON                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Configuration-Driven**: Every experiment is fully parameterized via TOML. Swap models, context sources, or datasets without touching code.

2. **Pluggable Context Sources**: Context is assembled from modular sources (`definition`, `guideline`, `document_context`, `few_shot`) that can be combined arbitrarily. Each dataset declares its capabilities in `dataset.json`.

3. **Provider Abstraction**: LLM access is unified through the OpenAI client interface with `base_url` switching. Adding a new provider requires only a new case in `make_client()`.

4. **Parallel Manipulations**: For each sample, all three text manipulations (M0, M1, M2) are classified in parallel via `ThreadPoolExecutor`, maximizing throughput.

5. **Reproducible Outputs**: Results JSON files include the full experiment config, prompts, and sample-level predictions for complete reproducibility.

## Methodology

The thesis follows a two-part structure, each addressing a core research question:

### Part 1: Do Zero-Shot LLMs Rely on Argument Structure?

Every sentence is classified three times under controlled text manipulations:

| Manipulation | What it does |
|---|---|
| **M0: Original** | Unmodified sentence |
| **M1: Content-Only** | Remove stop words, function words, discourse markers, punctuation (via spaCy) |
| **M2: Shuffle** | Randomly permute word order (seed=42) |

If removing linguistic scaffolding hurts performance (large Δ), the model relies on argument structure. If it doesn't (small Δ), the model uses shortcuts -- just like the encoders.

### Part 2: Can LLMs Leverage Rich Context?

The GAIC task provides annotation guidelines, source documents, and paper references -- but only for some datasets. We test three context conditions:

| Condition | Input |
|---|---|
| **Baseline** | Sentence + generic argument definition |
| **Guidelines** | Baseline + dataset-specific annotation guidelines |
| **Full** | Guidelines + document context (preceding sentences) |

## Models

| Model | Size | Provider |
|---|---|---|
| Mistral-7B-Instruct-v0.2 | 7B | [Mistral AI](https://mistral.ai) |
| Mistral-Small-24B-Instruct | 24B | [Mistral AI](https://mistral.ai) |
| GPT-5.2 | frontier | [Portkey](https://portkey.ai) (Azure OpenAI) |

Mistral models are accessed via the **[Mistral AI API](https://mistral.ai)**. GPT-5.2 is accessed via **[Portkey](https://portkey.ai)** as a unified gateway to Azure OpenAI.

## Datasets

10 benchmark datasets from the GAIC shared task (~17k labeled sentences):

| Dataset | Domain | Document Context | Guidelines |
|---|---|---|---|
| ABSTRCT | Biomedical abstracts | Yes | Yes |
| ARGUMINSCI | Scientific papers | Yes | Yes |
| PE | Persuasive essays | Yes | Yes |
| USELEC | US election debates | Yes | Yes |
| FINARG | Financial text | Yes | -- |
| SCIARK | Scientific articles | Yes | -- |
| ACQUA | Argument quality | -- | -- |
| AEC | Argument efficacy | -- | -- |
| AFS | Argument facet similarity | -- | -- |
| IAM | Internet argument mining | -- | -- |

## Running Experiments

```bash
# Install dependencies
uv sync

# Run an experiment with a specific config
uv run gaic/unified_experiment.py config/experiments/<config>.toml

# Cross-guideline transfer experiment
uv run gaic/cross_guideline_experiment.py

# Data contamination test
uv run gaic/contamination_test.py
```

### Example Configuration

```toml
# config/experiments/mistral_small_baseline.toml

[experiment]
experiment_name = "baseline"
sample_size = 30  # balanced Argument / No-Argument per dataset
output_dir = "experiments/unified_outputs"

[llm]
provider = "mistral"
model = "mistral-small-latest"
temperature = 0.0

[context]
sources = ["definition"]  # options: "definition", "guideline", "document_context"

[datasets]
enabled = ["ABSTRCT", "PE", "ARGUMINSCI", "USELEC", ...]
```

Experiments are fully parameterized via TOML configs in `config/experiments/`. Each config specifies the LLM provider, model, context sources, sample size, and enabled datasets.

## References

- Feger, M., Boland, K., & Dietze, S. (2025). *Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments.* Proceedings of ACL 2025, 23900--23915.
- Geirhos, R. et al. (2020). *Shortcut learning in deep neural networks.* Nature Machine Intelligence, 2(11), 665--673.
- GAIC Shared Task (2026). *Generalizable Argument Identification in Context.* Touche @ CLEF 2026.
