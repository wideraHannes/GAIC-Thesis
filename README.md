# Can Large Language Models Generalize Arguments in Context?

**Master's Thesis** | Heinrich Heine University Dusseldorf | February -- August 2026

**Student:** Johannes Widera | **Supervisor:** Marc Feger

**Target Venue:** [Touche @ CLEF 2026 -- Generalizable Argument Identification in Context (GAIC)](https://touche.webis.de/clef26/touche26-web/generalizable-argument-mining.html)

---

## The Problem

State-of-the-art argument mining models learn **datasets, not arguments**. Feger et al. ([ACL 2025](https://aclanthology.org/2025.acl-long.1280/)) showed that BERT, RoBERTa, and DistilBERT achieve strong in-distribution performance (mean F1 = 0.79) but collapse on cross-dataset transfer (F1 = 0.56--0.61). Removing stop words, function words, and discourse markers -- roughly 50% of all words -- caused *almost no performance change* (Delta <= 0.02). These encoder models rely on content-word shortcuts, not the linguistic structure that defines argumentation.

Their paper concludes with an open direction: **decoder-based argument mining**.

## The Idea

Decoder-based LLMs process language fundamentally differently. Causal attention enforces strict left-to-right processing -- word order is not just learned but architecturally required. A discourse marker like *"therefore"* directly shapes the entire chain of computation from that point forward. This leads to a concrete prediction: **LLMs should be sensitive to the linguistic scaffolding that encoders ignore.**

This thesis tests that prediction. Across five models spanning 7B to frontier scale and 10 benchmark datasets from the GAIC shared task, we systematically investigate whether decoder-based LLMs overcome the shortcut learning problem -- and what happens when you give them rich context like annotation guidelines and source documents.

## Methodology

The thesis follows a two-part structure, each addressing a core research question:

### Part 1: Do Zero-Shot LLMs Rely on Argument Structure?

Every sentence is classified three times under controlled text manipulations:

| Manipulation | What it does |
|---|---|
| **M0: Original** | Unmodified sentence |
| **M1: Feger** | Remove stop words, function words, discourse markers, punctuation |
| **M2: Shuffle** | Randomly permute word order |

If removing linguistic scaffolding hurts performance (large Delta), the model relies on argument structure. If it doesn't (small Delta), the model uses shortcuts -- just like the encoders.

**Preliminary result:** Four out of five LLMs show |Delta_feger| >= 0.085, that is 4--10x larger than encoder Delta <= 0.02. Decoders *do* process argument structure.

### Part 2: Can LLMs Leverage Rich Context?

The GAIC task provides annotation guidelines, source documents, and paper references -- but only for some datasets. We test three context conditions:

| Condition | Input |
|---|---|
| **Baseline** | Sentence + generic argument definition |
| **Guidelines** | Baseline + dataset-specific annotation guidelines |
| **Full** | Guidelines + document context (preceding sentences) |

**Preliminary result:** Context effects range from +0.33 to -0.20 F1 depending on model and dataset. Context helps struggling models but can introduce noise when performance is already adequate. Context also reduces manipulation sensitivity, suggesting it provides an alternative reasoning channel.

## Models Under Test

| Model | Size | Provider |
|---|---|---|
| Mistral-7B-Instruct-v0.2 | 7B | [Together AI](https://together.ai) |
| Llama-3.1-8B-Instruct | 8B | [Together AI](https://together.ai) |
| Mistral-Small-24B-Instruct | 24B | [Together AI](https://together.ai) |
| Llama-3.1-70B-Instruct | 70B | [Together AI](https://together.ai) |
| GPT-4.1 | frontier | [Portkey](https://portkey.ai) (Azure OpenAI) |

Open-source models run through **[Together AI](https://together.ai)** for blazingly fast inference. GPT-4.1 is accessed via **[Portkey](https://portkey.ai)** as a unified gateway to Azure OpenAI Foundry.

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
uv run python gaic/unified_experiment.py config/experiments/portkey_gpt4.1_config_30.toml

# Cross-guideline transfer experiment
uv run python gaic/cross_guideline_experiment.py

# Data contamination test
uv run python gaic/contamination_test.py
```

Experiments are fully parameterized via TOML configs in `config/experiments/`. Each config specifies the LLM provider, model, context sources, sample size, and enabled datasets.

## References

- Feger, M., Boland, K., & Dietze, S. (2025). *Limited generalizability in argument mining: State-of-the-art models learn datasets, not arguments.* Proceedings of ACL 2025, 23900--23915.
- Geirhos, R. et al. (2020). *Shortcut learning in deep neural networks.* Nature Machine Intelligence, 2(11), 665--673.
- GAIC Shared Task (2026). *Generalizable Argument Identification in Context.* Touche @ CLEF 2026.
