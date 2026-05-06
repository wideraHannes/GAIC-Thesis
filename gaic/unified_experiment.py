"""
Zero-shot manipulation experiment for GAIC thesis.

For each dataset: classify every sample 3 times (original, Feger, shuffle),
compute classification_report per variant, compare macro-F1 deltas.

Usage:
    uv run gaic/unified_experiment.py
    uv run gaic/unified_experiment.py config/experiments/my_config.toml
"""

import json
import os
import random
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report
from tqdm import tqdm

from gaic.helper import manipulate_sentence
from config.paths import PROJECT_ROOT, GAIC_DATA_DIR, CONTEXT_DIR

load_dotenv()

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiments" / "experiment_config.toml"


# -- data loading --


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_data() -> tuple[dict[str, str], dict[str, str]]:
    texts, labels = {}, {}
    with open(GAIC_DATA_DIR / "dev.jsonl") as f:
        for line in f:
            item = json.loads(line)
            texts[item["id"]] = item["sentence"]
    with open(GAIC_DATA_DIR / "dev_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]
    return texts, labels


def dataset_from_id(id_: str) -> str:
    return id_.rsplit("-", 2)[0]


# Map of context source names to their file names, dataset.json fallback keys,
# and the capability flag that must be true in dataset.json
CONTEXT_SOURCES = {
    "definition": {
        "file": "definition.md",
        "json_key": "definition",
        "capability": "has_definition",
    },
    "guideline": {
        "file": "guideline.md",
        "json_key": "guideline",
        "capability": "has_guidelines",
    },
    "document_context": {
        "capability": "has_document_context",
        "per_sample": True,  # Loaded per sample, not per dataset
    },
}


def _load_dataset_json(dataset: str) -> dict:
    """Load and cache dataset.json for a given dataset."""
    json_path = CONTEXT_DIR / dataset / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def load_context(dataset: str, sources: list[str]) -> dict[str, str]:
    """Load requested context sources for a dataset.

    Checks dataset.json capabilities to skip sources that aren't available
    (e.g. has_guidelines: false). Returns a dict mapping source name -> content.

    Note: document_context is per-sample, so it returns empty string and must be loaded separately.
    """
    result = {}
    dataset_json = _load_dataset_json(dataset)
    capabilities = dataset_json.get("capabilities", {})

    for source in sources:
        spec = CONTEXT_SOURCES.get(source)
        if spec is None:
            # treat as a raw filename in the context dir
            path = CONTEXT_DIR / dataset / source
            if path.exists():
                result[source] = path.read_text().strip()
            else:
                logger.warning(
                    f"Unknown context source '{source}' for {dataset}, skipped"
                )
            continue

        # check capability flag — skip if explicitly false
        cap_key = spec.get("capability")
        if cap_key and not capabilities.get(cap_key, True):
            logger.info(f"Skipping '{source}' for {dataset} ({cap_key}=false)")
            continue

        # Skip per-sample sources (loaded separately in the loop)
        if spec.get("per_sample", False):
            logger.info(f"'{source}' is per-sample, will load during classification")
            continue

        # try the .md file first
        md_path = CONTEXT_DIR / dataset / spec.get("file")
        if md_path and md_path.exists():
            content = md_path.read_text().strip()
            if content:
                result[source] = content
                continue

        # fallback to dataset.json value
        fallback = dataset_json.get(spec.get("json_key", ""), "")
        if fallback:
            result[source] = fallback

    return result


def load_document_context(dataset: str, sample_id: str) -> str:
    """Load document context (preceding sentences) for a specific sample."""
    context_file = CONTEXT_DIR / dataset / "data" / f"{sample_id}.txt"
    if context_file.exists():
        return context_file.read_text().strip()
    return ""


def assemble_context(
    context: dict[str, str],
    document_context: str = "",
    demonstrations: list[dict] | None = None,
) -> str:
    """Format loaded context sources into a single string for the prompt.

    Uses templates from CONTEXT_SECTION_TEMPLATES with explicit ordering:
    definition → guideline → few_shot → document_context (most general to most specific).
    Falls back to c0_fallback when no context is provided.
    """
    parts = []

    # Add context in explicit order
    for name in CONTEXT_ORDER:
        # Handle document_context specially (passed as separate argument)
        if name == "document_context":
            if document_context:
                template = CONTEXT_SECTION_TEMPLATES[name]
                parts.append(template.format(content=document_context))
        # Handle few_shot specially (passed as separate argument)
        elif name == "few_shot":
            if demonstrations:
                formatted = format_demonstrations(demonstrations)
                template = CONTEXT_SECTION_TEMPLATES[name]
                parts.append(template.format(content=formatted))
        elif name in context and context[name]:
            template = CONTEXT_SECTION_TEMPLATES[name]
            parts.append(template.format(content=context[name]))

    # Fallback for c0 (no context) - true zero-shot with minimal task instruction
    if not parts:
        parts.append(CONTEXT_SECTION_TEMPLATES["c0_fallback"])
    else:
        # For c1-c3+: add task section referencing the criteria above
        parts.append(
            "## Task\nClassify whether the following sentence is an argument based on the criteria above."
        )

    return "\n\n".join(parts)


def sample_balanced(
    texts: dict, labels: dict, dataset: str, n: int
) -> list[tuple[str, str, str]]:
    """Return first n/2 Argument + first n/2 No-Argument samples (deterministic)."""
    samples = [
        (id_, text, labels[id_])
        for id_, text in texts.items()
        if dataset_from_id(id_) == dataset
    ]
    args = [s for s in samples if s[2] == "Argument"]
    no_args = [s for s in samples if s[2] == "No-Argument"]
    k = n // 2
    return args[:k] + no_args[:k]


# -- few-shot demonstrations --


def load_train_data() -> tuple[dict[str, str], dict[str, str]]:
    """Load train split for few-shot demonstrations."""
    texts, labels = {}, {}
    with open(GAIC_DATA_DIR / "train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            texts[item["id"]] = item["sentence"]
    with open(GAIC_DATA_DIR / "train_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]
    return texts, labels


def load_demonstrations(
    dataset: str,
    k: int,
    strategy: str = "deterministic",
    test_sentence: str = None,
    collection=None,
) -> list[dict]:
    """Load demonstrations using specified strategy.

    Strategies:
    - "deterministic": First k Arg + first k No-Arg from dataset's train split
    - "retrieval": Top-k most similar Arg + No-Arg from same dataset's train data

    Returns 2k total demos, grouped by label.
    """
    if strategy == "retrieval":
        from gaic.embeddings import get_collection, retrieve_similar_demos

        if collection is None:
            collection = get_collection()
        return retrieve_similar_demos(test_sentence, k, collection, dataset=dataset)

    # Default: deterministic
    train_texts, train_labels = load_train_data()
    dataset_ids = sorted(
        [id_ for id_ in train_texts if dataset_from_id(id_) == dataset]
    )

    arg_ids = [id_ for id_ in dataset_ids if train_labels[id_] == "Argument"][:k]
    noarg_ids = [id_ for id_ in dataset_ids if train_labels[id_] == "No-Argument"][:k]

    # Interleave: Arg, No-Arg, Arg, No-Arg, ...
    demos = []
    for arg_id, noarg_id in zip(arg_ids, noarg_ids):
        demos.append({"sentence": train_texts[arg_id], "label": "Argument"})
        demos.append({"sentence": train_texts[noarg_id], "label": "No-Argument"})

    return demos


def format_demonstrations(demos: list[dict]) -> str:
    """Format demos for injection into context, grouped by label."""
    # Separate by label
    args = [d for d in demos if d["label"] == "Argument"]
    noargs = [d for d in demos if d["label"] == "No-Argument"]

    lines = []

    # Arguments section
    if args:
        lines.append("### Sentences that ARE Arguments:")
        for i, d in enumerate(args, 1):
            lines.append(f'{i}. "{d["sentence"]}"')
        lines.append("")

    # No-Arguments section
    if noargs:
        lines.append("### Sentences that are NOT Arguments:")
        for i, d in enumerate(noargs, 1):
            lines.append(f'{i}. "{d["sentence"]}"')

    return "\n".join(lines)


# -- manipulation --


def shuffle_sentence(sentence: str) -> str:
    # Preserve ending punctuation
    end_punct = ""
    if sentence and sentence[-1] in ".!?":
        end_punct = sentence[-1]
        sentence = sentence[:-1].strip()

    words = sentence.split()
    random.seed(42)  # fixed seed for reproducibility across runs
    random.shuffle(words)
    return " ".join(words) + end_punct


MANIPULATIONS = {
    "original": lambda s: s,
    "content_only": manipulate_sentence,
    "shuffle": shuffle_sentence,
}


# -- prompts --

# System prompt template - context comes FIRST for better attention
SYSTEM_PROMPT = """You are an expert in argumentation analysis.

{context}"""

USER_PROMPT = "{sentence}"

# Context section templates with usage guidance
# Order matters: definition → guideline (overrides) → document_context (supplementary)
CONTEXT_SECTION_TEMPLATES = {
    "c0_fallback": """## Task
Classify whether the sentence is an argument or not.""",
    "definition": """## Argument Definition
{content}""",
    "guideline": """## Annotation Guideline
{content}

When the definition and guideline conflict, follow the guideline.""",
    "document_context": """## Document Context
The following sentences immediately precede the target sentence in the original document:

{content}

Use this context to resolve ambiguity when the target sentence's meaning depends on preceding text.""",
    "few_shot": """## Examples

{content}""",
}

# Explicit ordering for context assembly (most general → most specific)
# few_shot goes after guideline but before document_context
CONTEXT_ORDER = ["definition", "guideline", "few_shot", "document_context"]


# -- classification --


class ClassificationWithReasoning(BaseModel):
    """Structured output schema for argument classification with reasoning."""

    reason: str = Field(description="Brief rationale (1 sentence max)")
    label: Literal["Argument", "No-Argument"]


class ClassificationNoReasoning(BaseModel):
    """Structured output schema for argument classification without reasoning."""

    label: Literal["Argument", "No-Argument"]


def make_client(cfg: dict) -> OpenAI:
    provider = cfg["provider"]
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    if provider == "groq":
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    if provider == "portkey":
        return OpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key=os.environ["PORTKEY_API_KEY"],
        )
    if provider == "together_ai":
        return OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )
    if provider == "mistral":
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
        )
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


def classify(
    client: OpenAI, cfg: dict, sentence: str, context: str, max_retries: int = 3
) -> dict[str, str]:
    """Classify a sentence and return both reasoning and label."""
    system_prompt = SYSTEM_PROMPT.format(context=context)
    user_prompt = USER_PROMPT.format(sentence=sentence)

    # Select schema based on reasoning flag (default False)
    use_reasoning = cfg.get("reasoning", False)
    schema = ClassificationWithReasoning if use_reasoning else ClassificationNoReasoning

    for attempt in range(max_retries):
        try:
            resp = client.beta.chat.completions.parse(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg["temperature"],
                response_format=schema,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is not None:
                reason = getattr(parsed, "reason", "") if use_reasoning else ""
                return {"reason": reason, "label": parsed.label}
            # Fallback: try to parse content if structured parsing failed
            content = resp.choices[0].message.content or ""
            return {"reason": "", "label": normalize_label(content)}
        except Exception as e:
            error_str = str(e).lower()
            # Retry on transient errors (rate limits, timeouts, server errors)
            is_retryable = any(
                x in error_str
                for x in ["rate", "limit", "timeout", "503", "502", "429", "overloaded"]
            )

            if is_retryable and attempt < max_retries - 1:
                wait_time = 2**attempt  # exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"LLM error (attempt {attempt + 1}/{max_retries}): {e} -> defaulting to No-Argument"
                )
                return {"reason": f"Error: {e}", "label": "No-Argument"}

    return {"reason": "Max retries exceeded", "label": "No-Argument"}


def normalize_label(pred: str) -> str:
    p = pred.lower().strip()
    # Check No-Argument first (more specific)
    if "no-argument" in p or "no argument" in p or "not an argument" in p or p == "no":
        return "No-Argument"
    # Then check Argument
    if "argument" in p or p == "yes":
        return "Argument"
    return pred


# -- main experiment --


def run(config: dict, config_path: Path | None = None):
    cfg_llm = config["llm"]
    datasets = config["datasets"]["enabled"]
    sample_size = config["experiment"]["sample_size"]
    experiment_name = config["experiment"].get("experiment_name", "experiment")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # context sources: default to ["definition"] for backward compat
    context_sources = config.get("context", {}).get("sources", ["definition"])
    logger.info(f"Context sources: {context_sources}")

    # reasoning flag: default False
    use_reasoning = cfg_llm.get("reasoning", False)
    logger.info(f"Reasoning enabled: {use_reasoning}")

    # few-shot config: k = number of positive AND negative examples (total = 2k)
    fs_config = config.get("few_shot", {})
    few_shot_k = fs_config.get("k", 0)
    few_shot_strategy = fs_config.get("strategy", "deterministic")
    if few_shot_k > 0:
        logger.info(
            f"Few-shot enabled: k={few_shot_k} (total {2 * few_shot_k} examples), strategy={few_shot_strategy}"
        )

    # Load ChromaDB collection once if using retrieval strategy
    chroma_collection = None
    if few_shot_k > 0 and few_shot_strategy == "retrieval":
        from gaic.embeddings import get_collection

        chroma_collection = get_collection()
        logger.info("Loaded ChromaDB collection for retrieval")

    client = make_client(cfg_llm)
    texts, labels = load_data()

    results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path) if config_path else None,
        "config": config,
        "prompts": {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
        },
        "model": cfg_llm["model"],
        "sample_size": sample_size,
        "reasoning_enabled": use_reasoning,
        "context_sources": context_sources,
        "few_shot_k": few_shot_k,
        "few_shot_strategy": few_shot_strategy,
        "datasets": {},
    }

    # Check if document context is requested
    use_doc_context = "document_context" in context_sources

    for dataset in datasets:
        logger.info(f"--- {dataset} ---")
        context_parts = load_context(dataset, context_sources)

        # Load few-shot demonstrations if enabled (deterministic: once per dataset)
        demonstrations = []
        if few_shot_k > 0 and few_shot_strategy == "deterministic":
            demonstrations = load_demonstrations(
                dataset, few_shot_k, strategy="deterministic"
            )
            logger.info(f"Loaded {len(demonstrations)} demonstrations for {dataset}")

        base_context_str = assemble_context(
            context_parts, demonstrations=demonstrations
        )
        logger.info(f"Loaded context for {dataset}: {list(context_parts.keys())}")
        if use_doc_context:
            logger.info("Document context enabled (per-sample loading)")
        if few_shot_strategy == "retrieval":
            logger.info("Retrieval strategy: demos will be loaded per-sample")
        samples = sample_balanced(texts, labels, dataset, sample_size)

        # show prompt once so you can sanity-check
        # (if doc context enabled, show example with first sample's context)
        # For retrieval, get example demos for first sample
        example_demos = demonstrations
        if few_shot_k > 0 and few_shot_strategy == "retrieval":
            example_demos = load_demonstrations(
                dataset,
                few_shot_k,
                strategy="retrieval",
                test_sentence=samples[0][1],
                collection=chroma_collection,
            )
        if use_doc_context:
            example_doc_ctx = load_document_context(dataset, samples[0][0])
            example_context = assemble_context(
                context_parts, example_doc_ctx, example_demos
            )
        else:
            example_context = assemble_context(
                context_parts, demonstrations=example_demos
            )
        example_system = SYSTEM_PROMPT.format(context=example_context)
        example_user = USER_PROMPT.format(sentence=samples[0][1])
        logger.info(f"[SYSTEM]\n{example_system}")
        logger.info(f"[USER]\n{example_user}")
        logger.info("-" * 40)

        # collect predictions per manipulation
        y_true = []
        preds = {name: [] for name in MANIPULATIONS}
        sample_records = []

        with ThreadPoolExecutor(max_workers=len(MANIPULATIONS)) as pool:
            for sample_id, sentence, true_label in tqdm(samples, desc=dataset):
                y_true.append(true_label)
                record = {
                    "id": sample_id,
                    "sentence": sentence,
                    "true_label": true_label,
                }

                # Load document context for this sample if enabled
                doc_context = ""
                if use_doc_context:
                    doc_context = load_document_context(dataset, sample_id)
                    record["document_context"] = doc_context

                # Load demonstrations (per-sample for retrieval, reuse for deterministic)
                sample_demos = demonstrations
                if few_shot_k > 0 and few_shot_strategy == "retrieval":
                    sample_demos = load_demonstrations(
                        dataset,
                        few_shot_k,
                        strategy="retrieval",
                        test_sentence=sentence,
                        collection=chroma_collection,
                    )

                # Assemble full context for this sample
                full_context = assemble_context(
                    context_parts, doc_context, sample_demos
                )

                # fire all 3 manipulations in parallel
                futures = {}
                for name, fn in MANIPULATIONS.items():
                    manipulated = fn(sentence)
                    record[f"sent_{name}"] = manipulated
                    futures[name] = pool.submit(
                        classify, client, cfg_llm, manipulated, full_context
                    )

                for name, fut in futures.items():
                    result = fut.result()
                    preds[name].append(result["label"])
                    record[f"pred_{name}"] = result["label"]
                    record[f"reason_{name}"] = result["reason"]
                sample_records.append(record)

                # Rate limit: 2 seconds per sample (3 requests) stays under 6 req/sec limit

                time.sleep(2)

        # classification_report per variant
        reports = {}
        for name, y_pred in preds.items():
            reports[name] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

        # deltas against original
        f1_original = reports["original"]["macro avg"]["f1-score"]
        deltas = {}
        for name in ["content_only", "shuffle"]:
            f1_manip = reports[name]["macro avg"]["f1-score"]
            deltas[f"delta_{name}"] = round(f1_manip - f1_original, 4)

        dataset_result = {
            "n_samples": len(y_true),
            "reports": reports,
            "macro_f1_original": round(f1_original, 4),
            "macro_f1_content_only": round(
                reports["content_only"]["macro avg"]["f1-score"], 4
            ),
            "macro_f1_shuffle": round(reports["shuffle"]["macro avg"]["f1-score"], 4),
            **deltas,
            "samples": sample_records,
        }
        # Track demonstrations used for reproducibility
        if demonstrations:
            dataset_result["demonstrations"] = demonstrations
        results["datasets"][dataset] = dataset_result

        logger.info(
            f"Macro-F1 original: {results['datasets'][dataset]['macro_f1_original']:.4f}"
        )
        logger.info(
            f"Macro-F1 content_only:    {results['datasets'][dataset]['macro_f1_content_only']:.4f}  (delta: {deltas['delta_content_only']:+.4f})"
        )
        logger.info(
            f"Macro-F1 shuffle:  {results['datasets'][dataset]['macro_f1_shuffle']:.4f}  (delta: {deltas['delta_shuffle']:+.4f})"
        )

    # overall summary
    ds = results["datasets"]
    n = len(ds)
    if n:
        results["overall"] = {
            "mean_macro_f1_original": round(
                sum(d["macro_f1_original"] for d in ds.values()) / n, 4
            ),
            "mean_macro_f1_content_only": round(
                sum(d["macro_f1_content_only"] for d in ds.values()) / n, 4
            ),
            "mean_macro_f1_shuffle": round(
                sum(d["macro_f1_shuffle"] for d in ds.values()) / n, 4
            ),
            "mean_delta_content_only": round(
                sum(d["delta_content_only"] for d in ds.values()) / n, 4
            ),
            "mean_delta_shuffle": round(
                sum(d["delta_shuffle"] for d in ds.values()) / n, 4
            ),
        }

    # save
    safe_model = (
        cfg_llm["model"]
        .replace("/", "_")
        .replace(":", "_")
        .replace("@", "")
        .replace(".", "_")
    )
    out_path = output_dir / f"{experiment_name}_{sample_size}_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return
    run(load_config(config_path), config_path=config_path)


if __name__ == "__main__":
    main()
