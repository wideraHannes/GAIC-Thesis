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

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL
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
    "zero_shot": {
        "zero_shot": True,  # True zero-shot: no definition, no context
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

        # Handle zero-shot (simple criteria, no dataset-specific definition)
        if spec.get("zero_shot", False):
            result[source] = ZERO_SHOT_CONTEXT
            logger.info(f"Zero-shot mode for {dataset} (using simple criteria)")
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


def assemble_context(context: dict[str, str], document_context: str = "") -> str:
    """Format loaded context sources into a single string for the prompt."""
    parts = []
    labels = {
        "definition": "Argument Definition",
        "zero_shot": "Classification Criteria",
        "guideline": "Annotation Guideline",
        "document_context": "Document Context (Preceding Sentences)",
    }

    # Add dataset-level context first
    for name, content in context.items():
        if content:
            label = labels.get(name, name.replace("_", " ").title())
            parts.append(f"## {label}\n{content}")

    # Add document context if provided
    if document_context:
        label = labels["document_context"]
        parts.append(f"## {label}\n{document_context}")

    return "\n\n".join(parts) if parts else ""


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


# -- manipulation --


def shuffle_sentence(sentence: str) -> str:
    words = sentence.split()
    random.seed(42)  # fixed seed for reproducibility across runs
    random.shuffle(words)
    return " ".join(words)


MANIPULATIONS = {
    "original": lambda s: s,
    "content_only": manipulate_sentence,
    "shuffle": shuffle_sentence,
}


# -- prompts --

# Zero-shot context: simple classification criteria without dataset-specific definitions
ZERO_SHOT_CONTEXT = """- "Argument" if the sentence is argumentative
- "No-Argument" if the sentence is not argumentative"""

SYSTEM_PROMPT = """## Role
You are a Dataset Annotator.

## Task
Classify the input as exactly one of these two labels:
- "Argument"
- "No-Argument"

## Output Format
Respond with ONLY the label. No explanation. No other text.

## Rules
- Classify as "Argument" if the sentence matches the definition below.
- Classify as "No-Argument" otherwise.

{context}"""

USER_PROMPT = "{sentence}"


# -- classification --


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
        return OpenAI(api_key=cfg.get("api_key", ""))
    raise ValueError(f"Unknown provider: {provider}")


def classify(client: OpenAI, cfg: dict, sentence: str, context: str, max_retries: int = 3) -> str:
    system_prompt = SYSTEM_PROMPT.format(context=context)
    user_prompt = USER_PROMPT.format(sentence=sentence)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg["temperature"],
            )
            return normalize_label(resp.choices[0].message.content or "")
        except Exception as e:
            error_str = str(e).lower()
            # Retry on transient errors (rate limits, timeouts, server errors)
            is_retryable = any(x in error_str for x in ["rate", "limit", "timeout", "503", "502", "429", "overloaded"])

            if is_retryable and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM error (attempt {attempt + 1}/{max_retries}): {e} -> defaulting to No-Argument")
                return "No-Argument"

    return "No-Argument"


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
        "context_sources": context_sources,
        "datasets": {},
    }

    # Check if document context is requested
    use_doc_context = "document_context" in context_sources

    for dataset in datasets:
        logger.info(f"--- {dataset} ---")
        context_parts = load_context(dataset, context_sources)
        base_context_str = assemble_context(context_parts)
        logger.info(f"Loaded context for {dataset}: {list(context_parts.keys())}")
        if use_doc_context:
            logger.info("Document context enabled (per-sample loading)")
        samples = sample_balanced(texts, labels, dataset, sample_size)

        # show prompt once so you can sanity-check
        # (if doc context enabled, show example with first sample's context)
        if use_doc_context:
            example_doc_ctx = load_document_context(dataset, samples[0][0])
            example_context = assemble_context(context_parts, example_doc_ctx)
        else:
            example_context = base_context_str
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

                # Assemble full context for this sample
                full_context = assemble_context(context_parts, doc_context)

                # fire all 3 manipulations in parallel
                futures = {}
                for name, fn in MANIPULATIONS.items():
                    manipulated = fn(sentence)
                    record[f"sent_{name}"] = manipulated
                    futures[name] = pool.submit(
                        classify, client, cfg_llm, manipulated, full_context
                    )

                for name, fut in futures.items():
                    pred = fut.result()
                    preds[name].append(pred)
                    record[f"pred_{name}"] = pred
                sample_records.append(record)

                # Rate limit: 2 seconds per sample (3 requests) stays under 6 req/sec limit
            # time.sleep(2)

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

        results["datasets"][dataset] = {
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
