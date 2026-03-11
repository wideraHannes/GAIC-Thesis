"""
Evaluation experiment for finetuned models.

Runs the same manipulation experiment as unified_experiment.py but with local
finetuned models (LoRA adapters) instead of API calls.

Usage:
    uv run gaic/finetuning/eval/experiment.py config/experiments/finetuned/ministral_8b_ABSTRCT.toml
"""

import json
import random
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.paths import PROJECT_ROOT, GAIC_DATA_DIR, CONTEXT_DIR
from gaic.helper import manipulate_sentence


# -- data loading (reused from unified_experiment) --


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


# Context loading
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
        "per_sample": True,
    },
    "zero_shot": {
        "zero_shot": True,
    },
}

ZERO_SHOT_CONTEXT = """- "Argument" if the sentence is argumentative
- "No-Argument" if the sentence is not argumentative"""


def _load_dataset_json(dataset: str) -> dict:
    json_path = CONTEXT_DIR / dataset / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def load_context(dataset: str, sources: list[str]) -> dict[str, str]:
    result = {}
    dataset_json = _load_dataset_json(dataset)
    capabilities = dataset_json.get("capabilities", {})

    for source in sources:
        spec = CONTEXT_SOURCES.get(source)
        if spec is None:
            path = CONTEXT_DIR / dataset / source
            if path.exists():
                result[source] = path.read_text().strip()
            continue

        if spec.get("zero_shot", False):
            result[source] = ZERO_SHOT_CONTEXT
            continue

        cap_key = spec.get("capability")
        if cap_key and not capabilities.get(cap_key, True):
            continue

        if spec.get("per_sample", False):
            continue

        md_path = CONTEXT_DIR / dataset / spec.get("file")
        if md_path and md_path.exists():
            content = md_path.read_text().strip()
            if content:
                result[source] = content
                continue

        fallback = dataset_json.get(spec.get("json_key", ""), "")
        if fallback:
            result[source] = fallback

    return result


def load_document_context(dataset: str, sample_id: str) -> str:
    context_file = CONTEXT_DIR / dataset / "data" / f"{sample_id}.txt"
    if context_file.exists():
        return context_file.read_text().strip()
    return ""


def assemble_context(context: dict[str, str], document_context: str = "") -> str:
    parts = []
    labels = {
        "definition": "Argument Definition",
        "zero_shot": "Classification Criteria",
        "guideline": "Annotation Guideline",
        "document_context": "Document Context (Preceding Sentences)",
    }

    for name, content in context.items():
        if content:
            label = labels.get(name, name.replace("_", " ").title())
            parts.append(f"## {label}\n{content}")

    if document_context:
        label = labels["document_context"]
        parts.append(f"## {label}\n{document_context}")

    return "\n\n".join(parts) if parts else ""


def sample_balanced(
    texts: dict, labels: dict, dataset: str, n: int
) -> list[tuple[str, str, str]]:
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
    random.seed(42)
    random.shuffle(words)
    return " ".join(words)


MANIPULATIONS = {
    "original": lambda s: s,
    "content_only": manipulate_sentence,
    "shuffle": shuffle_sentence,
}


# -- prompts (same as unified_experiment) --

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


# -- model loading --


def load_model_and_tokenizer(config: dict):
    """Load base model with LoRA adapter."""
    model_cfg = config["model"]
    base_model_name = model_cfg["base_model"]
    adapter_path = PROJECT_ROOT / model_cfg["adapter_path"]

    logger.info(f"Loading base model: {base_model_name}")
    logger.info(f"Loading adapter from: {adapter_path}")

    # Determine device and dtype
    if torch.backends.mps.is_available():
        device_map = "mps"
        torch_dtype = torch.float16
        logger.info("Using MPS (Apple Silicon) backend")
    elif torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
        logger.info("Using CUDA backend")
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        logger.warning("No GPU available, using CPU")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer


# -- classification --


def normalize_label(pred: str) -> str:
    p = pred.lower().strip()
    if "no-argument" in p or "no argument" in p or "not an argument" in p or p == "no":
        return "No-Argument"
    if "argument" in p or p == "yes":
        return "Argument"
    return pred


def classify(model, tokenizer, sentence: str, context: str) -> str:
    """Classify a sentence using the finetuned model."""
    system_prompt = SYSTEM_PROMPT.format(context=context)
    user_prompt = USER_PROMPT.format(sentence=sentence)

    # Build chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only new tokens
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return normalize_label(generated)


# -- main experiment --


def run(config: dict, config_path: Path | None = None):
    datasets_cfg = config["datasets"]["enabled"]
    sample_size = config["experiment"]["sample_size"]
    experiment_name = config["experiment"].get("experiment_name", "finetuned_eval")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    context_sources = config.get("context", {}).get("sources", ["definition"])
    logger.info(f"Context sources: {context_sources}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Load data
    texts, labels = load_data()

    results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path) if config_path else None,
        "config": config,
        "prompts": {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
        },
        "model": {
            "base_model": config["model"]["base_model"],
            "adapter_path": config["model"]["adapter_path"],
        },
        "sample_size": sample_size,
        "context_sources": context_sources,
        "datasets": {},
    }

    use_doc_context = "document_context" in context_sources

    for dataset in datasets_cfg:
        logger.info(f"--- {dataset} ---")
        context_parts = load_context(dataset, context_sources)
        base_context_str = assemble_context(context_parts)
        logger.info(f"Loaded context for {dataset}: {list(context_parts.keys())}")

        samples = sample_balanced(texts, labels, dataset, sample_size)

        # Show example prompt
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

        # Collect predictions
        y_true = []
        preds = {name: [] for name in MANIPULATIONS}
        sample_records = []

        for sample_id, sentence, true_label in tqdm(samples, desc=dataset):
            y_true.append(true_label)
            record = {
                "id": sample_id,
                "sentence": sentence,
                "true_label": true_label,
            }

            # Load document context if enabled
            doc_context = ""
            if use_doc_context:
                doc_context = load_document_context(dataset, sample_id)
                record["document_context"] = doc_context

            full_context = assemble_context(context_parts, doc_context)

            # Run all 3 manipulations
            for name, fn in MANIPULATIONS.items():
                manipulated = fn(sentence)
                record[f"sent_{name}"] = manipulated
                pred = classify(model, tokenizer, manipulated, full_context)
                preds[name].append(pred)
                record[f"pred_{name}"] = pred

            sample_records.append(record)

        # Classification reports
        reports = {}
        for name, y_pred in preds.items():
            reports[name] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

        # Deltas
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
            f"Macro-F1 content_only: {results['datasets'][dataset]['macro_f1_content_only']:.4f}  (delta: {deltas['delta_content_only']:+.4f})"
        )
        logger.info(
            f"Macro-F1 shuffle: {results['datasets'][dataset]['macro_f1_shuffle']:.4f}  (delta: {deltas['delta_shuffle']:+.4f})"
        )

    # Overall summary
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

    # Save
    safe_model = config["model"]["adapter_path"].replace("/", "_").replace(".", "_")
    out_path = output_dir / f"{experiment_name}_{sample_size}_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    if len(sys.argv) < 2:
        logger.error(
            "Usage: uv run gaic/finetuning/eval/experiment.py <config.toml>"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    run(load_config(config_path), config_path=config_path)


if __name__ == "__main__":
    main()
