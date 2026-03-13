"""
Context Quality Evaluation Experiment for GAIC thesis.

Evaluates the quality of extracted definitions and guidelines using LLM-as-judge:
- Task A: Definition-Sample Fit — Does the definition explain the ground-truth label?
- Task B: Definition-Guideline Alignment — Are definition and guideline consistent?

Usage:
    uv run gaic/context_quality_experiment.py
    uv run gaic/context_quality_experiment.py config/experiments/context_quality/eval_config.toml
"""

import json
import os
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL
from tqdm import tqdm

from config.paths import PROJECT_ROOT, GAIC_DATA_DIR, CONTEXT_DIR

load_dotenv()

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiments" / "context_quality" / "eval_config.toml"


# -- data loading (reused from unified_experiment.py) --


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


def _load_dataset_json(dataset: str) -> dict:
    """Load dataset.json for a given dataset."""
    json_path = CONTEXT_DIR / dataset / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def load_definition(dataset: str) -> str:
    """Load definition for a dataset."""
    md_path = CONTEXT_DIR / dataset / "definition.md"
    if md_path.exists():
        return md_path.read_text().strip()
    # Fallback to dataset.json
    dataset_json = _load_dataset_json(dataset)
    return dataset_json.get("definition", "")


def load_guideline(dataset: str) -> str | None:
    """Load guideline for a dataset if available."""
    dataset_json = _load_dataset_json(dataset)
    capabilities = dataset_json.get("capabilities", {})
    if not capabilities.get("has_guidelines", False):
        return None
    md_path = CONTEXT_DIR / dataset / "guideline.md"
    if md_path.exists():
        return md_path.read_text().strip()
    return dataset_json.get("guideline")


# -- LLM client (reused from unified_experiment.py) --


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


# -- Evaluation prompts --

TASK_A_SYSTEM = """You are an expert evaluator assessing definition quality for argument identification datasets.

Your task: Given a sentence with its ground-truth label ("Argument" or "No-Argument") and a definition of what constitutes an "Argument" in this dataset, evaluate how well the definition explains the assigned label.

## Rating Scale (1-5):
- 5: Definition clearly explains the label — the criteria unambiguously lead to this classification
- 4: Adequate fit — minor interpretation needed but definition largely supports the label
- 3: Partial fit — some ambiguity, definition could support either label
- 2: Poor fit — definition is too vague or unclear to derive the label
- 1: Definition contradicts the label — applying the definition would suggest the opposite label

## Output Format (JSON):
{
    "score": <1-5>,
    "reasoning": "<brief explanation of your assessment>",
    "label_derivable": <true/false — could the label be derived from the definition alone?>
}

Respond with ONLY the JSON object, no other text."""

TASK_A_USER = """## Definition
{definition}

## Sample
Sentence: "{sentence}"
Ground-truth label: {label}

Evaluate how well the definition explains this label assignment."""


TASK_B_SYSTEM = """You are an expert evaluator assessing consistency between argument definitions and annotation guidelines.

Your task: Given a definition of "Argument" and annotation guidelines for a dataset, evaluate how well they align.

## Rating Scale (1-5):
- 5: Fully aligned — definition and guidelines use the same criteria with no contradictions
- 4: Mostly aligned — minor gaps or additional details in one but no contradictions
- 3: Partially aligned — notable gaps or different emphasis, but no direct contradictions
- 2: Weakly aligned — significant mismatches in what counts as an argument
- 1: Not aligned — contradictions between definition and guidelines

## Output Format (JSON):
{
    "score": <1-5>,
    "reasoning": "<brief explanation of alignment/misalignment>",
    "gaps_in_definition": "<criteria in guideline but missing from definition, or 'none'>",
    "gaps_in_guidelines": "<criteria in definition but missing from guideline, or 'none'>"
}

Respond with ONLY the JSON object, no other text."""

TASK_B_USER = """## Definition
{definition}

## Annotation Guideline
{guideline}

Evaluate the alignment between this definition and guideline."""


# -- Evaluation functions --


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        response = match.group(1)
    # Try direct parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r"\{[^{}]*\}", response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse JSON", "raw_response": response}


def evaluate_definition_fit(
    client: OpenAI,
    cfg: dict,
    definition: str,
    sentence: str,
    label: str,
    max_retries: int = 3,
) -> dict:
    """Task A: Evaluate how well the definition explains the sample's label."""
    user_prompt = TASK_A_USER.format(
        definition=definition, sentence=sentence, label=label
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": TASK_A_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg["temperature"],
            )
            content = resp.choices[0].message.content or ""
            return parse_json_response(content)
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(
                x in error_str
                for x in ["rate", "limit", "timeout", "503", "502", "429", "overloaded"]
            )
            if is_retryable and attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM error: {e}")
                return {"error": str(e)}
    return {"error": "Max retries exceeded"}


def evaluate_alignment(
    client: OpenAI,
    cfg: dict,
    definition: str,
    guideline: str,
    max_retries: int = 3,
) -> dict:
    """Task B: Evaluate alignment between definition and guideline."""
    user_prompt = TASK_B_USER.format(definition=definition, guideline=guideline)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": TASK_B_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg["temperature"],
            )
            content = resp.choices[0].message.content or ""
            return parse_json_response(content)
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(
                x in error_str
                for x in ["rate", "limit", "timeout", "503", "502", "429", "overloaded"]
            )
            if is_retryable and attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM error: {e}")
                return {"error": str(e)}
    return {"error": "Max retries exceeded"}


def compute_metrics(results: dict) -> dict:
    """Compute aggregated metrics from results."""
    metrics = {
        "task_a": {
            "by_dataset": {},
            "by_label": {"Argument": [], "No-Argument": []},
            "overall": {"scores": [], "derivable_count": 0, "total": 0},
        },
        "task_b": {"scores": [], "datasets": []},
    }

    # Task A metrics
    for dataset, data in results.get("task_a", {}).items():
        dataset_scores = []
        for sample in data.get("samples", []):
            if "error" not in sample.get("evaluation", {}):
                score = sample["evaluation"].get("score")
                if score is not None:
                    dataset_scores.append(score)
                    metrics["task_a"]["overall"]["scores"].append(score)
                    metrics["task_a"]["by_label"][sample["label"]].append(score)
                    if sample["evaluation"].get("label_derivable"):
                        metrics["task_a"]["overall"]["derivable_count"] += 1
                    metrics["task_a"]["overall"]["total"] += 1

        if dataset_scores:
            metrics["task_a"]["by_dataset"][dataset] = {
                "mean_score": round(sum(dataset_scores) / len(dataset_scores), 2),
                "n_samples": len(dataset_scores),
            }

    # Task A overall
    if metrics["task_a"]["overall"]["scores"]:
        scores = metrics["task_a"]["overall"]["scores"]
        metrics["task_a"]["overall"]["mean_score"] = round(sum(scores) / len(scores), 2)
        metrics["task_a"]["overall"]["derivable_rate"] = round(
            metrics["task_a"]["overall"]["derivable_count"]
            / metrics["task_a"]["overall"]["total"],
            2,
        )

    # Task A by label
    for label in ["Argument", "No-Argument"]:
        scores = metrics["task_a"]["by_label"][label]
        if scores:
            metrics["task_a"]["by_label"][label] = {
                "mean_score": round(sum(scores) / len(scores), 2),
                "n_samples": len(scores),
            }

    # Task B metrics
    for dataset, data in results.get("task_b", {}).items():
        if "error" not in data:
            score = data.get("score")
            if score is not None:
                metrics["task_b"]["scores"].append(score)
                metrics["task_b"]["datasets"].append({"dataset": dataset, "score": score})

    if metrics["task_b"]["scores"]:
        scores = metrics["task_b"]["scores"]
        metrics["task_b"]["mean_score"] = round(sum(scores) / len(scores), 2)

    return metrics


# -- Main experiment --


def run(config: dict, config_path: Path | None = None):
    cfg_llm = config["llm"]
    datasets = config["datasets"]["enabled"]
    sample_size = config["experiment"]["sample_size"]
    experiment_name = config["experiment"].get("experiment_name", "context_quality_eval")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(cfg_llm)
    texts, labels = load_data()

    results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path) if config_path else None,
        "config": config,
        "prompts": {
            "task_a_system": TASK_A_SYSTEM,
            "task_a_user": TASK_A_USER,
            "task_b_system": TASK_B_SYSTEM,
            "task_b_user": TASK_B_USER,
        },
        "model": cfg_llm["model"],
        "sample_size": sample_size,
        "task_a": {},
        "task_b": {},
    }

    # Task A: Definition-Sample Fit (all datasets)
    logger.info("=" * 60)
    logger.info("TASK A: Definition-Sample Fit")
    logger.info("=" * 60)

    for dataset in datasets:
        logger.info(f"--- {dataset} ---")
        definition = load_definition(dataset)
        if not definition:
            logger.warning(f"No definition found for {dataset}, skipping")
            continue

        samples = sample_balanced(texts, labels, dataset, sample_size)
        sample_results = []

        for sample_id, sentence, label in tqdm(samples, desc=f"{dataset} Task A"):
            evaluation = evaluate_definition_fit(
                client, cfg_llm, definition, sentence, label
            )
            sample_results.append({
                "id": sample_id,
                "sentence": sentence,
                "label": label,
                "evaluation": evaluation,
            })
            # Rate limit
            time.sleep(1)

        results["task_a"][dataset] = {
            "definition": definition,
            "samples": sample_results,
        }

        # Log summary
        scores = [
            s["evaluation"]["score"]
            for s in sample_results
            if "error" not in s["evaluation"] and "score" in s["evaluation"]
        ]
        if scores:
            mean_score = sum(scores) / len(scores)
            logger.info(f"Mean score: {mean_score:.2f} (n={len(scores)})")

    # Task B: Definition-Guideline Alignment (datasets with guidelines)
    logger.info("=" * 60)
    logger.info("TASK B: Definition-Guideline Alignment")
    logger.info("=" * 60)

    for dataset in datasets:
        guideline = load_guideline(dataset)
        if guideline is None:
            logger.info(f"Skipping {dataset} (no guidelines)")
            continue

        logger.info(f"--- {dataset} ---")
        definition = load_definition(dataset)
        if not definition:
            logger.warning(f"No definition found for {dataset}, skipping")
            continue

        evaluation = evaluate_alignment(client, cfg_llm, definition, guideline)
        results["task_b"][dataset] = {
            "definition": definition,
            "guideline": guideline,
            **evaluation,
        }

        if "error" not in evaluation:
            logger.info(f"Alignment score: {evaluation.get('score', 'N/A')}")
        # Rate limit
        time.sleep(1)

    # Compute metrics
    results["metrics"] = compute_metrics(results)

    # Log overall summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    metrics = results["metrics"]
    if metrics["task_a"]["overall"].get("mean_score"):
        logger.info(f"Task A mean score: {metrics['task_a']['overall']['mean_score']}")
        logger.info(f"Task A derivable rate: {metrics['task_a']['overall']['derivable_rate']}")
    if metrics["task_b"].get("mean_score"):
        logger.info(f"Task B mean score: {metrics['task_b']['mean_score']}")

    # Save results
    safe_model = (
        cfg_llm["model"]
        .replace("/", "_")
        .replace(":", "_")
        .replace("@", "")
        .replace(".", "_")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{experiment_name}_{safe_model}_{timestamp}.json"
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
