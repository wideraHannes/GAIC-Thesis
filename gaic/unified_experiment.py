"""
Zero-shot manipulation experiment for GAIC thesis.

For each dataset: classify every sample 3 times (original, Feger, shuffle),
compute classification_report per variant, compare macro-F1 deltas.

Usage:
    uv run gaic/unified_experiment.py
    uv run gaic/unified_experiment.py config/experiments/my_config.toml
"""

import json
import random
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI
from sklearn.metrics import classification_report
from tqdm import tqdm

from gaic.helper import manipulate_sentence
from config.paths import PROJECT_ROOT, GAIC_DATA_DIR, CONTEXT_DIR

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiments" / "experiment_config.toml"


# -- data loading --


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_data() -> tuple[dict[str, str], dict[str, str]]:
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


def dataset_from_id(id_: str) -> str:
    return id_.rsplit("-", 2)[0]


def load_definition(dataset: str) -> str:
    path = CONTEXT_DIR / dataset / "definition.md"
    if path.exists():
        return path.read_text().strip()
    # fallback to dataset.json
    json_path = CONTEXT_DIR / dataset / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f).get("definition", "")
    return ""


def sample_balanced(
    texts: dict, labels: dict, dataset: str, n: int
) -> list[tuple[str, str, str]]:
    """Return list of (id, sentence, label) balanced across classes."""
    samples = [
        (id_, text, labels[id_])
        for id_, text in texts.items()
        if dataset_from_id(id_) == dataset
    ]
    args = [s for s in samples if s[2] == "Argument"]
    no_args = [s for s in samples if s[2] == "No-Argument"]
    k = n // 2
    return random.sample(args, min(k, len(args))) + random.sample(
        no_args, min(k, len(no_args))
    )


# -- manipulation --


def shuffle_sentence(sentence: str) -> str:
    words = sentence.split()
    random.shuffle(words)
    return " ".join(words)


MANIPULATIONS = {
    "original": lambda s: s,
    "feger": manipulate_sentence,
    "shuffle": shuffle_sentence,
}


# -- classification --


def make_client(cfg: dict) -> OpenAI:
    if cfg["provider"] == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    elif cfg["provider"] == "openai":
        return OpenAI(api_key=cfg.get("api_key", ""))
    raise ValueError(f"Unknown provider: {cfg['provider']}")


def classify(
    client: OpenAI, cfg: dict, prompts_cfg: dict, sentence: str, definition: str
) -> str:
    user_prompt = prompts_cfg["user"].format(definition=definition, sentence=sentence)
    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": prompts_cfg["system"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return normalize_label(resp.choices[0].message.content or "")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "ERROR"


def normalize_label(pred: str) -> str:
    p = pred.lower().strip()
    if "no-argument" in p or "no argument" in p or p == "no":
        return "No-Argument"
    if "argument" in p or p == "yes":
        return "Argument"
    return pred


# -- main experiment --


def run(config: dict):
    cfg_llm = config["llm"]
    cfg_prompts = config["prompts"]
    datasets = config["datasets"]["enabled"]
    sample_size = config["experiment"]["sample_size"]
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(cfg_llm)
    texts, labels = load_data()

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg_llm["model"],
        "sample_size": sample_size,
        "datasets": {},
    }

    for dataset in datasets:
        logger.info(f"--- {dataset} ---")
        definition = load_definition(dataset)
        samples = sample_balanced(texts, labels, dataset, sample_size)

        # show prompt once so you can sanity-check
        example_prompt = cfg_prompts["user"].format(
            definition=definition, sentence=samples[0][1]
        )
        logger.info(f"[SYSTEM]\n{cfg_prompts['system']}")
        logger.info(f"[USER]\n{example_prompt}")
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

                # fire all 3 manipulations in parallel
                futures = {}
                for name, fn in MANIPULATIONS.items():
                    manipulated = fn(sentence)
                    futures[name] = pool.submit(
                        classify, client, cfg_llm, cfg_prompts, manipulated, definition
                    )

                for name, fut in futures.items():
                    pred = fut.result()
                    preds[name].append(pred)
                    record[f"pred_{name}"] = pred
                sample_records.append(record)

        # classification_report per variant
        reports = {}
        for name, y_pred in preds.items():
            reports[name] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

        # deltas against original
        f1_original = reports["original"]["macro avg"]["f1-score"]
        deltas = {}
        for name in ["feger", "shuffle"]:
            f1_manip = reports[name]["macro avg"]["f1-score"]
            deltas[f"delta_{name}"] = round(f1_manip - f1_original, 4)

        results["datasets"][dataset] = {
            "n_samples": len(y_true),
            "reports": reports,
            "macro_f1_original": round(f1_original, 4),
            "macro_f1_feger": round(reports["feger"]["macro avg"]["f1-score"], 4),
            "macro_f1_shuffle": round(reports["shuffle"]["macro avg"]["f1-score"], 4),
            **deltas,
            "samples": sample_records,
        }

        logger.info(
            f"Macro-F1 original: {results['datasets'][dataset]['macro_f1_original']:.4f}"
        )
        logger.info(
            f"Macro-F1 feger:    {results['datasets'][dataset]['macro_f1_feger']:.4f}  (delta: {deltas['delta_feger']:+.4f})"
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
            "mean_macro_f1_feger": round(
                sum(d["macro_f1_feger"] for d in ds.values()) / n, 4
            ),
            "mean_macro_f1_shuffle": round(
                sum(d["macro_f1_shuffle"] for d in ds.values()) / n, 4
            ),
            "mean_delta_feger": round(
                sum(d["delta_feger"] for d in ds.values()) / n, 4
            ),
            "mean_delta_shuffle": round(
                sum(d["delta_shuffle"] for d in ds.values()) / n, 4
            ),
        }

    # save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        output_dir / f"manipulation_{cfg_llm['model'].replace(':', '_')}_{ts}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return
    run(load_config(config_path))


if __name__ == "__main__":
    main()
