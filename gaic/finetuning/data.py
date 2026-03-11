"""Dataset class for fine-tuning with chat formatting.

Supports balanced sampling: n samples per dataset (n/2 Argument + n/2 No-Argument).
"""

import json
import random
from collections import defaultdict

from datasets import Dataset
from loguru import logger

from config.paths import GAIC_DATA_DIR, CONTEXT_DIR


# All 10 datasets in the GAIC benchmark
ALL_DATASETS = [
    "ABSTRCT", "ACQUA", "AEC", "AFS", "ARGUMINSCI",
    "FINARG", "IAM", "PE", "SCIARK", "USELEC"
]

# System prompt template matching unified_experiment.py
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


def dataset_from_id(id_: str) -> str:
    """Extract dataset name from sample ID."""
    return id_.rsplit("-", 2)[0]


def load_definition(dataset: str) -> str:
    """Load definition.md for a dataset."""
    definition_path = CONTEXT_DIR / dataset / "definition.md"
    if definition_path.exists():
        return definition_path.read_text().strip()

    # Fallback to dataset.json
    json_path = CONTEXT_DIR / dataset / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            return data.get("definition", "")

    logger.warning(f"No definition found for {dataset}")
    return ""


def load_train_data() -> tuple[list[dict], dict[str, str]]:
    """Load training data and labels."""
    texts = []
    with open(GAIC_DATA_DIR / "train.jsonl") as f:
        for line in f:
            texts.append(json.loads(line))

    labels: dict[str, str] = {}
    with open(GAIC_DATA_DIR / "train_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]

    return texts, labels


def load_dev_data() -> tuple[list[dict], dict[str, str]]:
    """Load dev data and labels for evaluation."""
    texts = []
    with open(GAIC_DATA_DIR / "dev.jsonl") as f:
        for line in f:
            texts.append(json.loads(line))

    labels: dict[str, str] = {}
    with open(GAIC_DATA_DIR / "dev_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]

    return texts, labels


def sample_balanced_per_dataset(
    texts: list[dict],
    labels: dict[str, str],
    samples_per_dataset: int,
    datasets: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample n balanced samples per dataset (n/2 Argument + n/2 No-Argument).

    Args:
        texts: List of text items with 'id' and 'sentence' keys
        labels: Dict mapping sample ID to label
        samples_per_dataset: Number of samples per dataset (will take n/2 per class)
        datasets: List of datasets to include (default: all 10)
        seed: Random seed for reproducible sampling

    Returns:
        List of sampled items with labels attached
    """
    if datasets is None:
        datasets = ALL_DATASETS

    # Group samples by dataset and label
    by_dataset_label: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"Argument": [], "No-Argument": []}
    )

    for item in texts:
        sample_id = item["id"]
        dataset = dataset_from_id(sample_id)
        label = labels.get(sample_id)

        if label is None or dataset not in datasets:
            continue

        by_dataset_label[dataset][label].append({
            **item,
            "label": label,
        })

    # Sample balanced from each dataset
    rng = random.Random(seed)
    sampled = []
    k = samples_per_dataset // 2  # samples per class

    for dataset in datasets:
        args = by_dataset_label[dataset]["Argument"]
        no_args = by_dataset_label[dataset]["No-Argument"]

        # Shuffle before sampling to get different samples each time (with seed)
        rng.shuffle(args)
        rng.shuffle(no_args)

        # Take k from each class
        selected_args = args[:k]
        selected_no_args = no_args[:k]

        if len(selected_args) < k:
            logger.warning(f"{dataset}: Only {len(selected_args)} Argument samples (wanted {k})")
        if len(selected_no_args) < k:
            logger.warning(f"{dataset}: Only {len(selected_no_args)} No-Argument samples (wanted {k})")

        sampled.extend(selected_args)
        sampled.extend(selected_no_args)

        logger.info(f"{dataset}: {len(selected_args)} Arg + {len(selected_no_args)} No-Arg = {len(selected_args) + len(selected_no_args)} samples")

    # Shuffle all samples together to prevent dataset-order overfitting
    rng.shuffle(sampled)

    return sampled


def format_prompt_completion(sentence: str, label: str, definition: str) -> tuple[list[dict], list[dict]]:
    """Format a sample as prompt and completion for training.

    Returns separate prompt and completion in conversational format,
    which TRL uses for completion-only loss masking.

    Note: Mistral's chat template drops system messages, so we merge
    the system prompt into the user message to preserve the definition context.
    """
    context = f"## Argument Definition\n{definition}"
    system_content = SYSTEM_PROMPT.format(context=context)

    # Merge system + user into single user message (Mistral drops system messages)
    user_content = f"{system_content}\n\n## Input\n{sentence}"

    prompt = [{"role": "user", "content": user_content}]
    completion = [{"role": "assistant", "content": label}]

    return prompt, completion


def create_training_dataset(
    tokenizer,
    samples_per_dataset: int = 10,
    datasets: list[str] | None = None,
    seed: int = 42,
) -> Dataset:
    """Create a HuggingFace Dataset formatted for SFT training.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        samples_per_dataset: Number of samples per dataset (default: 10)
        datasets: List of datasets to include (default: all 10)
        seed: Random seed for reproducible sampling

    Returns:
        HuggingFace Dataset with balanced samples from all datasets
    """
    texts, labels = load_train_data()

    # Sample balanced from each dataset
    sampled = sample_balanced_per_dataset(
        texts, labels, samples_per_dataset, datasets, seed
    )

    # Cache definitions per dataset
    definition_cache = {}

    formatted_samples = []
    for item in sampled:
        sample_id = item["id"]
        sentence = item["sentence"]
        label = item["label"]
        dataset = dataset_from_id(sample_id)

        # Load and cache definition
        if dataset not in definition_cache:
            definition_cache[dataset] = load_definition(dataset)
        definition = definition_cache[dataset]

        # Format as prompt-completion (TRL auto-masks prompt tokens)
        prompt, completion = format_prompt_completion(sentence, label, definition)

        formatted_samples.append({
            "id": sample_id,
            "dataset": dataset,
            "label": label,
            "prompt": prompt,
            "completion": completion,
        })

    # Summary statistics
    dataset_counts = defaultdict(lambda: {"Argument": 0, "No-Argument": 0})
    for s in formatted_samples:
        dataset_counts[s["dataset"]][s["label"]] += 1

    logger.info(f"Created training dataset with {len(formatted_samples)} samples")
    logger.info(f"Datasets: {list(definition_cache.keys())}")
    logger.info(f"Samples per dataset: {samples_per_dataset} ({samples_per_dataset // 2} per class)")

    return Dataset.from_list(formatted_samples)


def create_eval_dataset(
    tokenizer,
    samples_per_dataset: int = 10,
    datasets: list[str] | None = None,
    seed: int = 42,
) -> Dataset:
    """Create evaluation dataset from dev split with balanced sampling.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        samples_per_dataset: Number of samples per dataset (default: 10)
        datasets: List of datasets to include (default: all 10)
        seed: Random seed for reproducible sampling
    """
    texts, labels = load_dev_data()

    # Sample balanced from each dataset
    sampled = sample_balanced_per_dataset(
        texts, labels, samples_per_dataset, datasets, seed
    )

    definition_cache = {}
    formatted_samples = []

    for item in sampled:
        sample_id = item["id"]
        sentence = item["sentence"]
        label = item["label"]
        dataset = dataset_from_id(sample_id)

        if dataset not in definition_cache:
            definition_cache[dataset] = load_definition(dataset)
        definition = definition_cache[dataset]

        # Format as prompt-completion (TRL auto-masks prompt tokens)
        prompt, completion = format_prompt_completion(sentence, label, definition)

        formatted_samples.append({
            "id": sample_id,
            "dataset": dataset,
            "label": label,
            "prompt": prompt,
            "completion": completion,
        })

    logger.info(f"Created eval dataset with {len(formatted_samples)} samples")

    return Dataset.from_list(formatted_samples)


def main():
    """Print sample examples from train and eval datasets."""
    from transformers import AutoTokenizer

    model_name = "mistralai/Ministral-8B-Instruct-2410"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create small datasets for inspection
    train_ds = create_training_dataset(
        tokenizer, samples_per_dataset=10, datasets=["ABSTRCT"]
    )
    eval_ds = create_eval_dataset(
        tokenizer, samples_per_dataset=10, datasets=["ABSTRCT"]
    )

    def print_sample(sample, idx: int):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx}")
        print(f"{'='*80}")
        print(f"ID: {sample['id']}")
        print(f"Dataset: {sample['dataset']}")
        print(f"Label: {sample['label']}")
        print("\n--- PROMPT (masked during training) ---")
        for msg in sample["prompt"]:
            print(f"\n[{msg['role'].upper()}]")
            print(msg["content"])
        print("\n--- COMPLETION (trained on this) ---")
        for msg in sample["completion"]:
            print(f"\n[{msg['role'].upper()}]")
            print(msg["content"])
        print("\n--- FULL TEXT (what model sees) ---")
        full_text = tokenizer.apply_chat_template(
            sample["prompt"] + sample["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        print(full_text)
        print("="*80)

    print("\n" + "#"*80)
    print("# TRAIN EXAMPLES")
    print("#"*80)
    print_sample(train_ds[0], 1)
    print_sample(train_ds[1], 2)

    print("\n" + "#"*80)
    print("# EVAL EXAMPLES")
    print("#"*80)
    print_sample(eval_ds[0], 1)
    print_sample(eval_ds[1], 2)


if __name__ == "__main__":
    main()
