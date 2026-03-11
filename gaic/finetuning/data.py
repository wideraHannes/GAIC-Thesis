"""Dataset class for fine-tuning with chat formatting."""

import json

from datasets import Dataset
from loguru import logger

from config.paths import GAIC_DATA_DIR, CONTEXT_DIR


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


def format_chat_messages(sentence: str, label: str, definition: str) -> list[dict]:
    """Format a sample as chat messages for training."""
    context = f"## Argument Definition\n{definition}"
    system_content = SYSTEM_PROMPT.format(context=context)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": sentence},
        {"role": "assistant", "content": label},
    ]


def create_training_dataset(tokenizer) -> Dataset:
    """Create a HuggingFace Dataset formatted for SFT training."""
    texts, labels = load_train_data()

    # Cache definitions per dataset
    definition_cache = {}

    samples = []
    for item in texts:
        sample_id = item["id"]
        sentence = item["sentence"]
        label = labels.get(sample_id)

        if label is None:
            logger.warning(f"No label for {sample_id}, skipping")
            continue

        dataset = dataset_from_id(sample_id)

        # Load and cache definition
        if dataset not in definition_cache:
            definition_cache[dataset] = load_definition(dataset)
        definition = definition_cache[dataset]

        # Format as chat messages
        messages = format_chat_messages(sentence, label, definition)

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        samples.append({
            "id": sample_id,
            "dataset": dataset,
            "text": text,
            "messages": messages,
        })

    logger.info(f"Created training dataset with {len(samples)} samples")
    logger.info(f"Datasets: {list(definition_cache.keys())}")

    return Dataset.from_list(samples)


def create_eval_dataset(tokenizer) -> Dataset:
    """Create evaluation dataset from dev split."""
    texts, labels = load_dev_data()

    definition_cache = {}
    samples = []

    for item in texts:
        sample_id = item["id"]
        sentence = item["sentence"]
        label = labels.get(sample_id)

        if label is None:
            continue

        dataset = dataset_from_id(sample_id)

        if dataset not in definition_cache:
            definition_cache[dataset] = load_definition(dataset)
        definition = definition_cache[dataset]

        messages = format_chat_messages(sentence, label, definition)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        samples.append({
            "id": sample_id,
            "dataset": dataset,
            "text": text,
            "messages": messages,
        })

    logger.info(f"Created eval dataset with {len(samples)} samples")

    return Dataset.from_list(samples)
