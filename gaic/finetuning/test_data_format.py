"""Validation tests for TRL 0.29 prompt-completion format.

Based on TRL documentation requirements:
- https://huggingface.co/docs/trl/sft_trainer

Run with: uv run python gaic/finetuning/test_data_format.py
"""

from transformers import AutoTokenizer
from gaic.finetuning.data import create_training_dataset, create_eval_dataset


def test_dataset_has_prompt_completion_fields(dataset, name: str):
    """TRL 0.29 requires 'prompt' and 'completion' fields for completion-only loss."""
    sample = dataset[0]

    assert "prompt" in sample, f"{name}: Missing 'prompt' field (required for completion-only loss)"
    assert "completion" in sample, f"{name}: Missing 'completion' field (required for completion-only loss)"
    assert "text" not in sample, f"{name}: Has 'text' field - TRL won't mask prompt tokens!"

    print(f"  [PASS] {name} has prompt/completion fields (no text field)")


def test_prompt_completion_are_message_lists(dataset, name: str):
    """Conversational format: prompt/completion must be lists of message dicts."""
    sample = dataset[0]

    assert isinstance(sample["prompt"], list), f"{name}: 'prompt' must be a list of messages"
    assert isinstance(sample["completion"], list), f"{name}: 'completion' must be a list of messages"
    assert len(sample["prompt"]) > 0, f"{name}: 'prompt' is empty"
    assert len(sample["completion"]) > 0, f"{name}: 'completion' is empty"

    # Check message structure
    for msg in sample["prompt"]:
        assert "role" in msg, f"{name}: prompt message missing 'role'"
        assert "content" in msg, f"{name}: prompt message missing 'content'"

    for msg in sample["completion"]:
        assert "role" in msg, f"{name}: completion message missing 'role'"
        assert "content" in msg, f"{name}: completion message missing 'content'"

    print(f"  [PASS] {name} has valid message structure")


def test_completion_is_assistant_role(dataset, name: str):
    """Completion should be assistant messages."""
    sample = dataset[0]

    for msg in sample["completion"]:
        assert msg["role"] == "assistant", f"{name}: completion role should be 'assistant', got '{msg['role']}'"

    print(f"  [PASS] {name} completion has assistant role")


def test_completion_contains_valid_label(dataset, name: str):
    """Completion should contain 'Argument' or 'No-Argument'."""
    for i, sample in enumerate(dataset):
        label = sample["completion"][0]["content"]
        assert label in ["Argument", "No-Argument"], (
            f"{name}[{i}]: completion content '{label}' not in ['Argument', 'No-Argument']"
        )

    print(f"  [PASS] {name} all completions have valid labels")


def test_label_field_matches_completion(dataset, name: str):
    """The 'label' metadata field should match completion content."""
    for i, sample in enumerate(dataset):
        completion_label = sample["completion"][0]["content"]
        metadata_label = sample["label"]
        assert completion_label == metadata_label, (
            f"{name}[{i}]: label mismatch - completion='{completion_label}', metadata='{metadata_label}'"
        )

    print(f"  [PASS] {name} labels match completion content")


def test_tokenizer_applies_chat_template(tokenizer, dataset, name: str):
    """Verify chat template produces valid tokenization."""
    sample = dataset[0]

    # TRL concatenates prompt + completion for training
    full_messages = sample["prompt"] + sample["completion"]

    # Apply chat template (what TRL does internally)
    text = tokenizer.apply_chat_template(full_messages, tokenize=False)

    assert len(text) > 0, f"{name}: chat template produced empty text"
    assert sample["completion"][0]["content"] in text, (
        f"{name}: completion label not found in templated text"
    )

    print(f"  [PASS] {name} chat template works correctly")


def test_prompt_only_template_for_inference(tokenizer, dataset, name: str):
    """Verify prompt-only template works for F1 evaluation."""
    sample = dataset[0]

    # For inference, we use only prompt with generation prompt
    prompt_text = tokenizer.apply_chat_template(
        sample["prompt"], tokenize=False, add_generation_prompt=True
    )

    assert len(prompt_text) > 0, f"{name}: prompt template produced empty text"
    # Should NOT contain the completion yet
    assert sample["completion"][0]["content"] not in prompt_text or prompt_text.endswith(sample["completion"][0]["content"]) is False, (
        f"{name}: prompt-only template should not contain full completion"
    )

    print(f"  [PASS] {name} prompt-only template works for inference")


def test_balanced_labels(dataset, name: str):
    """Check that dataset has balanced Argument/No-Argument labels."""
    arg_count = sum(1 for s in dataset if s["label"] == "Argument")
    no_arg_count = sum(1 for s in dataset if s["label"] == "No-Argument")

    total = len(dataset)
    balance_ratio = min(arg_count, no_arg_count) / max(arg_count, no_arg_count) if max(arg_count, no_arg_count) > 0 else 0

    print(f"  [INFO] {name}: {arg_count} Argument, {no_arg_count} No-Argument ({balance_ratio:.1%} balance)")

    assert balance_ratio > 0.8, f"{name}: Labels imbalanced - {arg_count} Arg vs {no_arg_count} No-Arg"
    print(f"  [PASS] {name} labels are balanced")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("TRL 0.29 DATA FORMAT VALIDATION")
    print("=" * 60)

    # Load tokenizer
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    print("\nCreating datasets...")
    train_ds = create_training_dataset(
        tokenizer, samples_per_dataset=10, datasets=["ABSTRCT"], seed=42
    )
    eval_ds = create_eval_dataset(
        tokenizer, samples_per_dataset=10, datasets=["ABSTRCT"], seed=43
    )

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Eval: {len(eval_ds)} samples")

    # Run tests
    print("\n" + "-" * 60)
    print("DATASET STRUCTURE TESTS")
    print("-" * 60)

    for ds, name in [(train_ds, "train"), (eval_ds, "eval")]:
        test_dataset_has_prompt_completion_fields(ds, name)
        test_prompt_completion_are_message_lists(ds, name)
        test_completion_is_assistant_role(ds, name)
        test_completion_contains_valid_label(ds, name)
        test_label_field_matches_completion(ds, name)

    print("\n" + "-" * 60)
    print("TOKENIZATION TESTS")
    print("-" * 60)

    for ds, name in [(train_ds, "train"), (eval_ds, "eval")]:
        test_tokenizer_applies_chat_template(tokenizer, ds, name)
        test_prompt_only_template_for_inference(tokenizer, ds, name)

    print("\n" + "-" * 60)
    print("BALANCE TESTS")
    print("-" * 60)

    for ds, name in [(train_ds, "train"), (eval_ds, "eval")]:
        test_balanced_labels(ds, name)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # Show sample
    print("\nSAMPLE DATA:")
    sample = train_ds[0]
    print(f"  ID: {sample['id']}")
    print(f"  Label: {sample['label']}")
    print(f"  Prompt role: {sample['prompt'][0]['role']}")
    print(f"  Prompt content (first 200 chars): {sample['prompt'][0]['content'][:200]}...")
    print(f"  Completion role: {sample['completion'][0]['role']}")
    print(f"  Completion content: {sample['completion'][0]['content']}")


if __name__ == "__main__":
    run_all_tests()
