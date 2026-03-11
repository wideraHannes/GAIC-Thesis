"""Fine-tuning script for Mistral 3B on GAIC argument classification.

Usage:
    uv run gaic/finetuning/train.py config/finetuning/mistral_3b_c1.toml
"""

import sys
import tomllib
from datetime import datetime
from pathlib import Path

import torch
import wandb
from loguru import logger
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from config.paths import PROJECT_ROOT
from gaic.finetuning.data import create_training_dataset, create_eval_dataset


def normalize_label(pred: str) -> str:
    """Normalize model output to Argument/No-Argument."""
    p = pred.lower().strip()
    # Check No-Argument first (more specific)
    if "no-argument" in p or "no argument" in p or "not an argument" in p or p == "no":
        return "No-Argument"
    # Then check Argument
    if "argument" in p or p == "yes":
        return "Argument"
    return "Unknown"


class F1EvalCallback(TrainerCallback):
    """Callback to compute F1 score on eval set during training."""

    def __init__(self, eval_dataset, tokenizer, num_samples: int = 50):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = min(num_samples, len(eval_dataset))

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Generate predictions and compute F1 after each evaluation."""
        model.eval()

        # Sample a subset for faster evaluation
        indices = list(range(self.num_samples))
        y_true = []
        y_pred = []

        for idx in indices:
            sample = self.eval_dataset[idx]
            true_label = sample["completion"][0]["content"]  # Assistant message is the label

            # Create prompt from prompt messages (add generation prompt for inference)
            prompt = self.tokenizer.apply_chat_template(
                sample["prompt"], tokenize=False, add_generation_prompt=True
            )

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the new tokens
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            pred_label = normalize_label(generated)
            y_true.append(true_label)
            y_pred.append(pred_label)

        # Compute metrics
        # Filter out "Unknown" predictions for F1 calculation
        valid_indices = [i for i, p in enumerate(y_pred) if p != "Unknown"]
        if valid_indices:
            y_true_valid = [y_true[i] for i in valid_indices]
            y_pred_valid = [y_pred[i] for i in valid_indices]

            f1 = f1_score(y_true_valid, y_pred_valid, average="macro")
            accuracy = accuracy_score(y_true_valid, y_pred_valid)
            parse_rate = len(valid_indices) / len(y_pred)
        else:
            f1 = 0.0
            accuracy = 0.0
            parse_rate = 0.0

        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "eval/f1_macro": f1,
                "eval/accuracy": accuracy,
                "eval/parse_rate": parse_rate,
                "eval/num_samples": len(y_pred),
            }, step=state.global_step)

        logger.info(f"Step {state.global_step}: F1={f1:.4f}, Acc={accuracy:.4f}, ParseRate={parse_rate:.2%}")

        model.train()


def load_config(path: Path) -> dict:
    """Load TOML configuration."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def setup_model_and_tokenizer(config: dict):
    """Load model and tokenizer with LoRA configuration."""
    model_cfg = config["model"]
    model_name = model_cfg["name"]

    logger.info(f"Loading model: {model_name}")

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
        logger.warning("No GPU available, using CPU (will be slow)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing if configured
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Configure LoRA
    lora_config = LoraConfig(
        r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: dict):
    """Run the fine-tuning training loop."""
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    wandb_cfg = config.get("wandb", {})

    # Setup output directory
    output_dir = PROJECT_ROOT / exp_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if wandb_cfg.get("project"):
        wandb.init(
            project=wandb_cfg["project"],
            name=exp_cfg.get("name", f"finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=config,
        )
        logger.info(f"W&B initialized: {wandb_cfg['project']}")

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Get data sampling config
    data_cfg = config.get("data", {})
    samples_per_dataset = data_cfg.get("samples_per_dataset", 10)
    eval_samples_per_dataset = data_cfg.get("eval_samples_per_dataset", samples_per_dataset)
    datasets = data_cfg.get("datasets")  # None = all 10 datasets
    seed = data_cfg.get("seed", 42)

    logger.info(f"Training: {samples_per_dataset} samples per dataset")
    logger.info(f"Eval: {eval_samples_per_dataset} samples per dataset")
    if datasets:
        logger.info(f"Datasets: {datasets}")
    else:
        logger.info("Datasets: all 10")

    # Create datasets with balanced sampling
    logger.info("Creating training dataset...")
    train_dataset = create_training_dataset(
        tokenizer,
        samples_per_dataset=samples_per_dataset,
        datasets=datasets,
        seed=seed,
    )

    logger.info("Creating evaluation dataset...")
    eval_dataset = create_eval_dataset(
        tokenizer,
        samples_per_dataset=eval_samples_per_dataset,
        datasets=datasets,
        seed=seed + 1,  # Different seed to avoid overlap
    )

    # Log sample from train and eval
    logger.info("=" * 60)
    logger.info("SAMPLE TRAIN EXAMPLE:")
    train_sample = train_dataset[0]
    logger.info(f"ID: {train_sample['id']}")
    logger.info(f"Dataset: {train_sample['dataset']}")
    logger.info(f"Label: {train_sample['label']}")
    logger.info(f"Prompt: {train_sample['prompt'][0]['content'][:500]}...")
    logger.info(f"Completion: {train_sample['completion'][0]['content']}")
    logger.info("=" * 60)
    logger.info("SAMPLE EVAL EXAMPLE:")
    eval_sample = eval_dataset[0]
    logger.info(f"ID: {eval_sample['id']}")
    logger.info(f"Dataset: {eval_sample['dataset']}")
    logger.info(f"Label: {eval_sample['label']}")
    logger.info(f"Prompt: {eval_sample['prompt'][0]['content'][:500]}...")
    logger.info(f"Completion: {eval_sample['completion'][0]['content']}")
    logger.info("=" * 60)

    # SFTConfig (TRL 0.29+ API) - combines TrainingArguments with SFT-specific params
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if wandb_cfg.get("project") else "none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=False,  # Required for MPS
        remove_unused_columns=False,
        # SFT-specific parameters (prompt-completion format auto-enables completion_only_loss)
        max_length=train_cfg.get("max_seq_length", 2048),
        packing=False,
    )

    # F1 evaluation callback
    f1_callback = F1EvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        num_samples=min(50, len(eval_dataset)),  # Limit for speed
    )

    # Initialize trainer (prompt-completion format auto-masks prompt tokens)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[f1_callback],
    )

    # Evaluate at step 0 (before any training)
    logger.info("Evaluating at step 0 (before training)...")
    trainer.state.global_step = 0
    f1_callback.on_evaluate(sft_config, trainer.state, None, model)

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_output_dir = output_dir / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    logger.info(f"Model saved to {final_output_dir}")

    # Finish wandb
    if wandb_cfg.get("project"):
        wandb.finish()

    return trainer


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: uv run gaic/finetuning/train.py <config.toml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    train(config)


if __name__ == "__main__":
    main()
