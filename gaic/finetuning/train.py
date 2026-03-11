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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config.paths import PROJECT_ROOT
from gaic.finetuning.data import create_training_dataset, create_eval_dataset


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

    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = create_training_dataset(tokenizer)

    logger.info("Creating evaluation dataset...")
    eval_dataset = create_eval_dataset(tokenizer)

    # Debug mode: limit dataset sizes for quick sanity check
    max_train = train_cfg.get("max_train_samples")
    max_eval = train_cfg.get("max_eval_samples")
    if max_train:
        train_dataset = train_dataset.select(range(min(max_train, len(train_dataset))))
        logger.info(f"DEBUG: Limited train to {len(train_dataset)} samples")
    if max_eval:
        eval_dataset = eval_dataset.select(range(min(max_eval, len(eval_dataset))))
        logger.info(f"DEBUG: Limited eval to {len(eval_dataset)} samples")

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
        # SFT-specific parameters
        dataset_text_field="text",
        max_length=train_cfg.get("max_seq_length", 2048),
        packing=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

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
