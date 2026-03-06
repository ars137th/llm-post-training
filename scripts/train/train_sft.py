"""
Training Script for Supervised Fine-Tuning

Run SFT training with configuration management via Hydra.

Usage:
    # Basic training with defaults
    python scripts/train/train_sft.py

    # Override config
    python scripts/train/train_sft.py model=opt-350m data=conversation

    # Custom hyperparameters
    python scripts/train/train_sft.py training.learning_rate=1e-4 training.num_epochs=5

    # Full experiment config
    python scripts/train/train_sft.py experiment=sft_gpt2_conversation
"""

import os
import sys
from pathlib import Path

# Add src to path and set up project root for Hydra
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Hydra needs absolute path to configs
CONFIGS_PATH = str(project_root / "configs")

import torch
from transformers import TrainingArguments, set_seed
from datasets import Dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from src.models.language import LanguageModel
from src.data.loaders import load_dataset, split_dataset
from src.data.processors.text import TextProcessor, create_prompt_template
from src.core.sft.trainer import SFTTrainer, compute_sft_metrics
from src.core.sft.collator import DataCollatorForSFT, create_sft_dataset
from src.utils.compat import get_training_args_kwargs


console = Console()


def print_config(cfg: DictConfig):
    """Pretty print configuration."""
    console.print("\n[bold cyan]Configuration:[/bold cyan]")
    console.print(OmegaConf.to_yaml(cfg))


def print_model_info(model_wrapper):
    """Print model information in a table."""
    table = Table(title="Model Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Parameters", f"{model_wrapper.num_parameters:,}")
    table.add_row("Trainable Parameters", f"{model_wrapper.num_trainable_parameters:,}")
    table.add_row(
        "Trainable %",
        f"{100 * model_wrapper.num_trainable_parameters / model_wrapper.num_parameters:.2f}%"
    )
    table.add_row("Device", str(model_wrapper.device))
    table.add_row("Is PEFT Model", str(model_wrapper.is_peft_model))

    console.print(table)


def prepare_data(cfg: DictConfig, tokenizer):
    """
    Load and prepare dataset for SFT.

    Args:
        cfg: Configuration
        tokenizer: Tokenizer

    Returns:
        Train and eval datasets
    """
    console.print("\n[bold cyan]Loading dataset...[/bold cyan]")

    # Load dataset
    dataset = load_dataset(
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.get('dataset_config'),
        cache_dir=cfg.data.get('cache_dir'),
    )

    # Handle DatasetDict vs Dataset
    if hasattr(dataset, 'keys'):
        # DatasetDict with splits
        train_data = dataset[cfg.data.train_split]
        eval_data = dataset.get(cfg.data.eval_split)
    else:
        # Single dataset, need to split
        console.print("[yellow]Splitting dataset...[/yellow]")
        splits = split_dataset(dataset, train_size=0.9, val_size=0.1)
        train_data = splits['train']
        eval_data = splits['validation']

    # Limit samples if specified
    if cfg.data.max_train_samples:
        train_data = train_data.select(range(min(cfg.data.max_train_samples, len(train_data))))
    if cfg.data.max_eval_samples and eval_data:
        eval_data = eval_data.select(range(min(cfg.data.max_eval_samples, len(eval_data))))

    console.print(f"[green]Train samples: {len(train_data)}[/green]")
    if eval_data:
        console.print(f"[green]Eval samples: {len(eval_data)}[/green]")

    # Convert to SFT format
    console.print("\n[bold cyan]Processing dataset...[/bold cyan]")

    # Create prompt template
    prompt_template_fn = create_prompt_template(
        template_type=cfg.technique.get('prompt_template', 'alpaca')
    )

    # Process datasets
    def process_example(example):
        """Convert example to SFT format."""
        # Adapt to different dataset formats
        if 'instruction' in example and 'output' in example:
            # Alpaca-style
            prompt = prompt_template_fn(example['instruction'], example.get('input'))
            response = example['output']
        elif 'prompt' in example and 'response' in example:
            # Direct prompt-response
            prompt = example['prompt']
            response = example['response']
        elif 'text' in example:
            # Single text field, split roughly in half
            text = example['text']
            mid = len(text) // 2
            prompt = text[:mid]
            response = text[mid:]
        else:
            # Try to infer from keys
            keys = list(example.keys())
            console.print(f"[yellow]Warning: Unknown format. Keys: {keys}[/yellow]")
            prompt = str(example[keys[0]])
            response = str(example[keys[1]] if len(keys) > 1 else "")

        return {'prompt': prompt, 'response': response}

    # Process train data
    processed_train = []
    for example in train_data:
        try:
            proc = process_example(example)
            processed_train.append(proc)
        except Exception as e:
            console.print(f"[red]Error processing example: {e}[/red]")
            continue

    # Tokenize
    processor = TextProcessor(
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.max_length,
    )

    train_tokenized = create_sft_dataset(
        examples=processed_train,
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.max_length,
    )

    # Convert to Dataset
    train_dataset = Dataset.from_list(train_tokenized)

    eval_dataset = None
    if eval_data:
        processed_eval = []
        for example in eval_data:
            try:
                proc = process_example(example)
                processed_eval.append(proc)
            except Exception as e:
                continue

        eval_tokenized = create_sft_dataset(
            examples=processed_eval,
            tokenizer=tokenizer,
            max_length=cfg.tokenizer.max_length,
        )
        eval_dataset = Dataset.from_list(eval_tokenized)

    return train_dataset, eval_dataset


def main(cfg: DictConfig):
    """Main training function."""
    console.print("[bold green]Starting SFT Training[/bold green]")
    print_config(cfg)

    # Set seed
    set_seed(cfg.seed)

    # Load model
    console.print("\n[bold cyan]Loading model...[/bold cyan]")
    model_wrapper = LanguageModel.from_pretrained(
        model_name=cfg.model.name,
        use_lora=cfg.model.use_lora,
        lora_config=OmegaConf.to_container(cfg.model.lora_config, resolve=True) if cfg.model.use_lora else None,
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        trust_remote_code=True,
    )

    print_model_info(model_wrapper)

    # Prepare data
    train_dataset, eval_dataset = prepare_data(cfg, model_wrapper.tokenizer)

    # Create data collator
    data_collator = DataCollatorForSFT(
        tokenizer=model_wrapper.tokenizer,
        max_length=cfg.tokenizer.max_length,
        mask_prompt=cfg.technique.loss.mask_prompt,
    )

    # Setup training arguments
    console.print("\n[bold cyan]Setting up training...[/bold cyan]")

    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare logging directory for tensorboard
    logging_dir = f"{output_dir}/logs" if cfg.logging.use_tensorboard else None

    # Use compatibility helper for version-aware TrainingArguments
    training_args_kwargs = get_training_args_kwargs(
        output_dir=output_dir,
        eval_enabled=cfg.evaluation.do_eval,
        logging_dir=logging_dir,
        # All other arguments
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps if cfg.evaluation.do_eval else None,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.checkpoint.load_best_model_at_end,
        metric_for_best_model=cfg.checkpoint.metric_for_best_model,
        greater_is_better=cfg.checkpoint.greater_is_better,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        report_to=["wandb"] if cfg.logging.use_wandb else ["tensorboard"] if cfg.logging.use_tensorboard else [],
        run_name=cfg.logging.wandb_run_name,
        seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers,
    )

    training_args = TrainingArguments(**training_args_kwargs)

    # Create trainer
    trainer = SFTTrainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model_wrapper.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_sft_metrics if cfg.evaluation.do_eval else None,
        loss_type=cfg.technique.loss.type,
        log_predictions=cfg.technique.logging.log_predictions,
        num_predictions_to_log=cfg.technique.logging.num_predictions_to_log,
    )

    # Train
    console.print("\n[bold green]Starting training...[/bold green]")
    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from_checkpoint)

    # Save final model
    console.print("\n[bold cyan]Saving model...[/bold cyan]")
    final_model_path = f"{output_dir}/final_model"
    model_wrapper.model.save_pretrained(final_model_path)
    model_wrapper.tokenizer.save_pretrained(final_model_path)
    console.print(f"[green]Model saved to {final_model_path}[/green]")

    # Final evaluation
    if eval_dataset:
        console.print("\n[bold cyan]Final evaluation...[/bold cyan]")
        metrics = trainer.evaluate()
        console.print("[green]Evaluation metrics:[/green]")
        for key, value in metrics.items():
            console.print(f"  {key}: {value:.4f}")

    console.print("\n[bold green]Training complete! 🎉[/bold green]")


if __name__ == "__main__":
    # Initialize Hydra with config path
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Use absolute path to configs directory
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_PATH, job_name="train_sft"):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
        main(cfg)
