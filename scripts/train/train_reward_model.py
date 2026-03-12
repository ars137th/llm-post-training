"""
Training Script for Reward Modeling

Run reward model training with configuration management via Hydra.

Usage:
    # Basic training with defaults
    python scripts/train/train_reward_model.py

    # Override config
    python scripts/train/train_reward_model.py model=gpt2 data.num_examples=1000

    # Custom hyperparameters
    python scripts/train/train_reward_model.py training.learning_rate=1e-5 training.num_epochs=2

    # Full experiment config
    python scripts/train/train_reward_model.py experiment=reward_gpt2_synthetic
"""

import os
import sys
from pathlib import Path

# Force CPU if MPS has issues (Apple Silicon compatibility)
# Set this before importing torch
if os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

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
from src.models.reward import RewardModel
from src.data.processors.preference import (
    load_preference_dataset,
    prepare_preference_data,
    create_synthetic_preference_data,
    PreferenceDataCollator,
    parse_anthropic_format,
)
from src.core.reward_modeling.trainer import (
    RewardModelTrainer,
    compute_reward_metrics,
    evaluate_reward_model,
)
from src.utils.compat import get_training_args_kwargs


console = Console()


def print_config(cfg: DictConfig):
    """Pretty print configuration."""
    console.print("\n[bold cyan]Configuration:[/bold cyan]")
    console.print(OmegaConf.to_yaml(cfg))


def print_reward_model_info(reward_model: RewardModel):
    """Print reward model information in a table."""
    table = Table(title="Reward Model Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Base Model", reward_model.model.config._name_or_path)
    table.add_row("Total Parameters", f"{reward_model.num_parameters:,}")
    table.add_row("Trainable Parameters", f"{reward_model.num_trainable_parameters:,}")
    table.add_row(
        "Trainable %",
        f"{reward_model.percent_trainable:.2f}%"
    )
    table.add_row("Value Head Parameters", f"{reward_model.num_value_head_parameters:,}")
    table.add_row("Device", str(reward_model.device))
    table.add_row("Is PEFT Model", str(reward_model.is_peft_model))

    console.print(table)


def prepare_data(cfg: DictConfig, tokenizer):
    """
    Load and prepare preference dataset.

    Args:
        cfg: Configuration
        tokenizer: Tokenizer

    Returns:
        Train and eval datasets
    """
    console.print("\n[bold cyan]Loading dataset...[/bold cyan]")

    if cfg.data.use_synthetic:
        # Generate synthetic data for testing
        console.print("[yellow]Generating synthetic preference data...[/yellow]")
        train_examples = create_synthetic_preference_data(
            num_examples=cfg.data.num_train_examples,
            seed=cfg.seed,
        )
        eval_examples = create_synthetic_preference_data(
            num_examples=cfg.data.num_eval_examples,
            seed=cfg.seed + 1,
        )

        # Convert to Dataset
        train_data = Dataset.from_list(train_examples)
        eval_data = Dataset.from_list(eval_examples)

        console.print(f"[green]Generated {len(train_data)} train examples[/green]")
        console.print(f"[green]Generated {len(eval_data)} eval examples[/green]")

    else:
        # Load real dataset
        console.print(f"[yellow]Loading {cfg.data.dataset_name}...[/yellow]")

        train_data = load_preference_dataset(
            dataset_name=cfg.data.dataset_name,
            dataset_config=cfg.data.get('dataset_config'),
            split=cfg.data.train_split,
            cache_dir=cfg.data.get('cache_dir'),
        )

        eval_data = None
        if cfg.data.eval_split:
            eval_data = load_preference_dataset(
                dataset_name=cfg.data.dataset_name,
                dataset_config=cfg.data.get('dataset_config'),
                split=cfg.data.eval_split,
                cache_dir=cfg.data.get('cache_dir'),
            )

        # Limit examples if specified
        if cfg.data.num_train_examples:
            train_data = train_data.select(range(min(cfg.data.num_train_examples, len(train_data))))
        if cfg.data.num_eval_examples and eval_data:
            eval_data = eval_data.select(range(min(cfg.data.num_eval_examples, len(eval_data))))

        console.print(f"[green]Train examples: {len(train_data)}[/green]")
        if eval_data:
            console.print(f"[green]Eval examples: {len(eval_data)}[/green]")

    # Prepare data (tokenize)
    console.print("\n[bold cyan]Processing dataset...[/bold cyan]")

    # Determine format parser
    format_fn = None
    if cfg.data.format == "anthropic":
        format_fn = parse_anthropic_format

    train_dataset = prepare_preference_data(
        train_data,
        tokenizer,
        max_length=cfg.tokenizer.max_length,
        format_fn=format_fn,
    )

    eval_dataset = None
    if eval_data:
        eval_dataset = prepare_preference_data(
            eval_data,
            tokenizer,
            max_length=cfg.tokenizer.max_length,
            format_fn=format_fn,
        )

    console.print(f"[green]Tokenized {len(train_dataset)} train pairs[/green]")
    if eval_dataset:
        console.print(f"[green]Tokenized {len(eval_dataset)} eval pairs[/green]")

    return train_dataset, eval_dataset


def main(cfg: DictConfig):
    """Main training function."""
    console.print("[bold green]Starting Reward Model Training[/bold green]")
    print_config(cfg)

    # Set seed
    set_seed(cfg.seed)

    # Load base model (SFT model if available)
    console.print("\n[bold cyan]Loading base model...[/bold cyan]")
    base_model = LanguageModel.from_pretrained(
        model_name=cfg.model.name,
        use_lora=cfg.model.use_lora,
        lora_config=OmegaConf.to_container(cfg.model.lora_config, resolve=True) if cfg.model.use_lora else None,
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        trust_remote_code=True,
    )

    # Create reward model
    console.print("\n[bold cyan]Creating reward model...[/bold cyan]")
    reward_model = RewardModel(
        base_model=base_model,
        freeze_base=cfg.technique.freeze_base_model,
    )

    # Force to CPU if specified (workaround for MPS limitations)
    if cfg.device == "cpu":
        console.print("[yellow]Forcing model to CPU (MPS compatibility mode)[/yellow]")
        reward_model = reward_model.cpu()

    print_reward_model_info(reward_model)

    # Prepare data
    train_dataset, eval_dataset = prepare_data(cfg, reward_model.tokenizer)

    # Create data collator
    data_collator = PreferenceDataCollator(
        tokenizer=reward_model.tokenizer,
        max_length=cfg.tokenizer.max_length,
    )

    # Setup training arguments
    console.print("\n[bold cyan]Setting up training...[/bold cyan]")

    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare logging directory
    logging_dir = f"{output_dir}/logs" if cfg.logging.use_tensorboard else None

    # Use compatibility helper for version-aware TrainingArguments
    training_args_kwargs = get_training_args_kwargs(
        output_dir=output_dir,
        eval_enabled=cfg.evaluation.do_eval and eval_dataset is not None,
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
        eval_steps=cfg.training.eval_steps if cfg.evaluation.do_eval and eval_dataset else None,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.checkpoint.load_best_model_at_end and eval_dataset is not None,
        metric_for_best_model=cfg.checkpoint.metric_for_best_model,
        greater_is_better=cfg.checkpoint.greater_is_better,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        report_to=["wandb"] if cfg.logging.use_wandb else ["tensorboard"] if cfg.logging.use_tensorboard else [],
        run_name=cfg.logging.wandb_run_name,
        seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers,
        no_cuda=(cfg.device == "cpu"),  # Force CPU if specified
        use_mps_device=(cfg.device == "mps"),  # Use MPS if specified
    )

    training_args = TrainingArguments(**training_args_kwargs)

    # Create trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_reward_metrics if cfg.evaluation.do_eval and eval_dataset else None,
        margin=cfg.technique.loss.margin,
        log_rewards=cfg.technique.logging.log_rewards,
        num_rewards_to_log=cfg.technique.logging.num_rewards_to_log,
    )

    # Train
    console.print("\n[bold green]Starting training...[/bold green]")
    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from_checkpoint)

    # Save final model
    console.print("\n[bold cyan]Saving reward model...[/bold cyan]")
    final_model_path = f"{output_dir}/final_model"
    reward_model.save_pretrained(final_model_path)
    console.print(f"[green]Reward model saved to {final_model_path}[/green]")

    # Final evaluation
    if eval_dataset:
        console.print("\n[bold cyan]Final evaluation...[/bold cyan]")
        metrics = trainer.evaluate()

        # Print metrics in a nice table
        table = Table(title="Evaluation Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

        console.print(table)

        # Highlight key metrics
        console.print(f"\n[bold green]Key Results:[/bold green]")
        console.print(f"  Ranking Accuracy: {metrics.get('eval_accuracy', 0):.2%}")
        console.print(f"  Reward Margin: {metrics.get('eval_margin_mean', 0):.4f}")

        # Check if model is performing well
        accuracy = metrics.get('eval_accuracy', 0)
        if accuracy > 0.70:
            console.print("\n[bold green]✅ Excellent! Accuracy > 70%[/bold green]")
        elif accuracy > 0.60:
            console.print("\n[bold yellow]⚠️  Good, but could be better. Try more data or longer training.[/bold yellow]")
        else:
            console.print("\n[bold red]❌ Poor performance. Check data quality or model capacity.[/bold red]")

    console.print("\n[bold green]Training complete! 🎉[/bold green]")


if __name__ == "__main__":
    # Initialize Hydra with config path
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Use absolute path to configs directory
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_PATH, job_name="train_reward_model"):
        cfg = compose(config_name="config_reward", overrides=sys.argv[1:])
        main(cfg)
