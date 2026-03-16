"""
Training Script for PPO/RLHF

Run PPO training with configuration management via Hydra.

Usage:
    # Basic training with defaults
    python scripts/train/train_ppo.py

    # Override config
    python scripts/train/train_ppo.py model=gpt2 data.num_prompts=100

    # Custom hyperparameters
    python scripts/train/train_ppo.py training.kl_coef=0.1 training.learning_rate=1e-6

    # Full experiment config
    python scripts/train/train_ppo.py experiment=ppo_gpt2_synthetic

    # Test on CPU with minimal data
    python scripts/train/train_ppo.py device=cpu data.use_synthetic=true data.num_prompts=10 training.num_rollouts=2
"""

import os
import sys
from pathlib import Path

# Force CPU if MPS has issues (Apple Silicon compatibility)
if os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

# Add src to path and set up project root for Hydra
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Hydra needs absolute path to configs
CONFIGS_PATH = str(project_root / "configs")

import torch
from transformers import set_seed
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from src.models.language import LanguageModel
from src.models.reward import RewardModel
from src.core.ppo import PPOConfig, PPOTrainer
from src.data.processors.preference import create_synthetic_preference_data


console = Console()


def print_config(cfg: DictConfig):
    """Pretty print configuration."""
    console.print("\n[bold cyan]Configuration:[/bold cyan]")
    console.print(OmegaConf.to_yaml(cfg))


def print_model_info(model, model_name: str = "Model"):
    """Print model information in a table."""
    table = Table(title=f"{model_name} Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Handle both LanguageModel and RewardModel
    if hasattr(model, 'model'):
        # LanguageModel wrapper
        inner_model = model.model
        config_name = inner_model.config._name_or_path if hasattr(inner_model.config, '_name_or_path') else "unknown"
        num_params = model.num_parameters
        num_trainable = model.num_trainable_parameters
        is_peft = model.is_peft_model
        device = model.device
    elif hasattr(model, 'base_model'):
        # RewardModel
        config_name = model.base_model.model.config._name_or_path if hasattr(model.base_model.model.config, '_name_or_path') else "unknown"
        num_params = model.num_parameters
        num_trainable = model.num_trainable_parameters
        is_peft = False
        device = model.device
    else:
        # Raw PreTrainedModel
        config_name = model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown"
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        is_peft = False
        device = next(model.parameters()).device

    percent_trainable = (num_trainable / num_params * 100) if num_params > 0 else 0

    table.add_row("Base Model", config_name)
    table.add_row("Total Parameters", f"{num_params:,}")
    table.add_row("Trainable Parameters", f"{num_trainable:,}")
    table.add_row("Trainable %", f"{percent_trainable:.2f}%")
    table.add_row("Device", str(device))
    table.add_row("Is PEFT Model", str(is_peft))

    console.print(table)


def load_prompts(cfg: DictConfig):
    """
    Load prompts for PPO training.

    Args:
        cfg: Configuration

    Returns:
        List of prompt strings
    """
    console.print("\n[bold cyan]Loading prompts...[/bold cyan]")

    if cfg.data.use_synthetic:
        # Generate synthetic prompts
        console.print("[yellow]Generating synthetic prompts...[/yellow]")

        # Create synthetic preference data (we only need prompts)
        examples = create_synthetic_preference_data(
            num_examples=cfg.data.num_prompts,
            seed=cfg.seed,
        )

        prompts = [ex['prompt'] for ex in examples]
        console.print(f"[green]Generated {len(prompts)} synthetic prompts[/green]")

    else:
        # Load real prompts from file or dataset
        console.print(f"[yellow]Loading prompts from {cfg.data.prompt_file}...[/yellow]")

        # Read prompts from file (one per line)
        with open(cfg.data.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        # Limit if specified
        if cfg.data.num_prompts:
            prompts = prompts[:cfg.data.num_prompts]

        console.print(f"[green]Loaded {len(prompts)} prompts[/green]")

    # Show some examples
    console.print("\n[bold]Example Prompts:[/bold]")
    for i, prompt in enumerate(prompts[:3]):
        console.print(f"  [{i+1}] {prompt}")
    if len(prompts) > 3:
        console.print(f"  ... and {len(prompts) - 3} more")

    return prompts


def main(cfg: DictConfig):
    """Main training function."""
    console.print("[bold green]Starting PPO/RLHF Training[/bold green]")
    print_config(cfg)

    # Set seed
    set_seed(cfg.seed)

    # Load prompts
    prompts = load_prompts(cfg)

    # Load actor (policy model) - trainable
    console.print("\n[bold cyan]Loading actor (policy model)...[/bold cyan]")
    actor = LanguageModel.from_pretrained(
        model_name=cfg.model.name,
        use_lora=cfg.model.use_lora,
        lora_config=OmegaConf.to_container(cfg.model.lora_config, resolve=True) if cfg.model.use_lora else None,
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        device=cfg.device if cfg.device != "auto" else None,
        trust_remote_code=True,
    )
    print_model_info(actor, "Actor (Policy)")

    # Load critic (value function) - trainable
    console.print("\n[bold cyan]Loading critic (value function)...[/bold cyan]")
    # Critic uses same base but with value head
    critic_base = LanguageModel.from_pretrained(
        model_name=cfg.model.name,
        use_lora=cfg.model.use_lora,
        lora_config=OmegaConf.to_container(cfg.model.lora_config, resolve=True) if cfg.model.use_lora else None,
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        device=cfg.device if cfg.device != "auto" else None,
        trust_remote_code=True,
    )
    critic = RewardModel(
        base_model=critic_base,
        freeze_base=False,  # Train base + value head
    )
    print_model_info(critic, "Critic (Value Function)")

    # Load reference model - frozen
    console.print("\n[bold cyan]Loading reference model (frozen)...[/bold cyan]")
    reference = LanguageModel.from_pretrained(
        model_name=cfg.model.name,
        use_lora=False,  # No LoRA for reference
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        device=cfg.device if cfg.device != "auto" else None,
        trust_remote_code=True,
    )
    print_model_info(reference, "Reference Model")

    # Load reward model - frozen
    console.print("\n[bold cyan]Loading reward model (frozen)...[/bold cyan]")
    if cfg.reward_model.from_checkpoint:
        # Load pre-trained reward model
        console.print(f"[yellow]Loading from {cfg.reward_model.checkpoint_path}...[/yellow]")
        reward_model = RewardModel.from_pretrained(cfg.reward_model.checkpoint_path)
    else:
        # Create reward model from same base (for testing)
        console.print("[yellow]Creating reward model from base (testing mode)...[/yellow]")
        rm_base = LanguageModel.from_pretrained(
            model_name=cfg.model.name,
            use_lora=False,
            use_4bit=cfg.model.get('use_4bit', False),
            use_8bit=cfg.model.get('use_8bit', False),
            device=cfg.device if cfg.device != "auto" else None,
            trust_remote_code=True,
        )
        reward_model = RewardModel(
            base_model=rm_base,
            freeze_base=True,  # Freeze for reward model
        )
    print_model_info(reward_model, "Reward Model")

    # Create PPO config
    console.print("\n[bold cyan]Setting up PPO configuration...[/bold cyan]")
    ppo_config = PPOConfig(
        # Rollout
        batch_size=cfg.training.batch_size,
        max_prompt_length=cfg.data.max_prompt_length,
        max_response_length=cfg.data.max_response_length,

        # Generation
        temperature=cfg.training.temperature,
        top_p=cfg.training.top_p,
        top_k=cfg.training.top_k,

        # PPO hyperparameters
        ppo_epochs=cfg.training.ppo_epochs,
        mini_batch_size=cfg.training.mini_batch_size,
        clip_range=cfg.training.clip_range,
        clip_range_vf=cfg.training.clip_range_vf if cfg.training.get('clip_range_vf') else None,

        # GAE
        gamma=cfg.training.gamma,
        lam=cfg.training.lam,

        # Loss coefficients
        vf_coef=cfg.training.vf_coef,
        ent_coef=cfg.training.ent_coef,
        kl_coef=cfg.training.kl_coef,

        # Optimization
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,

        # Training
        num_rollouts=cfg.training.num_rollouts,
        log_every=cfg.training.log_every,
        save_every=cfg.training.save_every,

        # Adaptive KL
        use_adaptive_kl=cfg.training.use_adaptive_kl,
        target_kl=cfg.training.target_kl,

        # Normalization
        normalize_advantages=cfg.training.normalize_advantages,
        reward_clip=cfg.training.reward_clip if cfg.training.get('reward_clip') else None,

        # Device
        device=cfg.device,
    )

    # Print config
    table = Table(title="PPO Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Num Rollouts", str(ppo_config.num_rollouts))
    table.add_row("Batch Size", str(ppo_config.batch_size))
    table.add_row("PPO Epochs", str(ppo_config.ppo_epochs))
    table.add_row("Mini Batch Size", str(ppo_config.mini_batch_size))
    table.add_row("Learning Rate", f"{ppo_config.learning_rate:.2e}")
    table.add_row("Clip Range (ε)", f"{ppo_config.clip_range}")
    table.add_row("GAE Lambda (λ)", f"{ppo_config.lam}")
    table.add_row("KL Coefficient (β)", f"{ppo_config.kl_coef}")
    table.add_row("Value Coef (c₁)", f"{ppo_config.vf_coef}")
    table.add_row("Entropy Coef (c₂)", f"{ppo_config.ent_coef}")

    console.print(table)

    # Create output directory
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create trainer
    console.print("\n[bold cyan]Creating PPO Trainer...[/bold cyan]")
    trainer = PPOTrainer(
        actor=actor.model,  # Pass inner model
        critic=critic,  # Pass RewardModel wrapper (has value head)
        reference=reference.model,  # Pass inner model
        reward_model=reward_model,  # Pass RewardModel wrapper
        tokenizer=actor.tokenizer,
        config=ppo_config,
    )

    console.print("[green]✅ Trainer created successfully[/green]")

    # Train
    console.print("\n[bold green]Starting PPO training...[/bold green]")
    console.print(f"[yellow]This will take a while (~{cfg.training.num_rollouts * 2} minutes on CPU)[/yellow]")

    training_metrics = trainer.train(prompts)

    # Save models
    console.print("\n[bold cyan]Saving models...[/bold cyan]")
    trainer.save(output_dir)
    console.print(f"[green]✅ Models saved to {output_dir}/[/green]")

    # Print final metrics
    console.print("\n[bold cyan]Training Summary:[/bold cyan]")

    final_metrics_table = Table(title="Final Metrics")
    final_metrics_table.add_column("Metric", style="cyan")
    final_metrics_table.add_column("Value", style="green")

    if training_metrics['reward']:
        final_metrics_table.add_row("Final Reward", f"{training_metrics['reward'][-1]:.4f}")
    if training_metrics['loss']:
        final_metrics_table.add_row("Final Loss", f"{training_metrics['loss'][-1]:.4f}")
    if training_metrics['kl']:
        final_metrics_table.add_row("Final KL", f"{training_metrics['kl'][-1]:.6f}")
    if training_metrics['clip_fraction']:
        final_metrics_table.add_row("Final Clip Fraction", f"{training_metrics['clip_fraction'][-1]:.2%}")

    console.print(final_metrics_table)

    # Performance assessment
    console.print("\n[bold]Performance Assessment:[/bold]")

    if training_metrics['reward']:
        reward_improvement = training_metrics['reward'][-1] - training_metrics['reward'][0]
        if reward_improvement > 0.1:
            console.print("[green]✅ Great! Reward improved significantly (+{:.4f})[/green]".format(reward_improvement))
        elif reward_improvement > 0:
            console.print("[yellow]⚠️  Reward improved slightly (+{:.4f})[/yellow]".format(reward_improvement))
        else:
            console.print("[red]❌ Reward did not improve ({:.4f})[/red]".format(reward_improvement))

    if training_metrics['kl']:
        final_kl = training_metrics['kl'][-1]
        if final_kl < 0.01:
            console.print(f"[green]✅ KL divergence is healthy ({final_kl:.6f} < 0.01)[/green]")
        elif final_kl < 0.1:
            console.print(f"[yellow]⚠️  KL divergence is elevated ({final_kl:.6f})[/yellow]")
        else:
            console.print(f"[red]❌ KL divergence is too high ({final_kl:.6f})[/red]")

    console.print("\n[bold green]PPO training complete! 🎉[/bold green]")
    console.print(f"[cyan]Models saved to: {output_dir}/[/cyan]")
    console.print(f"[cyan]  - Actor: {output_dir}/actor/[/cyan]")
    console.print(f"[cyan]  - Critic: {output_dir}/critic/[/cyan]")


if __name__ == "__main__":
    # Initialize Hydra with config path
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Use absolute path to configs directory
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_PATH, job_name="train_ppo"):
        cfg = compose(config_name="config_ppo", overrides=sys.argv[1:])
        main(cfg)
