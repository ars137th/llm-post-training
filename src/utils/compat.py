"""
Compatibility utilities for handling different library versions.

This module provides version-aware utilities to handle API differences
across transformers, PyTorch, and other library versions.
"""

from packaging import version
import transformers
import torch
from typing import Dict, Any
import inspect


def get_transformers_version() -> version.Version:
    """Get the installed transformers version."""
    return version.parse(transformers.__version__)


def get_torch_version() -> version.Version:
    """Get the installed PyTorch version."""
    return version.parse(torch.__version__)


# Version checks
TRANSFORMERS_VERSION = get_transformers_version()
TORCH_VERSION = get_torch_version()

# API change thresholds
TRANSFORMERS_4_36 = version.parse("4.36.0")


def get_training_args_kwargs(
    output_dir: str,
    eval_enabled: bool = True,
    logging_dir: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get TrainingArguments kwargs compatible with installed transformers version.

    Handles API changes:
    - transformers <4.36: evaluation_strategy, logging_dir
    - transformers >=4.36: eval_strategy, TENSORBOARD_LOGGING_DIR env var

    Args:
        output_dir: Output directory
        eval_enabled: Whether evaluation is enabled
        logging_dir: Logging directory (for tensorboard)
        **kwargs: Other training arguments

    Returns:
        Dict of arguments compatible with current transformers version
    """
    training_kwargs = kwargs.copy()

    # Handle evaluation_strategy vs eval_strategy
    if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
        # New API (4.36+)
        training_kwargs['eval_strategy'] = "steps" if eval_enabled else "no"
        # Don't include logging_dir (deprecated, set via env var instead)
    else:
        # Old API (<4.36)
        training_kwargs['evaluation_strategy'] = "steps" if eval_enabled else "no"
        # Include logging_dir
        if logging_dir:
            training_kwargs['logging_dir'] = logging_dir

    return training_kwargs


def should_pass_tokenizer_to_trainer() -> bool:
    """
    Check if Trainer.__init__() accepts tokenizer parameter.

    Returns:
        True if tokenizer should be passed, False otherwise
    """
    # transformers <4.36 accepts tokenizer, >=4.36 does not
    return TRANSFORMERS_VERSION < TRANSFORMERS_4_36


def get_trainer_init_kwargs(
    model,
    args,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    compute_metrics=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get Trainer.__init__() kwargs compatible with installed transformers version.

    Handles API changes:
    - transformers <4.36: accepts tokenizer parameter
    - transformers >=4.36: does not accept tokenizer parameter

    Args:
        model: Model
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Eval dataset
        tokenizer: Tokenizer (may or may not be passed to Trainer)
        data_collator: Data collator
        compute_metrics: Metrics function
        **kwargs: Other trainer arguments

    Returns:
        Tuple of (trainer_kwargs, tokenizer_to_store)
        - trainer_kwargs: Dict to pass to Trainer.__init__()
        - tokenizer_to_store: Tokenizer to store separately (if not passed to parent)
    """
    trainer_kwargs = {
        'model': model,
        'args': args,
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator,
        'compute_metrics': compute_metrics,
        **kwargs
    }

    tokenizer_to_store = None

    if should_pass_tokenizer_to_trainer():
        # Old API: pass tokenizer to Trainer
        trainer_kwargs['tokenizer'] = tokenizer
    else:
        # New API: don't pass tokenizer, caller should store it separately
        tokenizer_to_store = tokenizer

    return trainer_kwargs, tokenizer_to_store


def training_step_accepts_num_items(trainer_class) -> bool:
    """
    Check if training_step() method accepts num_items_in_batch parameter.

    Args:
        trainer_class: The Trainer class or instance

    Returns:
        True if num_items_in_batch parameter is accepted
    """
    # transformers <4.36: training_step(model, inputs)
    # transformers >=4.36: training_step(model, inputs, num_items_in_batch)

    # Use version check as it's more reliable
    return TRANSFORMERS_VERSION >= TRANSFORMERS_4_36


def get_version_info() -> Dict[str, str]:
    """
    Get version information for all relevant libraries.

    Returns:
        Dict with library versions
    """
    return {
        'transformers': str(TRANSFORMERS_VERSION),
        'torch': str(TORCH_VERSION),
        'python': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        'platform': __import__('platform').system(),
    }


def print_version_info():
    """Print version information for debugging."""
    info = get_version_info()
    print("=" * 60)
    print("Library Versions")
    print("=" * 60)
    for lib, ver in info.items():
        print(f"{lib:15s}: {ver}")
    print("=" * 60)

    # Print API compatibility info
    print("\nAPI Compatibility:")
    if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
        print("  - Using transformers 4.36+ API")
        print("    * eval_strategy (not evaluation_strategy)")
        print("    * No tokenizer in Trainer.__init__()")
        print("    * training_step() with num_items_in_batch")
        print("    * TENSORBOARD_LOGGING_DIR env var (not logging_dir)")
    else:
        print("  - Using transformers <4.36 API")
        print("    * evaluation_strategy (not eval_strategy)")
        print("    * tokenizer in Trainer.__init__()")
        print("    * training_step() without num_items_in_batch")
        print("    * logging_dir parameter")
    print("=" * 60)
