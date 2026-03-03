"""
Dataset Loading Utilities

Provides functions to load datasets from various sources:
- HuggingFace Hub
- Local files (JSON, JSONL, CSV)
- Custom datasets
"""

from typing import Optional, Union, Dict, List, Callable
from pathlib import Path
from datasets import load_dataset as hf_load_dataset, Dataset, DatasetDict
import json


def load_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, List[str], Dict[str, str]]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset from HuggingFace Hub or local files.

    Args:
        dataset_name: Name of dataset on HuggingFace Hub or file type
                     ("json", "jsonl", "csv", "parquet")
        dataset_config: Dataset configuration/subset name
        split: Which split to load ("train", "test", "validation")
        data_dir: Directory containing data files
        data_files: Path(s) to data file(s)
        cache_dir: Directory to cache downloaded datasets
        **kwargs: Additional arguments for load_dataset

    Returns:
        Dataset or DatasetDict

    Examples:
        >>> # Load from HuggingFace Hub
        >>> dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        >>>
        >>> # Load local JSON file
        >>> dataset = load_dataset("json", data_files="data.json")
        >>>
        >>> # Load multiple splits
        >>> dataset = load_dataset("json", data_files={
        ...     "train": "train.json",
        ...     "test": "test.json"
        ... })
    """
    try:
        dataset = hf_load_dataset(
            dataset_name,
            name=dataset_config,
            split=split,
            data_dir=data_dir,
            data_files=data_files,
            cache_dir=cache_dir,
            **kwargs,
        )
        return dataset
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}'. "
            f"Error: {str(e)}\n"
            f"Make sure the dataset exists on HuggingFace Hub or "
            f"provide correct data_files for local loading."
        )


def load_conversation_dataset(
    dataset_name: str,
    split: Optional[str] = "train",
    format_type: str = "auto",
    **kwargs,
) -> Dataset:
    """
    Load a conversation dataset with automatic format detection.

    Supports common conversation formats:
    - ChatML
    - ShareGPT
    - Alpaca
    - OpenAI format

    Args:
        dataset_name: Dataset name or path
        split: Which split to load
        format_type: Conversation format ("auto", "chatml", "sharegpt", "alpaca")
        **kwargs: Additional arguments for load_dataset

    Returns:
        Dataset with standardized conversation format
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)

    # Auto-detect format if needed
    if format_type == "auto":
        format_type = _detect_conversation_format(dataset[0])

    # TODO: Add format conversion logic in processors/conversation.py
    return dataset


def load_preference_dataset(
    dataset_name: str,
    split: Optional[str] = "train",
    **kwargs,
) -> Dataset:
    """
    Load a preference dataset for reward modeling or DPO.

    Expected format: Each example should have:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The dispreferred response

    Args:
        dataset_name: Dataset name or path
        split: Which split to load
        **kwargs: Additional arguments for load_dataset

    Returns:
        Dataset with preference pairs
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)

    # Validate dataset has required fields
    required_fields = {"prompt", "chosen", "rejected"}
    if not required_fields.issubset(set(dataset.column_names)):
        raise ValueError(
            f"Preference dataset must have fields: {required_fields}. "
            f"Found: {set(dataset.column_names)}"
        )

    return dataset


def _detect_conversation_format(example: Dict) -> str:
    """
    Auto-detect conversation format from a sample example.

    Args:
        example: A single example from the dataset

    Returns:
        Detected format: "chatml", "sharegpt", "alpaca", or "unknown"
    """
    if "messages" in example:
        # ChatML or OpenAI format
        return "chatml"
    elif "conversations" in example:
        # ShareGPT format
        return "sharegpt"
    elif "instruction" in example and "output" in example:
        # Alpaca format
        return "alpaca"
    else:
        return "unknown"


def create_dataset_from_list(
    data: List[Dict],
    validate: bool = True,
) -> Dataset:
    """
    Create a HuggingFace Dataset from a list of dictionaries.

    Args:
        data: List of examples (each a dictionary)
        validate: Whether to validate data format

    Returns:
        Dataset object

    Example:
        >>> data = [
        ...     {"prompt": "Hello", "response": "Hi there!"},
        ...     {"prompt": "How are you?", "response": "I'm doing well!"}
        ... ]
        >>> dataset = create_dataset_from_list(data)
    """
    if validate and len(data) > 0:
        # Check all examples have same keys
        keys = set(data[0].keys())
        for i, example in enumerate(data[1:], 1):
            if set(example.keys()) != keys:
                raise ValueError(
                    f"Example {i} has different keys than example 0. "
                    f"All examples must have the same structure."
                )

    return Dataset.from_list(data)


def split_dataset(
    dataset: Dataset,
    train_size: float = 0.9,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Split a dataset into train/test/validation sets.

    Args:
        dataset: Dataset to split
        train_size: Fraction for training (0-1)
        test_size: Fraction for testing (if None, computed from train_size and val_size)
        val_size: Fraction for validation (if None, no validation set)
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with splits

    Example:
        >>> splits = split_dataset(dataset, train_size=0.8, val_size=0.1)
        >>> # Results in: train=80%, val=10%, test=10%
    """
    if val_size is not None:
        # Three-way split
        if test_size is None:
            test_size = 1.0 - train_size - val_size

        if not abs(train_size + test_size + val_size - 1.0) < 1e-6:
            raise ValueError(
                f"train_size + test_size + val_size must sum to 1.0. "
                f"Got: {train_size + test_size + val_size}"
            )

        # First split: train vs (test + val)
        temp_split = dataset.train_test_split(
            test_size=(test_size + val_size),
            seed=seed,
        )

        # Second split: test vs val
        test_val_split = temp_split["test"].train_test_split(
            test_size=test_size / (test_size + val_size),
            seed=seed,
        )

        return DatasetDict(
            {
                "train": temp_split["train"],
                "validation": test_val_split["train"],
                "test": test_val_split["test"],
            }
        )
    else:
        # Two-way split
        if test_size is None:
            test_size = 1.0 - train_size

        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split


def apply_preprocessing(
    dataset: Dataset,
    preprocessing_fn: Callable,
    num_proc: int = 1,
    remove_columns: Optional[List[str]] = None,
    desc: Optional[str] = None,
) -> Dataset:
    """
    Apply a preprocessing function to a dataset.

    Args:
        dataset: Dataset to preprocess
        preprocessing_fn: Function that takes an example and returns processed example
        num_proc: Number of processes for parallel processing
        remove_columns: Columns to remove after preprocessing
        desc: Description for progress bar

    Returns:
        Preprocessed dataset

    Example:
        >>> def preprocess(example):
        ...     example["text"] = example["text"].lower()
        ...     return example
        >>> dataset = apply_preprocessing(dataset, preprocess, num_proc=4)
    """
    processed = dataset.map(
        preprocessing_fn,
        num_proc=num_proc,
        remove_columns=remove_columns,
        desc=desc or "Preprocessing",
    )
    return processed
