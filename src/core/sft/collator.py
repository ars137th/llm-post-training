"""
Data Collator for Supervised Fine-Tuning

Handles batching and padding of sequences for SFT training.
Properly masks prompt tokens so loss is only computed on responses.
"""

from typing import Dict, List, Any, Optional
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForSFT:
    """
    Data collator for supervised fine-tuning.

    Handles:
    - Padding sequences to the same length within a batch
    - Creating attention masks
    - Masking prompt tokens in labels (loss only on response)
    - Proper handling of special tokens

    Example:
        >>> collator = DataCollatorForSFT(tokenizer, max_length=512)
        >>> batch = collator([example1, example2, example3])
    """

    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    mask_prompt: bool = True

    def __post_init__(self):
        """Ensure tokenizer has pad token."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            features: List of examples, each with 'input_ids', 'attention_mask', 'labels'

        Returns:
            Dictionary with batched and padded tensors
        """
        # Extract keys from first example
        if not features:
            return {}

        # Get max length in batch
        if self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = max(len(f['input_ids']) for f in features)

        # Optionally pad to multiple of N (for tensor cores)
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Prepare batch
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }

        for feature in features:
            # Get input_ids and labels
            input_ids = feature['input_ids']
            labels = feature.get('labels', input_ids.copy())

            # Truncate if needed
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

            # Calculate padding length
            pad_len = max_len - len(input_ids)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(input_ids) + [0] * pad_len

            # Pad input_ids
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

            # Pad labels with -100 (ignore in loss)
            labels = labels + [-100] * pad_len

            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)

        # Convert to tensors
        batch = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }

        return batch


@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Data collator that only computes loss on the completion (response).

    This is useful when you have a clear separation between prompt and response,
    and you want to ensure loss is ONLY computed on the response tokens.

    Example format:
        "### Instruction: What is AI?\n### Response: AI is..."
                                              ^^^^^^^^^^^^ Only compute loss here
    """

    tokenizer: PreTrainedTokenizer
    response_template: str = "### Response:"
    instruction_template: str = "### Instruction:"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        """Tokenize templates for efficient matching."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Tokenize response template to find where it appears
        self.response_template_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and mask everything except response.

        Args:
            features: List of examples with 'input_ids' and optionally 'labels'

        Returns:
            Batched tensors with properly masked labels
        """
        # Determine max length
        if self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = max(len(f['input_ids']) for f in features)

        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }

        for feature in features:
            input_ids = feature['input_ids']

            # Find response start position
            response_start = self._find_response_start(input_ids)

            # Create labels: -100 for prompt, actual tokens for response
            labels = [-100] * len(input_ids)
            if response_start is not None:
                # Copy response tokens to labels
                labels[response_start:] = input_ids[response_start:]

            # Truncate if needed
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

            # Padding
            pad_len = max_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)

        # Convert to tensors
        batch = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }

        return batch

    def _find_response_start(self, input_ids: List[int]) -> Optional[int]:
        """
        Find the start position of the response in the sequence.

        Args:
            input_ids: List of token IDs

        Returns:
            Index where response starts, or None if not found
        """
        # Search for response template
        template_len = len(self.response_template_ids)

        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i:i + template_len] == self.response_template_ids:
                # Response starts after the template
                return i + template_len

        # If template not found, return None (will compute loss on everything)
        return None


def create_sft_dataset(
    examples: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    prompt_column: str = "prompt",
    response_column: str = "response",
    max_length: int = 512,
    prompt_template: Optional[str] = None,
) -> List[Dict[str, List[int]]]:
    """
    Convert raw examples into tokenized format for SFT.

    Args:
        examples: List of dicts with prompt and response
        tokenizer: Tokenizer to use
        prompt_column: Name of prompt column
        response_column: Name of response column
        max_length: Maximum sequence length
        prompt_template: Optional template string (e.g., "### Instruction: {prompt}\n### Response: ")

    Returns:
        List of tokenized examples ready for DataCollator

    Example:
        >>> examples = [
        ...     {"prompt": "What is AI?", "response": "AI is..."},
        ...     {"prompt": "Hello", "response": "Hi there!"}
        ... ]
        >>> dataset = create_sft_dataset(examples, tokenizer)
    """
    tokenized = []

    for example in examples:
        prompt = example[prompt_column]
        response = example[response_column]

        # Apply template if provided
        if prompt_template:
            full_text = prompt_template.format(prompt=prompt) + response
        else:
            full_text = prompt + response

        # Tokenize
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )

        # Tokenize prompt separately to know where to mask
        if prompt_template:
            prompt_text = prompt_template.format(prompt=prompt)
        else:
            prompt_text = prompt

        prompt_encoded = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )
        prompt_len = len(prompt_encoded['input_ids'])

        # Create labels: -100 for prompt, actual tokens for response
        labels = [-100] * prompt_len + encoded['input_ids'][prompt_len:]
        # Make sure labels same length as input_ids
        labels = labels[:len(encoded['input_ids'])]

        tokenized.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        })

    return tokenized
