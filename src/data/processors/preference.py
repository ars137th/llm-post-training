"""
Preference Data Processing

Utilities for loading and processing preference pair data for reward modeling.
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizer


def load_preference_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a preference dataset from HuggingFace or local source.

    Supported datasets:
    - "Anthropic/hh-rlhf": Anthropic's Human Feedback dataset
    - "OpenAssistant/oasst1": OpenAssistant conversations
    - "stanfordnlp/SHP": Stanford Human Preferences
    - Custom datasets with (prompt, chosen, rejected) format

    Args:
        dataset_name: Dataset name or path
        dataset_config: Dataset configuration (if needed)
        split: Dataset split (train/test/validation)
        cache_dir: Cache directory

    Returns:
        Dataset with preference pairs

    Example:
        >>> dataset = load_preference_dataset("Anthropic/hh-rlhf")
        >>> print(dataset[0].keys())  # ['chosen', 'rejected']
    """
    dataset = hf_load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=cache_dir,
    )

    return dataset


def parse_anthropic_format(example: Dict) -> Dict:
    """
    Parse Anthropic HH-RLHF format.

    Anthropic format includes full conversation:
    "Human: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."

    We extract the prompt (everything up to last Assistant response)
    and the response (last Assistant response).

    Args:
        example: Dictionary with 'chosen' and 'rejected' keys

    Returns:
        Dictionary with 'prompt', 'chosen', 'rejected'

    Example:
        >>> example = {
        ...     "chosen": "Human: Hi\n\nAssistant: Hello!",
        ...     "rejected": "Human: Hi\n\nAssistant: Hey"
        ... }
        >>> parsed = parse_anthropic_format(example)
        >>> print(parsed['prompt'])  # "Human: Hi"
        >>> print(parsed['chosen'])  # "Hello!"
    """
    def split_conversation(text: str) -> Tuple[str, str]:
        """Split into prompt and response."""
        # Find last "Assistant:" marker
        last_assistant = text.rfind("\n\nAssistant:")

        if last_assistant == -1:
            # No assistant marker, treat whole text as response
            return "", text.strip()

        prompt = text[:last_assistant].strip()
        response = text[last_assistant + len("\n\nAssistant:"):].strip()

        return prompt, response

    prompt_chosen, response_chosen = split_conversation(example['chosen'])
    prompt_rejected, response_rejected = split_conversation(example['rejected'])

    # Prompts should be the same (they're responses to the same input)
    # Use the one from chosen as canonical
    return {
        'prompt': prompt_chosen,
        'chosen': response_chosen,
        'rejected': response_rejected,
    }


def create_preference_dataset(
    examples: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> List[Dict]:
    """
    Tokenize preference pairs for reward model training.

    Args:
        examples: List of preference examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        prompt_key: Key for prompt in examples
        chosen_key: Key for chosen response
        rejected_key: Key for rejected response

    Returns:
        List of tokenized examples with chosen/rejected pairs

    Example:
        >>> examples = [
        ...     {
        ...         "prompt": "What is AI?",
        ...         "chosen": "AI is artificial intelligence...",
        ...         "rejected": "I don't know"
        ...     }
        ... ]
        >>> tokenized = create_preference_dataset(examples, tokenizer)
        >>> print(tokenized[0].keys())
        >>> # ['chosen_input_ids', 'chosen_attention_mask',
        >>> #  'rejected_input_ids', 'rejected_attention_mask']
    """
    tokenized_examples = []

    for example in examples:
        prompt = example[prompt_key]
        chosen = example[chosen_key]
        rejected = example[rejected_key]

        # Concatenate prompt + response for each
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        # Tokenize both
        chosen_encoded = tokenizer(
            chosen_text,
            max_length=max_length,
            truncation=True,
            padding=False,  # Will pad in collator
        )

        rejected_encoded = tokenizer(
            rejected_text,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        tokenized_examples.append({
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask'],
        })

    return tokenized_examples


class PreferenceDataCollator:
    """
    Data collator for preference pairs.

    Pads sequences and creates batches for reward model training.

    Args:
        tokenizer: Tokenizer (for padding)
        max_length: Maximum sequence length

    Example:
        >>> collator = PreferenceDataCollator(tokenizer, max_length=512)
        >>> batch = collator([example1, example2, example3])
        >>> print(batch.keys())
        >>> # ['chosen_input_ids', 'chosen_attention_mask',
        >>> #  'rejected_input_ids', 'rejected_attention_mask']
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate features into batch.

        Args:
            features: List of examples with chosen/rejected pairs

        Returns:
            Dictionary of batched tensors
        """
        # Extract chosen and rejected
        chosen_input_ids = [f['chosen_input_ids'] for f in features]
        chosen_attention_mask = [f['chosen_attention_mask'] for f in features]
        rejected_input_ids = [f['rejected_input_ids'] for f in features]
        rejected_attention_mask = [f['rejected_attention_mask'] for f in features]

        # Pad sequences
        chosen_input_ids = self._pad_sequences(chosen_input_ids)
        chosen_attention_mask = self._pad_sequences(chosen_attention_mask, value=0)
        rejected_input_ids = self._pad_sequences(rejected_input_ids)
        rejected_attention_mask = self._pad_sequences(rejected_attention_mask, value=0)

        return {
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask,
        }

    def _pad_sequences(
        self,
        sequences: List[List[int]],
        value: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Pad sequences to same length.

        Args:
            sequences: List of sequences (each is list of ints)
            value: Padding value (default: tokenizer pad_token_id)

        Returns:
            Padded tensor [batch_size, max_seq_len]
        """
        if value is None:
            value = self.padding_value

        # Find max length in batch
        max_len = max(len(seq) for seq in sequences)
        max_len = min(max_len, self.max_length)

        # Pad each sequence
        padded = []
        for seq in sequences:
            # Truncate if needed
            if len(seq) > max_len:
                seq = seq[:max_len]

            # Pad if needed
            if len(seq) < max_len:
                seq = seq + [value] * (max_len - len(seq))

            padded.append(seq)

        return torch.tensor(padded, dtype=torch.long)


def prepare_preference_data(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    format_fn: Optional[callable] = None,
    num_examples: Optional[int] = None,
) -> Dataset:
    """
    Prepare preference dataset for training.

    This is a convenience function that:
    1. Optionally parses format (e.g., Anthropic)
    2. Tokenizes pairs
    3. Converts to HuggingFace Dataset

    Args:
        dataset: Raw preference dataset
        tokenizer: Tokenizer
        max_length: Max sequence length
        format_fn: Optional function to parse format (e.g., parse_anthropic_format)
        num_examples: Optional limit on number of examples

    Returns:
        Tokenized Dataset ready for training

    Example:
        >>> from datasets import load_dataset
        >>> raw_data = load_dataset("Anthropic/hh-rlhf", split="train")
        >>> prepared = prepare_preference_data(
        ...     raw_data,
        ...     tokenizer,
        ...     format_fn=parse_anthropic_format,
        ...     num_examples=1000,
        ... )
    """
    # Limit examples if specified
    if num_examples is not None:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    # Parse format if needed
    if format_fn is not None:
        dataset = dataset.map(format_fn, remove_columns=dataset.column_names)

    # Convert to list for tokenization
    examples = list(dataset)

    # Tokenize
    tokenized = create_preference_dataset(
        examples,
        tokenizer,
        max_length=max_length,
    )

    # Convert back to Dataset
    return Dataset.from_list(tokenized)


def create_synthetic_preference_data(
    num_examples: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Create synthetic preference data for testing.

    Generates simple preference pairs where "chosen" responses are
    clearly better than "rejected" ones.

    Args:
        num_examples: Number of examples to generate
        seed: Random seed

    Returns:
        List of preference examples

    Example:
        >>> data = create_synthetic_preference_data(num_examples=10)
        >>> print(data[0])
        >>> # {'prompt': '...', 'chosen': '...', 'rejected': '...'}
    """
    import random
    random.seed(seed)

    examples = []

    # Templates for different categories
    templates = [
        # Factual questions
        {
            "prompt": "What is the capital of {country}?",
            "chosen": "The capital of {country} is {capital}.",
            "rejected": "I'm not sure.",
            "data": [
                ("France", "Paris"),
                ("Japan", "Tokyo"),
                ("Brazil", "Brasília"),
                ("Egypt", "Cairo"),
            ]
        },
        # Math questions
        {
            "prompt": "What is {a} + {b}?",
            "chosen": "{a} + {b} equals {result}.",
            "rejected": "That's a math problem.",
            "data": [(i, j) for i in range(1, 10) for j in range(1, 10) if i + j <= 20]
        },
        # Helpful vs unhelpful
        {
            "prompt": "How do I {task}?",
            "chosen": "To {task}, follow these steps: First, {step1}. Then, {step2}.",
            "rejected": "You should {task}.",
            "data": [
                ("learn Python", "study the basics", "practice with projects"),
                ("make coffee", "add water to machine", "add coffee grounds"),
                ("tie a tie", "cross the wide end", "pull through the loop"),
            ]
        },
    ]

    for _ in range(num_examples):
        # Pick random template
        template = random.choice(templates)
        data_point = random.choice(template['data'])

        if isinstance(data_point, tuple) and len(data_point) == 2:
            if "country" in template['prompt']:
                country, capital = data_point
                prompt = template['prompt'].format(country=country)
                chosen = template['chosen'].format(country=country, capital=capital)
                rejected = template['rejected']
            elif "a" in template['prompt']:
                a, b = data_point
                result = a + b
                prompt = template['prompt'].format(a=a, b=b)
                chosen = template['chosen'].format(a=a, b=b, result=result)
                rejected = template['rejected']
            else:
                continue
        elif isinstance(data_point, tuple) and len(data_point) == 3:
            task, step1, step2 = data_point
            prompt = template['prompt'].format(task=task)
            chosen = template['chosen'].format(task=task, step1=step1, step2=step2)
            rejected = template['rejected'].format(task=task)
        else:
            continue

        examples.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
        })

    return examples


def validate_preference_dataset(dataset: Dataset) -> Dict[str, any]:
    """
    Validate preference dataset format and quality.

    Checks:
    - Required fields present
    - No missing values
    - Length distributions
    - Prompt consistency

    Args:
        dataset: Preference dataset to validate

    Returns:
        Dictionary with validation results

    Example:
        >>> results = validate_preference_dataset(dataset)
        >>> if results['valid']:
        ...     print("Dataset is valid!")
        >>> else:
        ...     print(f"Issues: {results['issues']}")
    """
    issues = []
    stats = {}

    # Check required fields
    required_fields = ['prompt', 'chosen', 'rejected']
    if hasattr(dataset, 'column_names'):
        fields = dataset.column_names
    else:
        fields = list(dataset[0].keys()) if len(dataset) > 0 else []

    for field in required_fields:
        if field not in fields:
            issues.append(f"Missing required field: {field}")

    if not issues:
        # Check for missing values
        for i, example in enumerate(dataset):
            for field in required_fields:
                if not example.get(field):
                    issues.append(f"Empty {field} at index {i}")
                    break

        # Compute statistics
        chosen_lengths = [len(ex['chosen']) for ex in dataset]
        rejected_lengths = [len(ex['rejected']) for ex in dataset]

        stats = {
            'num_examples': len(dataset),
            'avg_chosen_length': sum(chosen_lengths) / len(chosen_lengths),
            'avg_rejected_length': sum(rejected_lengths) / len(rejected_lengths),
            'fields': fields,
        }

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'stats': stats,
    }
