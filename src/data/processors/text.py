"""
Text Processing Utilities

Handles tokenization, preprocessing, and formatting for text-only data.
"""

from typing import Dict, List, Optional, Union, Callable
import torch
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Container for processed text data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None


class TextProcessor:
    """
    Text processing and tokenization utilities.

    Handles:
    - Tokenization with padding and truncation
    - Creating labels for causal LM (with prompt masking)
    - Batch processing
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: Union[str, bool] = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize text processor.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy ("max_length", "longest", or False)
            truncation: Whether to truncate long sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenize(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            return_tensors: Return type ("pt" for PyTorch tensors)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        return encoded

    def create_causal_lm_labels(
        self,
        input_ids: torch.Tensor,
        prompt_length: Optional[int] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create labels for causal language modeling.

        For instruction fine-tuning, we typically only want to compute loss
        on the response tokens, not the prompt tokens. This function creates
        labels where prompt tokens are masked with -100.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            prompt_length: Length of prompt to mask (if same for all examples)
            prompt_mask: Binary mask [batch_size, seq_len] where 1=compute loss, 0=ignore
                        Takes precedence over prompt_length

        Returns:
            Labels tensor [batch_size, seq_len] with -100 for masked positions
        """
        labels = input_ids.clone()

        if prompt_mask is not None:
            # Use provided mask
            labels[prompt_mask == 0] = -100
        elif prompt_length is not None:
            # Mask first prompt_length tokens
            labels[:, :prompt_length] = -100
        # Otherwise, compute loss on all tokens

        return labels

    def process_for_sft(
        self,
        prompt: Union[str, List[str]],
        response: Union[str, List[str]],
        mask_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process prompt-response pairs for supervised fine-tuning.

        Args:
            prompt: Prompt text(s)
            response: Response text(s)
            mask_prompt: Whether to mask prompt in loss computation

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Handle single example vs batch
        is_batch = isinstance(prompt, list)
        if not is_batch:
            prompt = [prompt]
            response = [response]

        # Combine prompt and response
        full_texts = [p + r for p, r in zip(prompt, response)]

        # Tokenize full text
        encoded = self.tokenize(full_texts)

        # Create labels
        labels = encoded["input_ids"].clone()

        if mask_prompt:
            # Tokenize prompts separately to get their lengths
            prompt_encoded = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )

            # Mask prompt tokens in labels
            for i, prompt_ids in enumerate(prompt_encoded["input_ids"]):
                prompt_len = len(prompt_ids)
                labels[i, :prompt_len] = -100

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

        # Return single example if input was single
        if not is_batch:
            result = {k: v[0] for k, v in result.items()}

        return result

    def batch_encode(
        self,
        texts: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts.

        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments for tokenize

        Returns:
            Dictionary with batched input_ids and attention_mask
        """
        return self.tokenize(texts, **kwargs)

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens (BOS, EOS, PAD)
            clean_up_tokenization_spaces: Whether to clean up tokenization artifacts

        Returns:
            Decoded text(s)
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode multiple sequences.

        Args:
            token_ids: Batch of token IDs [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces

        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


def create_prompt_template(
    template_type: str = "alpaca",
    system_message: Optional[str] = None,
) -> Callable[[str, Optional[str]], str]:
    """
    Create a prompt formatting function for different template types.

    Args:
        template_type: Type of template ("alpaca", "chatml", "llama2", "plain")
        system_message: Optional system message to prepend

    Returns:
        Function that takes (instruction, input_text) and returns formatted prompt

    Example:
        >>> format_prompt = create_prompt_template("alpaca")
        >>> prompt = format_prompt("Summarize the following:", "Long text...")
    """
    if template_type == "alpaca":

        def format_fn(instruction: str, input_text: Optional[str] = None) -> str:
            if input_text:
                return (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n"
                )
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n"

    elif template_type == "chatml":

        def format_fn(instruction: str, input_text: Optional[str] = None) -> str:
            messages = []
            if system_message:
                messages.append(f"<|im_start|>system\n{system_message}<|im_end|>")

            user_content = instruction
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"

            messages.append(f"<|im_start|>user\n{user_content}<|im_end|>")
            messages.append("<|im_start|>assistant\n")

            return "\n".join(messages)

    elif template_type == "llama2":

        def format_fn(instruction: str, input_text: Optional[str] = None) -> str:
            sys_msg = system_message or "You are a helpful assistant."
            user_content = instruction
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"

            return (
                f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n"
                f"{user_content} [/INST] "
            )

    elif template_type == "plain":

        def format_fn(instruction: str, input_text: Optional[str] = None) -> str:
            if input_text:
                return f"{instruction}\n\n{input_text}\n\n"
            else:
                return f"{instruction}\n\n"

    else:
        raise ValueError(f"Unknown template type: {template_type}")

    return format_fn
