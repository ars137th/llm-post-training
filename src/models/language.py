"""
Language Model Wrapper for Text-Only Models

Provides a unified interface for working with text-only language models
(GPT-2, LLaMA, Mistral, OPT, etc.) with support for LoRA/PEFT.
"""

from typing import Dict, Optional, Union, List, Tuple
import sys
import warnings
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig


# Check Python version compatibility
if sys.version_info >= (3, 14):
    warnings.warn(
        "Python 3.14+ detected. PyTorch and transformers may have compatibility issues "
        "with multiprocessing, which can cause bus errors during model loading/generation. "
        "Consider using Python 3.10-3.11 for best stability. "
        "If you encounter issues, try: PYTORCH_ENABLE_MPS_FALLBACK=1 or force CPU mode.",
        UserWarning,
    )


class LanguageModel:
    """
    Unified wrapper for text-only language models.

    Supports:
    - Loading from HuggingFace Hub
    - LoRA/QLoRA integration for efficient fine-tuning
    - Text generation with various sampling strategies
    - Log probability computation for RLHF/DPO
    - Device management and mixed precision

    Example:
        >>> model = LanguageModel.from_pretrained("gpt2", use_lora=True)
        >>> output = model.generate({"input_ids": inputs})
        >>> logprobs = model.get_logprobs({"input_ids": inputs}, output)
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """
        Initialize LanguageModel wrapper.

        Args:
            model: The language model (with or without PEFT)
            tokenizer: The tokenizer
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device if not using device_map
        if not hasattr(model, "hf_device_map"):
            self.model = self.model.to(self.device)

        self.is_peft_model = isinstance(model, PeftModel)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **model_kwargs,
    ) -> "LanguageModel":
        """
        Load a language model from HuggingFace Hub.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
            use_lora: Whether to apply LoRA adapters
            lora_config: Custom LoRA configuration (if None, uses sensible defaults)
            use_4bit: Load model in 4-bit quantization (QLoRA)
            use_8bit: Load model in 8-bit quantization
            device: Device to use (None for auto-detect)
            trust_remote_code: Trust remote code (needed for some models)
            **model_kwargs: Additional arguments for model loading

        Returns:
            LanguageModel instance
        """
        # Setup quantization config if requested
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # For decoder-only models, use left padding for generation
        # (right padding can cause issues with attention)
        tokenizer.padding_side = "left"

        # Load model
        # Note: device_map="auto" uses multiprocessing which can cause issues
        # with Python 3.14+ or on some systems. Only use for quantization.
        # Note: low_cpu_mem_usage=True can trigger BLAS bugs on some macOS systems
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            **model_kwargs,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"  # Required for quantization

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        # Apply LoRA if requested
        if use_lora:
            if use_4bit or use_8bit:
                # Prepare model for k-bit training
                model = prepare_model_for_kbit_training(model)

            # Setup LoRA config with sensible defaults
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": cls._get_target_modules(model_name),
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": TaskType.CAUSAL_LM,
                }

            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        return cls(model=model, tokenizer=tokenizer, device=device)

    @staticmethod
    def _get_target_modules(model_name: str) -> List[str]:
        """
        Get appropriate LoRA target modules for different model architectures.

        Args:
            model_name: Model name to infer architecture from

        Returns:
            List of module names to apply LoRA to
        """
        model_name_lower = model_name.lower()

        if "llama" in model_name_lower or "mistral" in model_name_lower:
            # LLaMA/Mistral architecture
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in model_name_lower or "gpt-2" in model_name_lower:
            # GPT-2 architecture
            return ["c_attn", "c_proj"]
        elif "opt" in model_name_lower:
            # OPT architecture
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        else:
            # Default: target attention layers
            return ["q_proj", "v_proj"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for computing loss [batch_size, seq_len]
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'loss' (if labels provided), 'logits', and other outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate per input
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs [batch_size * num_return_sequences, total_seq_len]
        """
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs,
            )

        return outputs

    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for given sequences.

        Essential for DPO and PPO where we need to compute:
        - log π(y|x) for policy model
        - log π_ref(y|x) for reference model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Token IDs to compute log probs for [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Log probabilities [batch_size, seq_len]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Shift logits and labels for causal LM
        # We want to predict token i using tokens 0..i-1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for the actual labels
        # [batch_size, seq_len-1, vocab_size] -> [batch_size, seq_len-1]
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Pad back to original length
        token_log_probs = torch.nn.functional.pad(
            token_log_probs,
            (1, 0),  # Pad 1 on the left
            value=0.0,
        )

        return token_log_probs

    def compute_sequence_logprob(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total log probability of a sequence (sum over tokens).

        Useful for preference modeling where we compare log P(chosen) vs log P(rejected).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Token IDs to compute log probs for [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            label_mask: Mask for which tokens to include in sum [batch_size, seq_len]
                       (useful to exclude prompt tokens, only sum over response)

        Returns:
            Total log probability per sequence [batch_size]
        """
        token_log_probs = self.get_logprobs(input_ids, labels, attention_mask)

        # Apply label mask if provided (to exclude prompt tokens)
        if label_mask is not None:
            token_log_probs = token_log_probs * label_mask
        elif attention_mask is not None:
            # Otherwise use attention mask
            token_log_probs = token_log_probs * attention_mask

        # Sum over sequence length
        return token_log_probs.sum(dim=-1)

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer to directory."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def __call__(self, *args, **kwargs):
        """Allow model to be called directly."""
        return self.forward(*args, **kwargs)

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
