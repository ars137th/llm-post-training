"""
Reward Model for Preference Learning

Wraps a language model with a value head to predict scalar rewards.
Used for reward modeling in RLHF.
"""

from typing import Dict, Optional, Union, Tuple
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
)

from .language import LanguageModel


class RewardModel(nn.Module):
    """
    Reward model for predicting human preferences.

    Architecture:
        Input: [Prompt + Response]
           ↓
        Language Model (frozen or fine-tuned)
           ↓
        Last Token Hidden State
           ↓
        Linear Value Head
           ↓
        Scalar Reward

    Example:
        >>> from src.models.language import LanguageModel
        >>>
        >>> # Load base model
        >>> base_model = LanguageModel.from_pretrained("gpt2")
        >>>
        >>> # Create reward model
        >>> reward_model = RewardModel(base_model)
        >>>
        >>> # Score a response
        >>> prompt = "What is AI?"
        >>> response = "AI is artificial intelligence..."
        >>> text = prompt + response
        >>>
        >>> tokens = base_model.tokenizer(text, return_tensors="pt")
        >>> reward = reward_model(**tokens)
        >>> print(f"Reward: {reward.item():.2f}")
    """

    def __init__(
        self,
        base_model: LanguageModel,
        freeze_base: bool = False,
    ):
        """
        Initialize reward model.

        Args:
            base_model: Pre-trained language model (usually SFT model)
            freeze_base: Whether to freeze the base model (only train value head)
        """
        super().__init__()

        self.base_model = base_model
        self.model = base_model.model
        self.tokenizer = base_model.tokenizer

        # Get hidden dimension from model config
        self.hidden_dim = self.model.config.hidden_size

        # Value head: hidden_dim -> 1 (scalar reward)
        self.value_head = nn.Linear(self.hidden_dim, 1, bias=False)

        # Initialize value head with small weights
        nn.init.normal_(self.value_head.weight, std=1 / (self.hidden_dim + 1))

        # Optionally freeze base model
        if freeze_base:
            self.freeze_base_model()

        # Move to same device as base model
        self.value_head.to(self.model.device)

    def freeze_base_model(self):
        """Freeze all parameters in the base model (only train value head)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze base model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through reward model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return dict with additional info

        Returns:
            If return_dict=False: Tensor of rewards [batch_size]
            If return_dict=True: Dict with 'rewards' and 'hidden_states'
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last layer hidden states [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]

        # Get the last token's hidden state for each sequence
        # We use the position of the last non-padding token
        if attention_mask is not None:
            # Find the last non-padding token position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            batch_size = input_ids.shape[0]

            # Gather last token hidden state
            # hidden_states: [batch_size, seq_len, hidden_dim]
            # We want: hidden_states[i, sequence_lengths[i], :] for each i
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
                :
            ]  # [batch_size, hidden_dim]
        else:
            # If no attention mask, use the last position
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]

        # Pass through value head to get scalar reward
        rewards = self.value_head(last_hidden).squeeze(-1)  # [batch_size]

        if return_dict:
            return {
                'rewards': rewards,
                'hidden_states': hidden_states,
                'last_hidden': last_hidden,
            }
        else:
            return rewards

    def compute_rewards(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 8,
    ) -> Union[float, list[float]]:
        """
        Compute rewards for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing multiple texts

        Returns:
            Single reward (if single text) or list of rewards
        """
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        all_rewards = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to device
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

            # Forward pass
            with torch.no_grad():
                rewards = self.forward(**encoded, return_dict=False)

            all_rewards.extend(rewards.cpu().tolist())

        return all_rewards[0] if single_text else all_rewards

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_value_head_parameters(self) -> int:
        """Number of parameters in value head."""
        return sum(p.numel() for p in self.value_head.parameters())

    @property
    def percent_trainable(self) -> float:
        """Percentage of trainable parameters."""
        return 100 * self.num_trainable_parameters / self.num_parameters

    @property
    def device(self) -> torch.device:
        """Device the model is on."""
        return self.model.device

    @property
    def is_peft_model(self) -> bool:
        """Check if using PEFT (LoRA)."""
        return isinstance(self.model, PeftModel)

    def save_pretrained(self, save_directory: str):
        """
        Save reward model.

        Saves:
        - Base model (or LoRA adapters if PEFT)
        - Value head weights
        - Tokenizer

        Args:
            save_directory: Directory to save to
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base model (or LoRA adapters)
        if self.is_peft_model:
            # Save LoRA adapters
            self.model.save_pretrained(save_directory)
        else:
            # Save full model
            self.model.save_pretrained(save_directory)

        # Save value head
        value_head_path = os.path.join(save_directory, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        print(f"Reward model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        freeze_base: bool = False,
        **kwargs,
    ) -> "RewardModel":
        """
        Load reward model from pretrained weights.

        Args:
            model_name_or_path: Path to model or HuggingFace model name
            use_lora: Whether to use LoRA
            lora_config: LoRA configuration
            freeze_base: Whether to freeze base model
            **kwargs: Additional arguments for model loading

        Returns:
            RewardModel instance
        """
        import os

        # Check if this is a saved reward model (has value_head.pt)
        value_head_path = os.path.join(model_name_or_path, "value_head.pt")
        has_saved_value_head = os.path.exists(value_head_path)

        # Load base language model
        base_model = LanguageModel.from_pretrained(
            model_name_or_path,
            use_lora=use_lora,
            lora_config=lora_config,
            **kwargs,
        )

        # Create reward model
        reward_model = cls(base_model, freeze_base=freeze_base)

        # Load value head if it exists
        if has_saved_value_head:
            state_dict = torch.load(value_head_path, map_location=reward_model.device)
            reward_model.value_head.load_state_dict(state_dict)
            print(f"Loaded value head from {value_head_path}")

        return reward_model

    def print_info(self):
        """Print model information."""
        print("=" * 60)
        print("Reward Model Information")
        print("=" * 60)
        print(f"Base Model: {self.model.config._name_or_path}")
        print(f"Hidden Dim: {self.hidden_dim}")
        print(f"Device: {self.device}")
        print(f"Is PEFT: {self.is_peft_model}")
        print()
        print(f"Total Parameters: {self.num_parameters:,}")
        print(f"Trainable Parameters: {self.num_trainable_parameters:,}")
        print(f"Value Head Parameters: {self.num_value_head_parameters:,}")
        print(f"Trainable %: {self.percent_trainable:.2f}%")
        print()

        # Show which components are trainable
        base_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        value_trainable = sum(
            p.numel() for p in self.value_head.parameters() if p.requires_grad
        )

        print("Trainable Components:")
        print(f"  Base Model: {base_trainable:,} params")
        print(f"  Value Head: {value_trainable:,} params")
        print("=" * 60)

    def __repr__(self) -> str:
        return (
            f"RewardModel(\n"
            f"  base_model={self.model.config._name_or_path},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  trainable_params={self.num_trainable_parameters:,},\n"
            f"  device={self.device}\n"
            f")"
        )
