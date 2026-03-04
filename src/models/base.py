"""
Base Model Interface

Defines the protocol that all model wrappers (language, vision-language)
should implement for consistency across modalities.
"""

from typing import Dict, Optional, Protocol
import torch


class BaseModel(Protocol):
    """
    Protocol defining the interface for all model wrappers.

    This ensures that text-only and multimodal models have a consistent API,
    making it easy to swap between them in training scripts.
    """

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
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing loss (optional)
            **kwargs: Additional model-specific inputs (e.g., pixel_values for vision models)

        Returns:
            Dictionary containing:
            - loss: Loss value (if labels provided)
            - logits: Model logits
            - hidden_states: Hidden states (optional)
        """
        ...

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate sequences using the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        ...

    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for sequences.

        Essential for DPO and PPO implementations.

        Args:
            input_ids: Input token IDs
            labels: Token IDs to compute log probs for
            attention_mask: Attention mask

        Returns:
            Log probabilities per token
        """
        ...

    def train(self):
        """Set model to training mode."""
        ...

    def eval(self):
        """Set model to evaluation mode."""
        ...

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        ...
