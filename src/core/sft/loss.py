"""
Loss Functions for Supervised Fine-Tuning

Implements various loss functions for SFT including:
- Standard causal language modeling loss
- Focal loss for handling class imbalance
- Token-level accuracy computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalLMLoss(nn.Module):
    """
    Standard causal language modeling loss with label masking.

    This is the default loss for SFT. Computes cross-entropy loss only
    on non-masked tokens (labels != -100).
    """

    def __init__(self, ignore_index: int = -100):
        """
        Initialize causal LM loss.

        Args:
            ignore_index: Token ID to ignore in loss computation (typically -100)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Compute causal LM loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            return_details: If True, return additional metrics

        Returns:
            Loss tensor, or (loss, details_dict) if return_details=True
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # Compute loss
        loss = self.loss_fn(flat_logits, flat_labels)

        if not return_details:
            return loss

        # Compute additional metrics
        with torch.no_grad():
            # Token-level accuracy
            predictions = flat_logits.argmax(dim=-1)
            mask = flat_labels != self.ignore_index
            correct = (predictions == flat_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()

            # Perplexity
            perplexity = torch.exp(loss)

            # Number of tokens used in loss
            num_tokens = mask.sum().item()

        details = {
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item(),
            'num_tokens': num_tokens,
        }

        return loss, details


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in language modeling.

    Focal loss down-weights easy examples and focuses on hard examples.
    Useful when some tokens are much more common than others.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", 2017
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor (default 1.0)
            gamma: Focusing parameter (higher = focus more on hard examples)
            ignore_index: Token ID to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Compute focal loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            return_details: If True, return additional metrics

        Returns:
            Loss tensor, or (loss, details_dict) if return_details=True
        """
        # Shift and flatten
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.ignore_index,
            reduction='none',
        )

        # Compute focal term
        # p_t = probability of true class
        probs = F.softmax(flat_logits, dim=-1)
        mask = flat_labels != self.ignore_index

        # Gather probabilities for true labels
        p_t = probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
        p_t = torch.where(mask, p_t, torch.ones_like(p_t))  # Don't apply focal to masked

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * ce_loss

        # Average over non-masked tokens
        loss = focal_loss[mask].mean()

        if not return_details:
            return loss

        # Compute additional metrics
        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            correct = (predictions == flat_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            perplexity = torch.exp(ce_loss[mask].mean())
            num_tokens = mask.sum().item()

        details = {
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item(),
            'num_tokens': num_tokens,
            'avg_focal_weight': focal_weight[mask].mean().item(),
        }

        return loss, details


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Token ID to ignore

    Returns:
        Accuracy as a float
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Get predictions
    predictions = shift_logits.argmax(dim=-1)

    # Create mask for non-ignored tokens
    mask = shift_labels != ignore_index

    # Compute accuracy
    correct = (predictions == shift_labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity as a float
    """
    return torch.exp(loss).item()


def get_loss_function(loss_type: str = "causal_lm", **kwargs):
    """
    Factory function to get loss function by name.

    Args:
        loss_type: Type of loss ("causal_lm" or "focal")
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance

    Example:
        >>> loss_fn = get_loss_function("focal", alpha=1.0, gamma=2.0)
    """
    if loss_type == "causal_lm":
        return CausalLMLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: 'causal_lm', 'focal'"
        )
