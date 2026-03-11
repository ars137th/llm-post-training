"""
Loss Functions for Reward Modeling

Implements Bradley-Terry ranking loss and related metrics.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    margin: float = 0.0,
    return_details: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
    """
    Compute Bradley-Terry ranking loss.

    The Bradley-Terry model predicts the probability that response A
    is preferred over response B as:

        P(A > B) = sigmoid(R(A) - R(B))

    The loss maximizes the log-likelihood of observed preferences:

        Loss = -log(sigmoid(R(chosen) - R(rejected)))
             = log(1 + exp(R(rejected) - R(chosen)))

    Args:
        reward_chosen: Rewards for chosen responses [batch_size]
        reward_rejected: Rewards for rejected responses [batch_size]
        margin: Optional margin to add (for margin-based loss) [default: 0.0]
        return_details: Whether to return detailed metrics

    Returns:
        If return_details=False: Loss tensor (scalar)
        If return_details=True: (loss, details_dict)

    Example:
        >>> reward_chosen = torch.tensor([2.5, 3.0, 1.5])
        >>> reward_rejected = torch.tensor([-1.0, 0.5, 1.0])
        >>> loss = bradley_terry_loss(reward_chosen, reward_rejected)
        >>> print(f"Loss: {loss.item():.4f}")
    """
    # Compute reward difference (with optional margin)
    # We want: R(chosen) - R(rejected) - margin > 0
    reward_diff = reward_chosen - reward_rejected - margin

    # Bradley-Terry loss: -log(sigmoid(reward_diff))
    # Equivalent to: log(1 + exp(-reward_diff))
    # We use logsigmoid for numerical stability
    loss = -F.logsigmoid(reward_diff).mean()

    if return_details:
        with torch.no_grad():
            # Compute metrics
            accuracy = (reward_chosen > reward_rejected).float().mean()
            avg_chosen = reward_chosen.mean()
            avg_rejected = reward_rejected.mean()
            avg_margin = reward_diff.mean()

            details = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'reward_chosen_mean': avg_chosen.item(),
                'reward_rejected_mean': avg_rejected.item(),
                'reward_margin_mean': avg_margin.item(),
                'reward_chosen_std': reward_chosen.std().item(),
                'reward_rejected_std': reward_rejected.std().item(),
            }

        return loss, details
    else:
        return loss


def margin_ranking_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Hinge loss variant for reward modeling.

    Loss = max(0, margin - (R(chosen) - R(rejected)))

    This encourages a minimum separation between chosen and rejected rewards.

    Args:
        reward_chosen: Rewards for chosen responses [batch_size]
        reward_rejected: Rewards for rejected responses [batch_size]
        margin: Minimum desired separation [default: 1.0]

    Returns:
        Loss tensor (scalar)

    Example:
        >>> reward_chosen = torch.tensor([2.0, 3.0])
        >>> reward_rejected = torch.tensor([1.0, 0.5])
        >>> loss = margin_ranking_loss(reward_chosen, reward_rejected, margin=1.0)
    """
    # Hinge loss
    loss = F.relu(margin - (reward_chosen - reward_rejected))
    return loss.mean()


def compute_ranking_accuracy(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> float:
    """
    Compute ranking accuracy: % of pairs where R(chosen) > R(rejected).

    Args:
        reward_chosen: Rewards for chosen responses [batch_size]
        reward_rejected: Rewards for rejected responses [batch_size]

    Returns:
        Accuracy as float (0.0 to 1.0)

    Example:
        >>> reward_chosen = torch.tensor([2.0, 3.0, 1.0])
        >>> reward_rejected = torch.tensor([1.0, 2.5, 1.5])
        >>> acc = compute_ranking_accuracy(reward_chosen, reward_rejected)
        >>> print(f"Accuracy: {acc:.2%}")  # 66.67%
    """
    correct = (reward_chosen > reward_rejected).float().sum()
    total = reward_chosen.numel()
    return (correct / total).item()


def compute_reward_margin(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics about the reward margin (separation).

    Args:
        reward_chosen: Rewards for chosen responses [batch_size]
        reward_rejected: Rewards for rejected responses [batch_size]

    Returns:
        Dictionary with margin statistics

    Example:
        >>> reward_chosen = torch.tensor([2.0, 3.0, 1.5])
        >>> reward_rejected = torch.tensor([1.0, 0.5, 1.0])
        >>> stats = compute_reward_margin(reward_chosen, reward_rejected)
        >>> print(stats)
    """
    margin = reward_chosen - reward_rejected

    return {
        'margin_mean': margin.mean().item(),
        'margin_std': margin.std().item(),
        'margin_min': margin.min().item(),
        'margin_max': margin.max().item(),
        'chosen_mean': reward_chosen.mean().item(),
        'rejected_mean': reward_rejected.mean().item(),
    }


class BradleyTerryLoss(nn.Module):
    """
    Bradley-Terry loss as a PyTorch module.

    Useful for integration with Trainer classes.

    Args:
        margin: Optional margin for margin-based loss [default: 0.0]

    Example:
        >>> loss_fn = BradleyTerryLoss(margin=0.0)
        >>> reward_chosen = torch.tensor([2.0, 3.0])
        >>> reward_rejected = torch.tensor([1.0, 0.5])
        >>> loss = loss_fn(reward_chosen, reward_rejected)
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        reward_chosen: torch.Tensor,
        reward_rejected: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """Compute Bradley-Terry loss."""
        return bradley_terry_loss(
            reward_chosen,
            reward_rejected,
            margin=self.margin,
            return_details=return_details,
        )


def compute_reward_statistics(
    rewards: torch.Tensor,
    name: str = "rewards",
) -> Dict[str, float]:
    """
    Compute statistics for a set of rewards.

    Args:
        rewards: Tensor of reward values
        name: Name prefix for keys in returned dict

    Returns:
        Dictionary with statistics

    Example:
        >>> rewards = torch.randn(100)
        >>> stats = compute_reward_statistics(rewards, name="train_rewards")
        >>> print(stats.keys())
    """
    return {
        f'{name}_mean': rewards.mean().item(),
        f'{name}_std': rewards.std().item(),
        f'{name}_min': rewards.min().item(),
        f'{name}_max': rewards.max().item(),
        f'{name}_median': rewards.median().item(),
    }


def calibration_error(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """
    Compute calibration error for reward model.

    Measures whether the model's confidence (margin) matches actual accuracy.

    Args:
        reward_chosen: Rewards for chosen responses
        reward_rejected: Rewards for rejected responses
        num_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error (ECE)

    Example:
        >>> # Model should be more accurate when more confident (larger margin)
        >>> reward_chosen = torch.randn(1000) + 1.0  # Biased higher
        >>> reward_rejected = torch.randn(1000)
        >>> ece = calibration_error(reward_chosen, reward_rejected)
        >>> print(f"Calibration Error: {ece:.4f}")
    """
    margin = (reward_chosen - reward_rejected).abs()
    correct = (reward_chosen > reward_rejected).float()

    # Sort by confidence (margin)
    sorted_indices = margin.argsort()
    margin_sorted = margin[sorted_indices]
    correct_sorted = correct[sorted_indices]

    # Bin into num_bins
    bin_size = len(margin) // num_bins
    ece = 0.0

    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < num_bins - 1 else len(margin)

        bin_margin = margin_sorted[start_idx:end_idx]
        bin_correct = correct_sorted[start_idx:end_idx]

        if len(bin_correct) > 0:
            # Average confidence in this bin
            avg_confidence = torch.sigmoid(bin_margin).mean().item()

            # Actual accuracy in this bin
            actual_accuracy = bin_correct.mean().item()

            # Calibration error for this bin
            bin_error = abs(avg_confidence - actual_accuracy)

            # Weight by bin size
            weight = len(bin_correct) / len(margin)
            ece += weight * bin_error

    return ece


# For backwards compatibility and convenience
def reward_model_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Alias for bradley_terry_loss for convenience.

    This is the standard loss for reward modeling.
    """
    return bradley_terry_loss(reward_chosen, reward_rejected, **kwargs)
