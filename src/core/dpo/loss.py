"""
DPO (Direct Preference Optimization) Loss Functions

Implements the DPO loss for training language models directly from preference data
without needing a separate reward model or complex RL.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
           Rafailov et al., 2023 (https://arxiv.org/abs/2305.18290)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probabilities for a sequence.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        attention_mask: Optional mask [batch_size, seq_len]

    Returns:
        Log probabilities for each sequence [batch_size]

    Example:
        >>> logits = model(input_ids)
        >>> log_probs = compute_sequence_log_probs(logits, labels, attention_mask)
    """
    # Shift logits and labels for next-token prediction
    # logits: [batch, seq_len-1, vocab] (remove last position)
    # labels: [batch, seq_len-1] (remove first position)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len-1, vocab]

    # Gather log probs for actual tokens
    # [batch, seq_len-1, 1]
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len-1]

    # Apply attention mask if provided
    if attention_mask is not None:
        # Shift mask to align with shifted labels
        shift_mask = attention_mask[:, 1:].contiguous()
        token_log_probs = token_log_probs * shift_mask

        # Sum log probs only over valid tokens
        sequence_log_probs = token_log_probs.sum(dim=-1)  # [batch]
    else:
        # Sum over all positions
        sequence_log_probs = token_log_probs.sum(dim=-1)

    return sequence_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    return_details: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
    """
    Compute DPO loss.

    DPO Loss = -log(σ(β * (log(π_θ/π_ref)[y_w] - log(π_θ/π_ref)[y_l])))

    Where:
    - π_θ = policy model (being trained)
    - π_ref = reference model (frozen)
    - y_w = chosen/winner response
    - y_l = rejected/loser response
    - β = temperature parameter controlling strength of KL constraint
    - σ = sigmoid function

    Args:
        policy_chosen_logps: Log probs from policy for chosen responses [batch_size]
        policy_rejected_logps: Log probs from policy for rejected responses [batch_size]
        reference_chosen_logps: Log probs from reference for chosen responses [batch_size]
        reference_rejected_logps: Log probs from reference for rejected responses [batch_size]
        beta: Temperature parameter (default: 0.1)
        return_details: Whether to return detailed metrics

    Returns:
        If return_details=False: Loss tensor (scalar)
        If return_details=True: (loss, details_dict)

    Example:
        >>> # Forward pass through policy and reference
        >>> policy_chosen_lp = compute_sequence_log_probs(policy_logits_chosen, labels_chosen)
        >>> reference_chosen_lp = compute_sequence_log_probs(ref_logits_chosen, labels_chosen)
        >>> # ... same for rejected
        >>> loss = dpo_loss(policy_chosen_lp, policy_rejected_lp,
        ...                 reference_chosen_lp, reference_rejected_lp)
    """
    # Compute log ratios: log(π_θ(y|x) / π_ref(y|x))
    log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps

    # DPO objective: β * (log_ratio_chosen - log_ratio_rejected)
    # Intuition: Increase chosen, decrease rejected, relative to reference
    logits = beta * (log_ratio_chosen - log_ratio_rejected)

    # Loss: -log(sigmoid(logits))
    # Equivalent to: log(1 + exp(-logits))
    # Use logsigmoid for numerical stability
    loss = -F.logsigmoid(logits).mean()

    if return_details:
        with torch.no_grad():
            # Compute metrics
            # Implicit reward: r(x,y) = β * log(π(y|x) / π_ref(y|x))
            implicit_reward_chosen = beta * log_ratio_chosen
            implicit_reward_rejected = beta * log_ratio_rejected

            # Accuracy: How often does policy prefer chosen over rejected?
            # Based on implicit rewards
            accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()

            # Reward margin
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

            # KL divergence approximations (using log ratio)
            # KL(π || π_ref) ≈ log_ratio for small divergences
            chosen_kl = log_ratio_chosen.abs().mean()
            rejected_kl = log_ratio_rejected.abs().mean()

            details = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'reward_margin': reward_margin.item(),
                'reward_chosen_mean': implicit_reward_chosen.mean().item(),
                'reward_rejected_mean': implicit_reward_rejected.mean().item(),
                'reward_chosen_std': implicit_reward_chosen.std().item(),
                'reward_rejected_std': implicit_reward_rejected.std().item(),
                'log_ratio_chosen_mean': log_ratio_chosen.mean().item(),
                'log_ratio_rejected_mean': log_ratio_rejected.mean().item(),
                'chosen_kl': chosen_kl.item(),
                'rejected_kl': rejected_kl.item(),
            }

        return loss, details
    else:
        return loss


def dpo_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    Compute DPO evaluation metrics.

    Args:
        policy_chosen_logps: Log probs from policy for chosen
        policy_rejected_logps: Log probs from policy for rejected
        reference_chosen_logps: Log probs from reference for chosen
        reference_rejected_logps: Log probs from reference for rejected
        beta: Temperature parameter

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = dpo_metrics(policy_chosen_lp, policy_rejected_lp,
        ...                       ref_chosen_lp, ref_rejected_lp)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    # Compute log ratios
    log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps

    # Implicit rewards
    reward_chosen = beta * log_ratio_chosen
    reward_rejected = beta * log_ratio_rejected

    # Accuracy
    accuracy = (reward_chosen > reward_rejected).float().mean()

    # Margins
    margin = (reward_chosen - reward_rejected).mean()

    # KL divergences
    chosen_kl = log_ratio_chosen.abs().mean()
    rejected_kl = log_ratio_rejected.abs().mean()

    return {
        'accuracy': accuracy.item(),
        'margin_mean': margin.item(),
        'margin_std': (reward_chosen - reward_rejected).std().item(),
        'reward_chosen_mean': reward_chosen.mean().item(),
        'reward_chosen_std': reward_chosen.std().item(),
        'reward_rejected_mean': reward_rejected.mean().item(),
        'reward_rejected_std': reward_rejected.std().item(),
        'chosen_kl': chosen_kl.item(),
        'rejected_kl': rejected_kl.item(),
    }


class DPOLoss(nn.Module):
    """
    DPO loss as a PyTorch module.

    Useful for integration with Trainer classes.

    Args:
        beta: Temperature parameter controlling KL constraint strength

    Example:
        >>> loss_fn = DPOLoss(beta=0.1)
        >>> loss = loss_fn(policy_chosen_lp, policy_rejected_lp,
        ...                 ref_chosen_lp, ref_rejected_lp)
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """Compute DPO loss."""
        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.beta,
            return_details=return_details,
        )


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    return_details: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
    """
    Compute IPO (Identity Preference Optimization) loss.

    IPO is a variant of DPO that uses squared loss instead of log-sigmoid.
    More robust to outliers and noise.

    IPO Loss = (β * (log_ratio_chosen - log_ratio_rejected) - 0.5)²

    Args:
        policy_chosen_logps: Log probs from policy for chosen
        policy_rejected_logps: Log probs from policy for rejected
        reference_chosen_logps: Log probs from reference for chosen
        reference_rejected_logps: Log probs from reference for rejected
        beta: Temperature parameter
        return_details: Whether to return detailed metrics

    Returns:
        If return_details=False: Loss tensor (scalar)
        If return_details=True: (loss, details_dict)

    Reference: https://arxiv.org/abs/2310.12036
    """
    # Compute log ratios
    log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps

    # IPO objective
    logits = beta * (log_ratio_chosen - log_ratio_rejected) - 0.5

    # Squared loss
    loss = (logits ** 2).mean()

    if return_details:
        with torch.no_grad():
            # Same metrics as DPO
            implicit_reward_chosen = beta * log_ratio_chosen
            implicit_reward_rejected = beta * log_ratio_rejected

            accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

            details = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'reward_margin': reward_margin.item(),
                'reward_chosen_mean': implicit_reward_chosen.mean().item(),
                'reward_rejected_mean': implicit_reward_rejected.mean().item(),
            }

        return loss, details
    else:
        return loss
