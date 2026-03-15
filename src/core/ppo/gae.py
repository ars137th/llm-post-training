"""
Generalized Advantage Estimation (GAE)

Implements GAE for computing advantage estimates in PPO.

Advantage tells us "how much better is this action compared to average?"
GAE provides a bias-variance trade-off via the λ parameter.

References:
- GAE paper: https://arxiv.org/abs/1506.02438
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

from typing import Dict, Tuple
import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    return_details: bool = False,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation.

    GAE formula:
        A_t = sum_{l=0}^{∞} (γλ)^l δ_{t+l}

    Where δ_t is the temporal difference error:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

    In practice, we compute recursively:
        A_t = δ_t + γλ * A_{t+1}

    Args:
        rewards: Rewards for each step [batch_size, seq_len] or [batch_size]
        values: Value estimates for each step [batch_size, seq_len + 1] or [batch_size + 1]
            Note: values includes V(s_T+1) = 0 for terminal state
        gamma: Discount factor (default 0.99)
        lam: GAE lambda parameter (default 0.95)
        return_details: Return additional info

    Returns:
        advantages: GAE advantage estimates [batch_size, seq_len] or [batch_size]
        details (optional): Dict with metrics

    Note:
        For language models, we typically have:
        - rewards: [batch_size] (one reward per sequence)
        - values: [batch_size + 1] (value for each prompt, plus terminal)

        In this case, advantages is also [batch_size].

    For RL with intermediate rewards:
        - rewards: [batch_size, seq_len] (reward at each token)
        - values: [batch_size, seq_len + 1] (value at each state)

        This is more complex but allows credit assignment per token.
    """
    # Handle both 1D and 2D tensors
    if rewards.dim() == 1:
        # Single reward per sequence: [batch_size]
        return _compute_gae_single_reward(rewards, values, gamma, lam, return_details)
    else:
        # Reward per step: [batch_size, seq_len]
        return _compute_gae_per_step(rewards, values, gamma, lam, return_details)


def _compute_gae_single_reward(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    return_details: bool,
) -> torch.Tensor:
    """
    Compute GAE for single reward per sequence.

    This is the simpler case for language model RLHF:
    - Generate full response
    - Get single reward from reward model
    - Compute advantage

    Args:
        rewards: [batch_size]
        values: [batch_size + 1] where values[-1] = 0 (terminal)
        gamma: Discount factor
        lam: GAE lambda
        return_details: Return metrics

    Returns:
        advantages: [batch_size]
        details (optional): Dict with metrics
    """
    batch_size = rewards.shape[0]

    # Compute TD errors: δ_t = r_t + γ * V_{t+1} - V_t
    # For single reward, r_t is only non-zero at the final step
    # So: δ = reward + γ * 0 - V(s_0)
    #       = reward - V(s_0)
    deltas = rewards - values[:-1]  # values[:-1] is V(s_0), values[-1] is V(terminal)=0

    # For single-step, GAE simplifies to just the delta
    # (No summation over future steps since there's only one step)
    advantages = deltas

    if return_details:
        with torch.no_grad():
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            adv_max = advantages.max()
            adv_min = advantages.min()

            # Compute returns for reference
            returns = advantages + values[:-1]
            return_mean = returns.mean()

            details = {
                'advantage_mean': adv_mean.item(),
                'advantage_std': adv_std.item(),
                'advantage_max': adv_max.item(),
                'advantage_min': adv_min.item(),
                'return_mean': return_mean.item(),
            }
            return advantages, details

    return advantages


def _compute_gae_per_step(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    return_details: bool,
) -> torch.Tensor:
    """
    Compute GAE for rewards at each step.

    This is the more complex case with intermediate rewards:
    - Reward at each token
    - Value estimate at each state
    - GAE computed recursively

    Args:
        rewards: [batch_size, seq_len]
        values: [batch_size, seq_len + 1]
        gamma: Discount factor
        lam: GAE lambda
        return_details: Return metrics

    Returns:
        advantages: [batch_size, seq_len]
        details (optional): Dict with metrics
    """
    batch_size, seq_len = rewards.shape

    # Compute TD errors: δ_t = r_t + γ * V_{t+1} - V_t
    values_current = values[:, :-1]  # V(s_t)
    values_next = values[:, 1:]       # V(s_{t+1})

    deltas = rewards + gamma * values_next - values_current

    # Compute GAE recursively: A_t = δ_t + γλ * A_{t+1}
    advantages = torch.zeros_like(rewards)

    # Start from the end (A_T = δ_T)
    gae = torch.zeros(batch_size, device=rewards.device)

    # Work backwards
    for t in reversed(range(seq_len)):
        gae = deltas[:, t] + gamma * lam * gae
        advantages[:, t] = gae

    if return_details:
        with torch.no_grad():
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            adv_max = advantages.max()
            adv_min = advantages.min()

            # Compute returns
            returns = advantages + values_current
            return_mean = returns.mean()

            # Compute mean delta (TD error)
            delta_mean = deltas.mean()

            details = {
                'advantage_mean': adv_mean.item(),
                'advantage_std': adv_std.item(),
                'advantage_max': adv_max.item(),
                'advantage_min': adv_min.item(),
                'return_mean': return_mean.item(),
                'delta_mean': delta_mean.item(),
            }
            return advantages, details

    return advantages


def compute_returns(
    rewards: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute discounted returns (used for value function targets).

    R_t = sum_{k=0}^{∞} γ^k r_{t+k}

    Computed recursively:
        R_t = r_t + γ * R_{t+1}

    Args:
        rewards: Rewards [batch_size, seq_len] or [batch_size]
        gamma: Discount factor

    Returns:
        returns: Discounted returns (same shape as rewards)
    """
    if rewards.dim() == 1:
        # Single reward: return is just the reward
        return rewards

    # Per-step rewards: compute recursively
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)

    # Start from the end (R_T = r_T)
    running_return = torch.zeros(batch_size, device=rewards.device)

    # Work backwards
    for t in reversed(range(seq_len)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    return returns


def normalize_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.

    This is a common trick to stabilize PPO training.

    Args:
        advantages: Raw advantages
        epsilon: Small constant for numerical stability

    Returns:
        Normalized advantages
    """
    mean = advantages.mean()
    std = advantages.std()
    normalized = (advantages - mean) / (std + epsilon)
    return normalized


def whiten_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Whiten advantages (normalize per batch).

    Same as normalize_advantages but clearer name.

    Args:
        advantages: Raw advantages
        epsilon: Small constant for numerical stability

    Returns:
        Whitened advantages
    """
    return normalize_advantages(advantages, epsilon)


def compute_value_targets(
    advantages: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute value function targets.

    Target = Advantage + Baseline
           = (Q - V) + V
           = Q
           = Expected return

    Args:
        advantages: GAE advantages
        values: Value estimates (baseline)

    Returns:
        Value targets (returns)
    """
    # Handle both 1D and 2D
    if advantages.dim() == 1:
        # values is [batch_size + 1], we want [batch_size]
        return advantages + values[:-1]
    else:
        # values is [batch_size, seq_len + 1], we want [batch_size, seq_len]
        return advantages + values[:, :-1]


# Diagnostic utilities

def check_gae_health(
    advantages: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
) -> Dict[str, any]:
    """
    Check if GAE computation looks healthy.

    Good indicators:
    - Advantages have reasonable variance (not too large or too small)
    - Values are in same range as rewards
    - No NaN or Inf values

    Args:
        advantages: Computed advantages
        values: Value estimates
        rewards: Rewards

    Returns:
        Dict with diagnostic information
    """
    with torch.no_grad():
        # Advantage statistics
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        adv_min = advantages.min().item()
        adv_max = advantages.max().item()

        # Value statistics
        val_mean = values.mean().item()
        val_std = values.std().item()

        # Reward statistics
        rew_mean = rewards.mean().item()
        rew_std = rewards.std().item()

        # Check for numerical issues
        has_nan_adv = torch.isnan(advantages).any().item()
        has_inf_adv = torch.isinf(advantages).any().item()
        has_nan_val = torch.isnan(values).any().item()
        has_inf_val = torch.isinf(values).any().item()

        # Compute how well values predict rewards
        # (values should be in same ballpark as rewards)
        value_reward_ratio = val_mean / (rew_mean + 1e-8)

        diagnostics = {
            'advantage_mean': adv_mean,
            'advantage_std': adv_std,
            'advantage_min': adv_min,
            'advantage_max': adv_max,
            'value_mean': val_mean,
            'value_std': val_std,
            'reward_mean': rew_mean,
            'reward_std': rew_std,
            'value_reward_ratio': value_reward_ratio,
            'has_nan_advantages': has_nan_adv,
            'has_inf_advantages': has_inf_adv,
            'has_nan_values': has_nan_val,
            'has_inf_values': has_inf_val,
            'is_healthy': (
                not has_nan_adv and
                not has_inf_adv and
                not has_nan_val and
                not has_inf_val and
                adv_std > 0.01 and  # Not degenerate
                adv_std < 1000 and  # Not exploding
                0.1 < value_reward_ratio < 10  # Values in reasonable range
            )
        }

        return diagnostics
