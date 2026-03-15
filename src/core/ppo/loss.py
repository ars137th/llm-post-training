"""
PPO Loss Functions

Implements the core loss functions for Proximal Policy Optimization:
- Clipped surrogate objective (PPO-CLIP)
- Value function loss
- Entropy bonus
- KL divergence

References:
- PPO paper: https://arxiv.org/abs/1707.06347
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probabilities for a sequence.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Token IDs [batch_size, seq_len]
        attention_mask: Mask for valid tokens [batch_size, seq_len]

    Returns:
        Log probabilities per sequence [batch_size]
    """
    # Shift for causal LM (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Apply mask if provided
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        token_log_probs = token_log_probs * shift_mask

    # Sum over sequence length
    sequence_log_probs = token_log_probs.sum(dim=-1)

    return sequence_log_probs


def compute_entropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute entropy of the policy distribution.

    Entropy = -sum(p * log(p)) measures randomness of the distribution.
    Higher entropy = more exploration.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        attention_mask: Mask for valid tokens [batch_size, seq_len]

    Returns:
        Mean entropy per sequence [batch_size]
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()

    # Compute probabilities
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    # Apply mask if provided
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        entropy = entropy * shift_mask
        # Average over valid tokens
        num_valid = shift_mask.sum(dim=-1).clamp(min=1)
        mean_entropy = entropy.sum(dim=-1) / num_valid
    else:
        mean_entropy = entropy.mean(dim=-1)

    return mean_entropy


def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Compute PPO clipped surrogate objective.

    L^CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

    Where:
        r(θ) = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
        A = advantage estimate
        ε = clip range (default 0.2)

    The min() ensures we don't update the policy too much:
    - If A > 0 (good action): encourage, but cap at (1+ε)
    - If A < 0 (bad action): discourage, but cap at (1-ε)

    Args:
        log_probs: Log probs from current policy [batch_size]
        old_log_probs: Log probs from old policy [batch_size]
        advantages: Advantage estimates [batch_size]
        clip_range: PPO clipping parameter (epsilon)
        return_details: Return detailed metrics

    Returns:
        loss: PPO loss (scalar)
        details (optional): Dict with metrics
    """
    # Importance sampling ratio: π_new / π_old
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped ratio
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # Surrogate losses
    loss_unclipped = ratio * advantages
    loss_clipped = ratio_clipped * advantages

    # PPO objective: maximize min(unclipped, clipped)
    # We minimize negative, so: min(loss) = -max(objective)
    loss = -torch.min(loss_unclipped, loss_clipped).mean()

    if return_details:
        with torch.no_grad():
            # Clip fraction: how often we're clipping
            # Should be 10-30% (if 0%, not learning; if 100%, updates too large)
            clip_fraction = ((ratio - ratio_clipped).abs() > 1e-6).float().mean()

            # Approximate KL divergence (for monitoring)
            # approx_kl = E[(π_new / π_old - 1) - log(π_new / π_old)]
            approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean()

            # Policy ratio statistics
            ratio_mean = ratio.mean()
            ratio_std = ratio.std()

            # How many updates increased probability (ratio > 1)
            positive_advantages = (advantages > 0).float().mean()

            details = {
                'loss': loss.item(),
                'clip_fraction': clip_fraction.item(),
                'approx_kl': approx_kl.item(),
                'ratio_mean': ratio_mean.item(),
                'ratio_std': ratio_std.item(),
                'positive_advantages': positive_advantages.item(),
            }
            return loss, details

    return loss


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_range_vf: Optional[float] = None,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Compute value function loss.

    L^VF(φ) = E[(V_φ(s) - R)^2]

    Optionally with clipping (similar to PPO):
    V_clipped = V_old + clip(V - V_old, -ε, ε)
    L^VF = E[max((V - R)^2, (V_clipped - R)^2)]

    Args:
        values: Value predictions from critic [batch_size]
        returns: Actual returns (advantages + old_values) [batch_size]
        old_values: Value predictions from old critic (for clipping) [batch_size]
        clip_range_vf: Clipping range for value function (optional)
        return_details: Return detailed metrics

    Returns:
        loss: Value loss (scalar)
        details (optional): Dict with metrics
    """
    # Standard value loss: MSE
    loss_unclipped = (values - returns) ** 2

    if clip_range_vf is not None and old_values is not None:
        # Clipped value loss (prevents large updates)
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -clip_range_vf,
            clip_range_vf
        )
        loss_clipped = (values_clipped - returns) ** 2

        # Use max (more conservative)
        loss = torch.max(loss_unclipped, loss_clipped).mean()
    else:
        loss = loss_unclipped.mean()

    if return_details:
        with torch.no_grad():
            # Value prediction statistics
            value_mean = values.mean()
            value_std = values.std()
            return_mean = returns.mean()
            return_std = returns.std()

            # Explained variance: how well values predict returns
            # EV = 1 - Var(returns - values) / Var(returns)
            # EV = 1 means perfect prediction, 0 means no better than mean
            var_values = ((returns - values) ** 2).mean()
            var_returns = ((returns - return_mean) ** 2).mean()
            explained_variance = 1 - (var_values / (var_returns + 1e-8))

            details = {
                'loss': loss.item(),
                'value_mean': value_mean.item(),
                'value_std': value_std.item(),
                'return_mean': return_mean.item(),
                'return_std': return_std.item(),
                'explained_variance': explained_variance.item(),
            }
            return loss, details

    return loss


def kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Compute KL divergence between policy and reference.

    KL(π || π_ref) = E[log(π/π_ref)] = E[log π - log π_ref]

    For language models, this is computed per sequence:
    KL = sum_t [log π(y_t | x, y_<t) - log π_ref(y_t | x, y_<t)]

    Args:
        log_probs: Log probs from current policy [batch_size]
        ref_log_probs: Log probs from reference policy [batch_size]
        return_details: Return detailed metrics

    Returns:
        kl: Mean KL divergence (scalar)
        details (optional): Dict with metrics
    """
    kl_per_sample = log_probs - ref_log_probs
    kl_mean = kl_per_sample.mean()

    if return_details:
        with torch.no_grad():
            kl_std = kl_per_sample.std()
            kl_max = kl_per_sample.max()
            kl_min = kl_per_sample.min()

            details = {
                'kl_mean': kl_mean.item(),
                'kl_std': kl_std.item(),
                'kl_max': kl_max.item(),
                'kl_min': kl_min.item(),
            }
            return kl_mean, details

    return kl_mean


def compute_rlhf_reward(
    reward_model_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_coef: float = 0.05,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Compute RLHF reward with KL penalty.

    R(x, y) = R_RM(x, y) - β * KL(π_θ || π_ref)

    Where:
        R_RM = reward model score
        β = KL coefficient (controls how much to penalize drift)
        KL = KL divergence from reference policy

    Args:
        reward_model_scores: Scores from reward model [batch_size]
        log_probs: Log probs from current policy [batch_size]
        ref_log_probs: Log probs from reference policy [batch_size]
        kl_coef: KL penalty coefficient (beta)
        return_details: Return detailed metrics

    Returns:
        rewards: Total rewards with KL penalty [batch_size]
        details (optional): Dict with metrics
    """
    # KL divergence per sequence
    kl_per_sample = log_probs - ref_log_probs

    # Total reward = reward model score - KL penalty
    rewards = reward_model_scores - kl_coef * kl_per_sample

    if return_details:
        with torch.no_grad():
            rm_mean = reward_model_scores.mean()
            rm_std = reward_model_scores.std()
            kl_mean = kl_per_sample.mean()
            kl_std = kl_per_sample.std()
            reward_mean = rewards.mean()
            reward_std = rewards.std()

            # What fraction of reward comes from RM vs KL penalty?
            kl_penalty_mean = (kl_coef * kl_per_sample).mean()
            penalty_fraction = kl_penalty_mean / (rm_mean + 1e-8)

            details = {
                'reward_total_mean': reward_mean.item(),
                'reward_total_std': reward_std.item(),
                'reward_model_mean': rm_mean.item(),
                'reward_model_std': rm_std.item(),
                'kl_mean': kl_mean.item(),
                'kl_std': kl_std.item(),
                'kl_penalty_mean': kl_penalty_mean.item(),
                'penalty_fraction': penalty_fraction.item(),
            }
            return rewards, details

    return rewards


def policy_entropy_loss(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Compute entropy loss (for maximizing exploration).

    L^ENT = -E[H(π)]

    We minimize this loss, which maximizes entropy.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        attention_mask: Mask for valid tokens [batch_size, seq_len]
        return_details: Return detailed metrics

    Returns:
        loss: Negative entropy (scalar)
        details (optional): Dict with metrics
    """
    entropy = compute_entropy(logits, attention_mask)
    loss = -entropy.mean()  # Negative because we want to maximize

    if return_details:
        with torch.no_grad():
            entropy_mean = entropy.mean()
            entropy_std = entropy.std()

            details = {
                'entropy_loss': loss.item(),
                'entropy_mean': entropy_mean.item(),
                'entropy_std': entropy_std.item(),
            }
            return loss, details

    return loss


def total_ppo_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> torch.Tensor:
    """
    Compute total PPO loss.

    L_total = L^CLIP + c1 * L^VF - c2 * L^ENT

    Where:
        L^CLIP = PPO clipped loss
        L^VF = Value function loss
        L^ENT = Entropy loss (negative entropy)
        c1 = value loss coefficient
        c2 = entropy coefficient

    Args:
        policy_loss: PPO clipped loss
        value_loss: Value function MSE loss
        entropy_loss: Negative entropy
        vf_coef: Value loss coefficient (c1)
        ent_coef: Entropy coefficient (c2)

    Returns:
        total_loss: Combined loss for optimization
    """
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    return total_loss


# Utility functions for testing and debugging

def check_ppo_ratio(log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> Dict[str, float]:
    """
    Check if policy ratio is in reasonable range.

    Good indicators:
    - Mean ratio close to 1.0 (policy hasn't changed much)
    - Std ratio < 0.5 (not too much variance)
    - No extreme ratios (< 0.1 or > 10)

    Args:
        log_probs: Current policy log probs [batch_size]
        old_log_probs: Old policy log probs [batch_size]

    Returns:
        Dict with diagnostic information
    """
    with torch.no_grad():
        ratio = torch.exp(log_probs - old_log_probs)

        ratio_mean = ratio.mean().item()
        ratio_std = ratio.std().item()
        ratio_min = ratio.min().item()
        ratio_max = ratio.max().item()

        # Count extreme ratios
        extreme_low = (ratio < 0.1).float().mean().item()
        extreme_high = (ratio > 10.0).float().mean().item()

        # Check for NaN or Inf
        has_nan = torch.isnan(ratio).any().item()
        has_inf = torch.isinf(ratio).any().item()

        diagnostics = {
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'ratio_min': ratio_min,
            'ratio_max': ratio_max,
            'extreme_low_fraction': extreme_low,
            'extreme_high_fraction': extreme_high,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'is_healthy': (
                0.5 < ratio_mean < 2.0 and
                ratio_std < 1.0 and
                not has_nan and
                not has_inf and
                extreme_low < 0.01 and
                extreme_high < 0.01
            )
        }

        return diagnostics
