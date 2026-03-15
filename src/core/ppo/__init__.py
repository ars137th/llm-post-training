"""
PPO (Proximal Policy Optimization) for RLHF

Core components for PPO training:
- Loss functions (clipped objective, value loss, entropy)
- Generalized Advantage Estimation (GAE)
- Rollout buffer for trajectory storage
- PPO trainer with custom loop (to be implemented)
"""

from .loss import (
    compute_log_probs,
    compute_entropy,
    ppo_loss,
    value_loss,
    kl_divergence,
    compute_rlhf_reward,
    policy_entropy_loss,
    total_ppo_loss,
    check_ppo_ratio,
)

from .gae import (
    compute_gae,
    compute_returns,
    normalize_advantages,
    whiten_advantages,
    compute_value_targets,
    check_gae_health,
)

from .buffer import (
    RolloutBatch,
    RolloutBuffer,
    create_rollout_batch,
)

__all__ = [
    # Loss functions
    'compute_log_probs',
    'compute_entropy',
    'ppo_loss',
    'value_loss',
    'kl_divergence',
    'compute_rlhf_reward',
    'policy_entropy_loss',
    'total_ppo_loss',
    'check_ppo_ratio',
    # GAE
    'compute_gae',
    'compute_returns',
    'normalize_advantages',
    'whiten_advantages',
    'compute_value_targets',
    'check_gae_health',
    # Buffer
    'RolloutBatch',
    'RolloutBuffer',
    'create_rollout_batch',
]
