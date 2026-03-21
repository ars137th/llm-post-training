"""
Reward Modeling Module

Components for training reward models from preference data.
"""

from .loss import (
    bradley_terry_loss,
    compute_ranking_accuracy,
    compute_reward_margin,
)
from .trainer import RewardModelTrainer
from .multimodal_trainer import (
    MultimodalRewardModelTrainer,
    MultimodalPreferenceDataCollator,
    create_multimodal_reward_trainer,
)

__all__ = [
    'bradley_terry_loss',
    'compute_ranking_accuracy',
    'compute_reward_margin',
    'RewardModelTrainer',
    'MultimodalRewardModelTrainer',
    'MultimodalPreferenceDataCollator',
    'create_multimodal_reward_trainer',
]
