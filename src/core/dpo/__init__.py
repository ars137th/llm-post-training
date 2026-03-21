"""Direct Preference Optimization (DPO) module."""

from .trainer import DPOTrainer
from .multimodal_trainer import MultimodalDPOTrainer, create_multimodal_dpo_trainer
from .loss import dpo_loss, ipo_loss, dpo_metrics, compute_sequence_log_probs

__all__ = [
    'DPOTrainer',
    'MultimodalDPOTrainer',
    'create_multimodal_dpo_trainer',
    'dpo_loss',
    'ipo_loss',
    'dpo_metrics',
    'compute_sequence_log_probs',
]
