"""Supervised Fine-Tuning (SFT) module."""

from .trainer import SFTTrainer, compute_sft_metrics
from .multimodal_trainer import MultimodalSFTTrainer, create_multimodal_trainer
from .loss import CausalLMLoss, FocalLoss, compute_token_accuracy, compute_perplexity

__all__ = [
    'SFTTrainer',
    'compute_sft_metrics',
    'MultimodalSFTTrainer',
    'create_multimodal_trainer',
    'CausalLMLoss',
    'FocalLoss',
    'compute_token_accuracy',
    'compute_perplexity',
]
