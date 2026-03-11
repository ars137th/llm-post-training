"""
Reward Model Trainer

Custom trainer for training reward models from preference data.
"""

from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
import numpy as np
import time

from .loss import (
    bradley_terry_loss,
    compute_ranking_accuracy,
    compute_reward_margin,
    compute_reward_statistics,
)
from ...models.reward import RewardModel


class RewardModelTrainer(Trainer):
    """
    Custom trainer for reward models.

    Extends HuggingFace Trainer with:
    - Bradley-Terry loss computation
    - Ranking accuracy tracking
    - Reward statistics logging
    - Preference pair handling

    Example:
        >>> from src.models.reward import RewardModel
        >>> from transformers import TrainingArguments
        >>>
        >>> reward_model = RewardModel.from_pretrained("gpt2")
        >>> args = TrainingArguments(output_dir="./outputs")
        >>>
        >>> trainer = RewardModelTrainer(
        ...     model=reward_model,
        ...     args=args,
        ...     train_dataset=train_data,
        ...     eval_dataset=eval_data,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: RewardModel,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
        margin: float = 0.0,
        log_rewards: bool = True,
        num_rewards_to_log: int = 5,
        **kwargs,
    ):
        """
        Initialize reward model trainer.

        Args:
            model: RewardModel to train
            args: Training arguments
            train_dataset: Training dataset with preference pairs
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            compute_metrics: Custom metrics function
            margin: Margin for Bradley-Terry loss
            log_rewards: Whether to log reward values during training
            num_rewards_to_log: Number of reward examples to log
            **kwargs: Additional arguments for Trainer
        """
        # The model passed should be the RewardModel wrapper
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs,
        )

        self.margin = margin
        self.log_rewards = log_rewards
        self.num_rewards_to_log = num_rewards_to_log

        # Track training metrics
        self.training_metrics = {
            'steps': [],
            'losses': [],
            'accuracies': [],
            'margins': [],
            'reward_chosen_means': [],
            'reward_rejected_means': [],
        }

    def compute_loss(
        self,
        model: RewardModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,  # transformers 4.36+
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute Bradley-Terry loss for preference pairs.

        Args:
            model: RewardModel
            inputs: Batch with chosen/rejected pairs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items (for transformers 4.36+)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Forward pass for chosen responses
        rewards_chosen = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask'],
            return_dict=False,
        )

        # Forward pass for rejected responses
        rewards_rejected = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask'],
            return_dict=False,
        )

        # Compute Bradley-Terry loss
        loss, details = bradley_terry_loss(
            rewards_chosen,
            rewards_rejected,
            margin=self.margin,
            return_details=True,
        )

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train/loss': loss.item(),
                'train/accuracy': details['accuracy'],
                'train/reward_chosen_mean': details['reward_chosen_mean'],
                'train/reward_rejected_mean': details['reward_rejected_mean'],
                'train/reward_margin': details['reward_margin_mean'],
                'train/reward_chosen_std': details['reward_chosen_std'],
                'train/reward_rejected_std': details['reward_rejected_std'],
            })

        # Optionally log sample rewards
        if self.log_rewards and self.state.global_step % (self.args.logging_steps * 5) == 0:
            self._log_sample_rewards(rewards_chosen, rewards_rejected)

        outputs = {
            'rewards_chosen': rewards_chosen,
            'rewards_rejected': rewards_rejected,
            **details,
        }

        if return_outputs:
            return loss, outputs
        else:
            return loss

    def _log_sample_rewards(
        self,
        rewards_chosen: torch.Tensor,
        rewards_rejected: torch.Tensor,
    ):
        """
        Log sample reward values for debugging.

        Args:
            rewards_chosen: Rewards for chosen responses
            rewards_rejected: Rewards for rejected responses
        """
        num_to_log = min(self.num_rewards_to_log, len(rewards_chosen))

        log_dict = {}
        for i in range(num_to_log):
            log_dict[f'train/sample_{i}/reward_chosen'] = rewards_chosen[i].item()
            log_dict[f'train/sample_{i}/reward_rejected'] = rewards_rejected[i].item()
            log_dict[f'train/sample_{i}/margin'] = (
                rewards_chosen[i] - rewards_rejected[i]
            ).item()

        self.log(log_dict)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate reward model on preference pairs.

        Args:
            eval_dataset: Evaluation dataset
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary of evaluation metrics
        """
        # Standard evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        return metrics

    def prediction_step(
        self,
        model: RewardModel,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prediction step for evaluation.

        Args:
            model: RewardModel
            inputs: Input batch
            prediction_loss_only: Whether to only return loss
            ignore_keys: Keys to ignore

        Returns:
            (loss, predictions, labels) tuple
        """
        # Forward pass
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        # Predictions are the rewards
        predictions = torch.stack([
            outputs['rewards_chosen'],
            outputs['rewards_rejected']
        ], dim=1)  # [batch_size, 2]

        # Labels: 1 if chosen > rejected, 0 otherwise
        labels = (outputs['rewards_chosen'] > outputs['rewards_rejected']).long()

        return (loss, predictions, labels)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log metrics with timestamp.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time (for transformers 4.36+)
        """
        # Add timestamp
        logs['timestamp'] = time.time()

        # Track training metrics
        if 'train/loss' in logs:
            self.training_metrics['steps'].append(self.state.global_step)
            self.training_metrics['losses'].append(logs['train/loss'])
        if 'train/accuracy' in logs:
            self.training_metrics['accuracies'].append(logs['train/accuracy'])
        if 'train/reward_margin' in logs:
            self.training_metrics['margins'].append(logs['train/reward_margin'])
        if 'train/reward_chosen_mean' in logs:
            self.training_metrics['reward_chosen_means'].append(logs['train/reward_chosen_mean'])
        if 'train/reward_rejected_mean' in logs:
            self.training_metrics['reward_rejected_means'].append(logs['train/reward_rejected_mean'])

        # Call parent log with appropriate signature
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def get_training_metrics(self) -> Dict[str, List[float]]:
        """
        Get collected training metrics for analysis.

        Returns:
            Dictionary of training metrics over time
        """
        return self.training_metrics


def compute_reward_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for reward model.

    Args:
        eval_pred: Predictions and labels from evaluation

    Returns:
        Dictionary of metrics
    """
    predictions = eval_pred.predictions  # [batch_size, 2] (chosen, rejected)
    labels = eval_pred.label_ids  # [batch_size]

    # Extract chosen and rejected rewards
    rewards_chosen = predictions[:, 0]
    rewards_rejected = predictions[:, 1]

    # Convert to tensors
    rewards_chosen = torch.from_numpy(rewards_chosen)
    rewards_rejected = torch.from_numpy(rewards_rejected)

    # Compute ranking accuracy
    accuracy = compute_ranking_accuracy(rewards_chosen, rewards_rejected)

    # Compute reward margins
    margin_stats = compute_reward_margin(rewards_chosen, rewards_rejected)

    # Compute reward statistics
    chosen_stats = compute_reward_statistics(rewards_chosen, name="chosen")
    rejected_stats = compute_reward_statistics(rewards_rejected, name="rejected")

    # Combine all metrics
    metrics = {
        'accuracy': accuracy,
        **margin_stats,
        **chosen_stats,
        **rejected_stats,
    }

    return metrics


def evaluate_reward_model(
    model: RewardModel,
    eval_dataset,
    data_collator,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Evaluate reward model on test set.

    Standalone evaluation function (doesn't require Trainer).

    Args:
        model: RewardModel to evaluate
        eval_dataset: Evaluation dataset with preference pairs
        data_collator: Data collator for batching
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of evaluation metrics

    Example:
        >>> from src.data.processors.preference import PreferenceDataCollator
        >>>
        >>> collator = PreferenceDataCollator(model.tokenizer)
        >>> metrics = evaluate_reward_model(
        ...     model, eval_dataset, collator, batch_size=8
        ... )
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    from torch.utils.data import DataLoader

    model.eval()

    # Create dataloader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    all_rewards_chosen = []
    all_rewards_rejected = []

    # Evaluate
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward pass
            rewards_chosen = model(
                input_ids=batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask'],
                return_dict=False,
            )

            rewards_rejected = model(
                input_ids=batch['rejected_input_ids'],
                attention_mask=batch['rejected_attention_mask'],
                return_dict=False,
            )

            all_rewards_chosen.append(rewards_chosen.cpu())
            all_rewards_rejected.append(rewards_rejected.cpu())

    # Concatenate all batches
    rewards_chosen = torch.cat(all_rewards_chosen)
    rewards_rejected = torch.cat(all_rewards_rejected)

    # Compute metrics
    accuracy = compute_ranking_accuracy(rewards_chosen, rewards_rejected)
    margin_stats = compute_reward_margin(rewards_chosen, rewards_rejected)
    chosen_stats = compute_reward_statistics(rewards_chosen, name="chosen")
    rejected_stats = compute_reward_statistics(rewards_rejected, name="rejected")

    return {
        'accuracy': accuracy,
        **margin_stats,
        **chosen_stats,
        **rejected_stats,
    }
