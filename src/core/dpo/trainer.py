"""
DPO Trainer

Custom trainer for Direct Preference Optimization from preference data.
"""

from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, PreTrainedModel
from transformers.trainer_utils import EvalPrediction
import numpy as np
import time

from .loss import (
    compute_sequence_log_probs,
    dpo_loss,
    dpo_metrics,
    ipo_loss,
)


class DPOTrainer(Trainer):
    """
    Custom trainer for DPO (Direct Preference Optimization).

    Extends HuggingFace Trainer with:
    - Policy and reference model handling
    - DPO loss computation
    - Implicit reward tracking
    - KL divergence monitoring

    Example:
        >>> from src.models.language import LanguageModel
        >>> from transformers import TrainingArguments
        >>>
        >>> # Load SFT model
        >>> sft_model = LanguageModel.from_pretrained("gpt2-sft")
        >>>
        >>> # Policy starts from SFT, reference is frozen copy
        >>> policy_model = sft_model
        >>> reference_model = LanguageModel.from_pretrained("gpt2-sft")
        >>> reference_model.eval()  # Freeze
        >>>
        >>> args = TrainingArguments(output_dir="./outputs")
        >>> trainer = DPOTrainer(
        ...     model=policy_model,
        ...     ref_model=reference_model,
        ...     args=args,
        ...     train_dataset=train_data,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
        beta: float = 0.1,
        loss_type: str = "dpo",  # "dpo" or "ipo"
        log_rewards: bool = True,
        num_rewards_to_log: int = 5,
        **kwargs,
    ):
        """
        Initialize DPO trainer.

        Args:
            model: Policy model to train (starts from SFT)
            ref_model: Reference model (frozen copy of SFT)
            args: Training arguments
            train_dataset: Training dataset with preference pairs
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            compute_metrics: Custom metrics function
            beta: Temperature parameter for DPO loss (default: 0.1)
            loss_type: "dpo" or "ipo"
            log_rewards: Whether to log implicit reward values
            num_rewards_to_log: Number of reward examples to log
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs,
        )

        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        self.log_rewards = log_rewards
        self.num_rewards_to_log = num_rewards_to_log

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Track training metrics
        self.training_metrics = {
            'steps': [],
            'losses': [],
            'accuracies': [],
            'reward_margins': [],
            'chosen_kls': [],
            'rejected_kls': [],
        }

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,  # transformers 4.36+
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute DPO loss for preference pairs.

        Args:
            model: Policy model (being trained)
            inputs: Batch with chosen/rejected pairs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items (for transformers 4.36+)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Forward pass for chosen responses (policy)
        outputs_chosen = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask'],
            use_cache=False,
        )

        # Forward pass for rejected responses (policy)
        outputs_rejected = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask'],
            use_cache=False,
        )

        # Forward pass for chosen responses (reference)
        with torch.no_grad():
            ref_outputs_chosen = self.ref_model(
                input_ids=inputs['chosen_input_ids'],
                attention_mask=inputs['chosen_attention_mask'],
                use_cache=False,
            )

            # Forward pass for rejected responses (reference)
            ref_outputs_rejected = self.ref_model(
                input_ids=inputs['rejected_input_ids'],
                attention_mask=inputs['rejected_attention_mask'],
                use_cache=False,
            )

        # Compute log probabilities
        policy_chosen_logps = compute_sequence_log_probs(
            outputs_chosen.logits,
            inputs['chosen_input_ids'],
            inputs['chosen_attention_mask'],
        )

        policy_rejected_logps = compute_sequence_log_probs(
            outputs_rejected.logits,
            inputs['rejected_input_ids'],
            inputs['rejected_attention_mask'],
        )

        reference_chosen_logps = compute_sequence_log_probs(
            ref_outputs_chosen.logits,
            inputs['chosen_input_ids'],
            inputs['chosen_attention_mask'],
        )

        reference_rejected_logps = compute_sequence_log_probs(
            ref_outputs_rejected.logits,
            inputs['rejected_input_ids'],
            inputs['rejected_attention_mask'],
        )

        # Compute DPO or IPO loss
        if self.loss_type == "ipo":
            loss, details = ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.beta,
                return_details=True,
            )
        else:  # dpo
            loss, details = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.beta,
                return_details=True,
            )

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train/loss': loss.item(),
                'train/accuracy': details['accuracy'],
                'train/reward_margin': details['reward_margin'],
                'train/reward_chosen_mean': details['reward_chosen_mean'],
                'train/reward_rejected_mean': details['reward_rejected_mean'],
                'train/chosen_kl': details['chosen_kl'],
                'train/rejected_kl': details['rejected_kl'],
            })

        # Optionally log sample rewards
        if self.log_rewards and self.state.global_step % (self.args.logging_steps * 5) == 0:
            self._log_sample_rewards(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

        outputs = {
            'policy_chosen_logps': policy_chosen_logps,
            'policy_rejected_logps': policy_rejected_logps,
            'reference_chosen_logps': reference_chosen_logps,
            'reference_rejected_logps': reference_rejected_logps,
            **details,
        }

        if return_outputs:
            return loss, outputs
        else:
            return loss

    def _log_sample_rewards(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ):
        """
        Log sample implicit reward values for debugging.

        Args:
            policy_chosen_logps: Log probs from policy for chosen
            policy_rejected_logps: Log probs from policy for rejected
            reference_chosen_logps: Log probs from reference for chosen
            reference_rejected_logps: Log probs from reference for rejected
        """
        num_to_log = min(self.num_rewards_to_log, len(policy_chosen_logps))

        # Compute implicit rewards: r = β * log(π/π_ref)
        with torch.no_grad():
            reward_chosen = self.beta * (policy_chosen_logps - reference_chosen_logps)
            reward_rejected = self.beta * (policy_rejected_logps - reference_rejected_logps)

        log_dict = {}
        for i in range(num_to_log):
            log_dict[f'train/sample_{i}/reward_chosen'] = reward_chosen[i].item()
            log_dict[f'train/sample_{i}/reward_rejected'] = reward_rejected[i].item()
            log_dict[f'train/sample_{i}/margin'] = (
                reward_chosen[i] - reward_rejected[i]
            ).item()

        self.log(log_dict)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate DPO model on preference pairs.

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
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prediction step for evaluation.

        Args:
            model: Policy model
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

        # Predictions are the implicit rewards
        # [batch_size, 2] where [:, 0] = chosen, [:, 1] = rejected
        reward_chosen = self.beta * (
            outputs['policy_chosen_logps'] - outputs['reference_chosen_logps']
        )
        reward_rejected = self.beta * (
            outputs['policy_rejected_logps'] - outputs['reference_rejected_logps']
        )

        predictions = torch.stack([reward_chosen, reward_rejected], dim=1)

        # Labels: 1 if chosen > rejected, 0 otherwise
        labels = (reward_chosen > reward_rejected).long()

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
            self.training_metrics['reward_margins'].append(logs['train/reward_margin'])
        if 'train/chosen_kl' in logs:
            self.training_metrics['chosen_kls'].append(logs['train/chosen_kl'])
        if 'train/rejected_kl' in logs:
            self.training_metrics['rejected_kls'].append(logs['train/rejected_kl'])

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

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        """
        Override to prevent removal of our custom columns.

        The parent Trainer removes columns that don't match model signature,
        but we use custom column names (chosen_input_ids, rejected_input_ids, etc.)
        that are handled in compute_loss().

        Args:
            dataset: Dataset to process
            description: Description for logging

        Returns:
            Dataset unchanged (don't remove any columns)
        """
        # Don't remove any columns - we handle them in compute_loss()
        return dataset


def compute_dpo_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for DPO model.

    Args:
        eval_pred: Predictions and labels from evaluation

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = compute_dpo_metrics(eval_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    predictions = eval_pred.predictions  # [batch_size, 2] (chosen, rejected)
    labels = eval_pred.label_ids  # [batch_size]

    # Extract chosen and rejected implicit rewards
    reward_chosen = predictions[:, 0]
    reward_rejected = predictions[:, 1]

    # Convert to tensors
    reward_chosen = torch.from_numpy(reward_chosen)
    reward_rejected = torch.from_numpy(reward_rejected)

    # Compute accuracy
    accuracy = (reward_chosen > reward_rejected).float().mean()

    # Compute margins
    margin = (reward_chosen - reward_rejected).mean()

    # Compute reward statistics
    metrics = {
        'accuracy': accuracy.item(),
        'margin_mean': margin.item(),
        'margin_std': (reward_chosen - reward_rejected).std().item(),
        'chosen_mean': reward_chosen.mean().item(),
        'chosen_std': reward_chosen.std().item(),
        'rejected_mean': reward_rejected.mean().item(),
        'rejected_std': reward_rejected.std().item(),
    }

    return metrics
