"""
Supervised Fine-Tuning Trainer

Custom trainer for SFT with detailed logging and educational features.
Extends HuggingFace Trainer with:
- Custom loss computation with detailed metrics
- Token-level accuracy tracking
- Gradient norm monitoring
- Sample generation during training
- Learning rate scheduling
"""

from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
import numpy as np
from dataclasses import dataclass
import time

from .loss import CausalLMLoss, FocalLoss, compute_token_accuracy, compute_perplexity


class SFTTrainer(Trainer):
    """
    Custom trainer for supervised fine-tuning.

    Extends HuggingFace Trainer with:
    - Custom loss functions (CausalLM or Focal)
    - Detailed metrics logging (accuracy, perplexity, gradient norms)
    - Sample generation during training for qualitative evaluation
    - Educational logging for understanding training dynamics

    Example:
        >>> from transformers import TrainingArguments
        >>> from src.models.language import LanguageModel
        >>>
        >>> model = LanguageModel.from_pretrained("gpt2", use_lora=True)
        >>> args = TrainingArguments(output_dir="./outputs")
        >>>
        >>> trainer = SFTTrainer(
        ...     model=model.model,
        ...     args=args,
        ...     train_dataset=train_data,
        ...     loss_type="causal_lm",
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        loss_type: str = "causal_lm",
        loss_kwargs: Optional[Dict] = None,
        log_predictions: bool = True,
        num_predictions_to_log: int = 3,
        **kwargs,
    ):
        """
        Initialize SFT trainer.

        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer (for generating samples)
            data_collator: Data collator
            compute_metrics: Custom metrics function
            loss_type: Type of loss ("causal_lm" or "focal")
            loss_kwargs: Additional arguments for loss function
            log_predictions: Whether to log sample predictions during training
            num_predictions_to_log: Number of predictions to log
            **kwargs: Additional arguments for Trainer
        """
        # Store tokenizer separately (not passed to parent in transformers 4.36+)
        self.tokenizer = tokenizer

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs,
        )

        # Setup custom loss
        loss_kwargs = loss_kwargs or {}
        if loss_type == "causal_lm":
            self.loss_fn = CausalLMLoss(**loss_kwargs)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.log_predictions = log_predictions
        self.num_predictions_to_log = num_predictions_to_log

        # Track training metrics
        self.training_metrics = {
            'steps': [],
            'losses': [],
            'accuracies': [],
            'perplexities': [],
            'grad_norms': [],
            'learning_rates': [],
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss with detailed metrics.

        This method is called by the Trainer during training and evaluation.

        Args:
            model: The model
            inputs: Input batch with 'input_ids', 'attention_mask', 'labels'
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        labels = inputs['labels']

        # Compute loss with details
        loss, details = self.loss_fn(logits, labels, return_details=True)

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train/loss': loss.item(),
                'train/accuracy': details['accuracy'],
                'train/perplexity': details['perplexity'],
                'train/num_tokens': details['num_tokens'],
            })

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform a training step with additional logging.

        Args:
            model: The model
            inputs: Input batch

        Returns:
            Loss tensor
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward and backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Backward pass
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Log gradient norms (educational)
        if self.state.global_step % self.args.logging_steps == 0:
            grad_norm = self._compute_grad_norm()
            if grad_norm is not None:
                self.log({'train/grad_norm': grad_norm})

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log({'train/learning_rate': current_lr})

        return loss.detach()

    def _compute_grad_norm(self) -> Optional[float]:
        """
        Compute the norm of gradients (for monitoring training stability).

        Returns:
            Gradient norm as a float, or None if no gradients
        """
        total_norm = 0.0
        parameters = [
            p for p in self.model.parameters()
            if p.grad is not None and p.requires_grad
        ]

        if not parameters:
            return None

        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        return total_norm

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate model and optionally generate samples.

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

        # Generate samples for qualitative evaluation
        if self.log_predictions and self.tokenizer is not None:
            self._log_predictions(prefix=metric_key_prefix)

        return metrics

    def _log_predictions(self, prefix: str = "eval"):
        """
        Generate and log sample predictions.

        This helps understand what the model is learning qualitatively.

        Args:
            prefix: Prefix for logging (e.g., "eval" or "train")
        """
        if not hasattr(self, 'eval_dataset') or self.eval_dataset is None:
            return

        self.model.eval()

        # Sample a few examples
        num_samples = min(self.num_predictions_to_log, len(self.eval_dataset))
        indices = np.random.choice(len(self.eval_dataset), num_samples, replace=False)

        samples = []
        for idx in indices:
            example = self.eval_dataset[int(idx)]

            # Get input
            input_ids = torch.tensor([example['input_ids']]).to(self.args.device)
            attention_mask = torch.tensor([example['attention_mask']]).to(self.args.device)

            # Truncate to get prompt only (first 50% of sequence)
            prompt_len = max(1, len(example['input_ids']) // 2)
            input_ids = input_ids[:, :prompt_len]
            attention_mask = attention_mask[:, :prompt_len]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            samples.append({
                'prompt': prompt_text,
                'generated': generated_text,
            })

        # Log to wandb/tensorboard if available
        if self.args.report_to and samples:
            log_dict = {
                f'{prefix}/sample_{i}/prompt': s['prompt']
                for i, s in enumerate(samples)
            }
            log_dict.update({
                f'{prefix}/sample_{i}/generated': s['generated']
                for i, s in enumerate(samples)
            })
            self.log(log_dict)

        self.model.train()

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log metrics with timestamp.

        Args:
            logs: Dictionary of metrics to log
        """
        # Add timestamp
        logs['timestamp'] = time.time()

        # Track training metrics
        if 'train/loss' in logs:
            self.training_metrics['steps'].append(self.state.global_step)
            self.training_metrics['losses'].append(logs['train/loss'])
        if 'train/accuracy' in logs:
            self.training_metrics['accuracies'].append(logs['train/accuracy'])
        if 'train/perplexity' in logs:
            self.training_metrics['perplexities'].append(logs['train/perplexity'])
        if 'train/grad_norm' in logs:
            self.training_metrics['grad_norms'].append(logs['train/grad_norm'])
        if 'train/learning_rate' in logs:
            self.training_metrics['learning_rates'].append(logs['train/learning_rate'])

        super().log(logs)

    def get_training_metrics(self) -> Dict[str, List[float]]:
        """
        Get collected training metrics for analysis.

        Returns:
            Dictionary of training metrics over time
        """
        return self.training_metrics


def compute_sft_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for SFT.

    Args:
        eval_pred: Predictions and labels from evaluation

    Returns:
        Dictionary of metrics
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert to tensors if needed
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels)

    # Compute accuracy
    accuracy = compute_token_accuracy(logits, labels)

    # Compute loss for perplexity
    loss_fn = CausalLMLoss()
    loss, details = loss_fn(logits, labels, return_details=True)

    return {
        'accuracy': accuracy,
        'perplexity': details['perplexity'],
        'num_tokens': details['num_tokens'],
    }
