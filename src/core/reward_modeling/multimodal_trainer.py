"""
Multimodal Reward Model Trainer

Extends RewardModelTrainer to handle vision-language preference pairs.
Used for training reward models on multimodal data (image + text).
"""

from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import TrainingArguments

from .trainer import RewardModelTrainer
from .loss import bradley_terry_loss
from ...models.reward import RewardModel


class MultimodalRewardModelTrainer(RewardModelTrainer):
    """
    Reward model trainer for vision-language models.

    Extends RewardModelTrainer to handle multimodal preference pairs.
    Used for RLHF with vision-language models like CLIP and LLaVA.

    Example:
        >>> from src.models.reward import RewardModel
        >>> from src.data.collators.multimodal import MultimodalDataCollator
        >>>
        >>> # Create reward model from vision-language base
        >>> reward_model = RewardModel.from_vision_language_model(
        ...     "llava-hf/llava-1.5-7b-hf"
        ... )
        >>>
        >>> trainer = MultimodalRewardModelTrainer(
        ...     model=reward_model,
        ...     args=training_args,
        ...     train_dataset=preference_pairs,
        ...     model_type="llava",
        ... )
    """

    def __init__(
        self,
        model: RewardModel,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
        model_type: str = "clip",  # "clip" or "llava"
        margin: float = 0.0,
        log_rewards: bool = True,
        num_rewards_to_log: int = 5,
        **kwargs,
    ):
        """
        Initialize multimodal reward model trainer.

        Args:
            model: RewardModel (with vision-language base)
            args: Training arguments
            train_dataset: Dataset with multimodal preference pairs
            eval_dataset: Evaluation dataset
            data_collator: Multimodal data collator
            compute_metrics: Custom metrics function
            model_type: "clip" or "llava"
            margin: Margin for Bradley-Terry loss
            log_rewards: Whether to log reward values
            num_rewards_to_log: Number of rewards to log
            **kwargs: Additional trainer arguments
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            margin=margin,
            log_rewards=log_rewards,
            num_rewards_to_log=num_rewards_to_log,
            **kwargs,
        )

        self.model_type = model_type.lower()

        if self.model_type not in ["clip", "llava"]:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def compute_loss(
        self,
        model: RewardModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute Bradley-Terry loss for multimodal preference pairs.

        Expected inputs:
            - chosen_pixel_values: Images for chosen responses
            - chosen_input_ids: Text for chosen responses
            - chosen_attention_mask: Attention mask for chosen
            - rejected_pixel_values: Images for rejected responses
            - rejected_input_ids: Text for rejected responses
            - rejected_attention_mask: Attention mask for rejected

        Args:
            model: RewardModel with vision-language base
            inputs: Batch with chosen/rejected multimodal pairs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items (transformers 4.36+)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Forward pass for chosen responses (with images)
        chosen_inputs = {
            'input_ids': inputs['chosen_input_ids'],
            'attention_mask': inputs['chosen_attention_mask'],
        }

        if 'chosen_pixel_values' in inputs:
            chosen_inputs['pixel_values'] = inputs['chosen_pixel_values']

        rewards_chosen = model(
            **chosen_inputs,
            return_dict=False,
        )

        # Forward pass for rejected responses (with images)
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'],
            'attention_mask': inputs['rejected_attention_mask'],
        }

        if 'rejected_pixel_values' in inputs:
            rejected_inputs['pixel_values'] = inputs['rejected_pixel_values']

        rewards_rejected = model(
            **rejected_inputs,
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


@dataclass
class MultimodalPreferenceDataCollator:
    """
    Data collator for multimodal preference pairs.

    Batches (chosen_image, chosen_text) and (rejected_image, rejected_text) pairs.

    Example:
        collator = MultimodalPreferenceDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        batch = collator(preference_examples)
        # Returns: {
        #   'chosen_pixel_values', 'chosen_input_ids', 'chosen_attention_mask',
        #   'rejected_pixel_values', 'rejected_input_ids', 'rejected_attention_mask'
        # }
    """

    tokenizer: Any
    image_processor: Any
    max_length: int = 512
    padding: str = "max_length"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate multimodal preference pairs.

        Args:
            examples: List of dicts with:
                - 'chosen_image': PIL Image for chosen
                - 'chosen_text': Text for chosen
                - 'rejected_image': PIL Image for rejected
                - 'rejected_text': Text for rejected

        Returns:
            Batch dict with chosen/rejected pixel_values, input_ids, attention_mask
        """
        # Separate chosen and rejected
        chosen_images = [ex['chosen_image'] for ex in examples]
        chosen_texts = [ex['chosen_text'] for ex in examples]
        rejected_images = [ex['rejected_image'] for ex in examples]
        rejected_texts = [ex['rejected_text'] for ex in examples]

        # Process chosen images
        chosen_image_inputs = self.image_processor(
            images=chosen_images,
            return_tensors="pt",
        )

        # Process rejected images
        rejected_image_inputs = self.image_processor(
            images=rejected_images,
            return_tensors="pt",
        )

        # Process chosen texts
        chosen_text_inputs = self.tokenizer(
            chosen_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Process rejected texts
        rejected_text_inputs = self.tokenizer(
            rejected_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            'chosen_pixel_values': chosen_image_inputs['pixel_values'],
            'chosen_input_ids': chosen_text_inputs['input_ids'],
            'chosen_attention_mask': chosen_text_inputs['attention_mask'],
            'rejected_pixel_values': rejected_image_inputs['pixel_values'],
            'rejected_input_ids': rejected_text_inputs['input_ids'],
            'rejected_attention_mask': rejected_text_inputs['attention_mask'],
        }


def create_multimodal_reward_trainer(
    model: RewardModel,
    model_type: str,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    **kwargs,
):
    """
    Factory function to create multimodal reward model trainer.

    Args:
        model: RewardModel with vision-language base
        model_type: "clip" or "llava"
        training_args: HuggingFace TrainingArguments
        train_dataset: Dataset with preference pairs
        eval_dataset: Evaluation dataset
        data_collator: Multimodal preference data collator
        **kwargs: Additional trainer arguments

    Returns:
        MultimodalRewardModelTrainer instance

    Example:
        trainer = create_multimodal_reward_trainer(
            model=reward_model,
            model_type="llava",
            training_args=training_args,
            train_dataset=preference_data,
            data_collator=collator,
        )
        trainer.train()
    """
    return MultimodalRewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_type=model_type,
        **kwargs,
    )


# Import dataclass
from dataclasses import dataclass
from typing import Any
