"""
Multimodal DPO Trainer

Extends DPO to handle vision-language models (CLIP, LLaVA).
Trains models to prefer chosen image-text pairs over rejected ones.
"""

from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import TrainingArguments, PreTrainedModel

from .trainer import DPOTrainer
from .loss import compute_sequence_log_probs, dpo_loss, dpo_metrics


class MultimodalDPOTrainer(DPOTrainer):
    """
    DPO trainer for vision-language models.

    Extends DPOTrainer to handle multimodal preference pairs.
    Works with CLIP (contrastive) and LLaVA (generative) models.

    Example:
        >>> from src.models.vision_language import create_vision_language_model
        >>>
        >>> # Load SFT model and create reference copy
        >>> policy_model = create_vision_language_model("llava", "llava-hf/llava-1.5-7b-hf")
        >>> ref_model = create_vision_language_model("llava", "llava-hf/llava-1.5-7b-hf")
        >>> ref_model.model.eval()
        >>>
        >>> trainer = MultimodalDPOTrainer(
        ...     model=policy_model.model,
        ...     ref_model=ref_model.model,
        ...     args=training_args,
        ...     train_dataset=preference_pairs,
        ...     model_type="llava",
        ... )
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
        model_type: str = "llava",  # "clip" or "llava"
        beta: float = 0.1,
        loss_type: str = "dpo",
        log_rewards: bool = True,
        num_rewards_to_log: int = 5,
        **kwargs,
    ):
        """
        Initialize multimodal DPO trainer.

        Args:
            model: Policy model to train
            ref_model: Reference model (frozen)
            args: Training arguments
            train_dataset: Dataset with multimodal preference pairs
            eval_dataset: Evaluation dataset
            data_collator: Multimodal DPO data collator
            compute_metrics: Custom metrics function
            model_type: "clip" or "llava"
            beta: DPO temperature parameter
            loss_type: "dpo" or "ipo"
            log_rewards: Whether to log implicit rewards
            num_rewards_to_log: Number of rewards to log
            **kwargs: Additional trainer arguments
        """
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            beta=beta,
            loss_type=loss_type,
            log_rewards=log_rewards,
            num_rewards_to_log=num_rewards_to_log,
            **kwargs,
        )

        self.model_type = model_type.lower()

        if self.model_type not in ["clip", "llava"]:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute DPO loss for multimodal preference pairs.

        For CLIP:
            - Computes contrastive similarity scores for chosen/rejected
            - Uses similarity as "log probability"

        For LLaVA:
            - Standard DPO with causal language modeling
            - Computes log probs for chosen/rejected responses

        Args:
            model: Policy model (being trained)
            inputs: Batch with chosen/rejected multimodal pairs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items (transformers 4.36+)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        if self.model_type == "clip":
            # CLIP DPO: Preference learning on image-text alignment
            loss, details = self._compute_clip_dpo_loss(model, inputs)
        else:  # llava
            # LLaVA DPO: Standard DPO on generated captions
            loss, details = self._compute_llava_dpo_loss(model, inputs)

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'train/loss': loss.item(),
                'train/accuracy': details['accuracy'],
                'train/reward_margin': details['reward_margin'],
                'train/chosen_rewards': details['chosen_rewards_mean'],
                'train/rejected_rewards': details['rejected_rewards_mean'],
            })

        outputs = details

        if return_outputs:
            return loss, outputs
        else:
            return loss

    def _compute_clip_dpo_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss for CLIP using contrastive similarity.

        CLIP DPO learns to prefer chosen image-text pairs over rejected ones
        by maximizing the difference in their similarity scores.

        Args:
            model: Policy CLIP model
            inputs: Batch with chosen/rejected pixel_values and input_ids

        Returns:
            Tuple of (loss, details_dict)
        """
        # === Policy Forward Pass ===

        # Chosen: Get embeddings separately (avoid LoRA routing issues)
        policy_chosen_image = model.get_image_features(
            pixel_values=inputs['chosen_pixel_values']
        )
        policy_chosen_text = model.get_text_features(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask'],
        )

        # Rejected: Get embeddings
        policy_rejected_image = model.get_image_features(
            pixel_values=inputs['rejected_pixel_values']
        )
        policy_rejected_text = model.get_text_features(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask'],
        )

        # Normalize embeddings
        policy_chosen_image = policy_chosen_image / policy_chosen_image.norm(dim=-1, keepdim=True)
        policy_chosen_text = policy_chosen_text / policy_chosen_text.norm(dim=-1, keepdim=True)
        policy_rejected_image = policy_rejected_image / policy_rejected_image.norm(dim=-1, keepdim=True)
        policy_rejected_text = policy_rejected_text / policy_rejected_text.norm(dim=-1, keepdim=True)

        # Compute similarities (use as "log probs" for DPO)
        policy_chosen_sim = (policy_chosen_image * policy_chosen_text).sum(dim=-1)
        policy_rejected_sim = (policy_rejected_image * policy_rejected_text).sum(dim=-1)

        # === Reference Forward Pass ===
        with torch.no_grad():
            # Chosen
            ref_chosen_image = self.ref_model.get_image_features(
                pixel_values=inputs['chosen_pixel_values']
            )
            ref_chosen_text = self.ref_model.get_text_features(
                input_ids=inputs['chosen_input_ids'],
                attention_mask=inputs['chosen_attention_mask'],
            )

            # Rejected
            ref_rejected_image = self.ref_model.get_image_features(
                pixel_values=inputs['rejected_pixel_values']
            )
            ref_rejected_text = self.ref_model.get_text_features(
                input_ids=inputs['rejected_input_ids'],
                attention_mask=inputs['rejected_attention_mask'],
            )

            # Normalize
            ref_chosen_image = ref_chosen_image / ref_chosen_image.norm(dim=-1, keepdim=True)
            ref_chosen_text = ref_chosen_text / ref_chosen_text.norm(dim=-1, keepdim=True)
            ref_rejected_image = ref_rejected_image / ref_rejected_image.norm(dim=-1, keepdim=True)
            ref_rejected_text = ref_rejected_text / ref_rejected_text.norm(dim=-1, keepdim=True)

            # Similarities
            ref_chosen_sim = (ref_chosen_image * ref_chosen_text).sum(dim=-1)
            ref_rejected_sim = (ref_rejected_image * ref_rejected_text).sum(dim=-1)

        # Use similarities as log probabilities for DPO
        # (CLIP similarity is already in [-1, 1] range, scale it for DPO)
        scale = 10.0  # Scale factor to make similarities behave like log probs
        policy_chosen_logprob = policy_chosen_sim * scale
        policy_rejected_logprob = policy_rejected_sim * scale
        ref_chosen_logprob = ref_chosen_sim * scale
        ref_rejected_logprob = ref_rejected_sim * scale

        # Compute DPO loss
        loss, details = dpo_loss(
            policy_chosen_logps=policy_chosen_logprob,
            policy_rejected_logps=policy_rejected_logprob,
            reference_chosen_logps=ref_chosen_logprob,
            reference_rejected_logps=ref_rejected_logprob,
            beta=self.beta,
            return_details=True,
        )

        return loss, details

    def _compute_llava_dpo_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss for LLaVA using causal language modeling.

        LLaVA DPO learns to generate preferred captions given an image.

        Args:
            model: Policy LLaVA model
            inputs: Batch with chosen/rejected multimodal pairs

        Returns:
            Tuple of (loss, details_dict)
        """
        # === Policy Forward Pass ===

        # Chosen responses
        outputs_chosen = model(
            pixel_values=inputs['chosen_pixel_values'],
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask'],
            use_cache=False,
        )

        # Rejected responses
        outputs_rejected = model(
            pixel_values=inputs['rejected_pixel_values'],
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask'],
            use_cache=False,
        )

        # Compute log probabilities
        policy_chosen_logps = compute_sequence_log_probs(
            logits=outputs_chosen.logits,
            labels=inputs['chosen_labels'],
            attention_mask=inputs['chosen_attention_mask'],
        )

        policy_rejected_logps = compute_sequence_log_probs(
            logits=outputs_rejected.logits,
            labels=inputs['rejected_labels'],
            attention_mask=inputs['rejected_attention_mask'],
        )

        # === Reference Forward Pass ===
        with torch.no_grad():
            ref_outputs_chosen = self.ref_model(
                pixel_values=inputs['chosen_pixel_values'],
                input_ids=inputs['chosen_input_ids'],
                attention_mask=inputs['chosen_attention_mask'],
                use_cache=False,
            )

            ref_outputs_rejected = self.ref_model(
                pixel_values=inputs['rejected_pixel_values'],
                input_ids=inputs['rejected_input_ids'],
                attention_mask=inputs['rejected_attention_mask'],
                use_cache=False,
            )

            ref_chosen_logps = compute_sequence_log_probs(
                logits=ref_outputs_chosen.logits,
                labels=inputs['chosen_labels'],
                attention_mask=inputs['chosen_attention_mask'],
            )

            ref_rejected_logps = compute_sequence_log_probs(
                logits=ref_outputs_rejected.logits,
                labels=inputs['rejected_labels'],
                attention_mask=inputs['rejected_attention_mask'],
            )

        # Compute DPO loss
        loss, details = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            beta=self.beta,
            return_details=True,
        )

        return loss, details


def create_multimodal_dpo_trainer(
    policy_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    model_type: str,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    beta: float = 0.1,
    **kwargs,
):
    """
    Factory function to create multimodal DPO trainer.

    Args:
        policy_model: Model to train
        ref_model: Reference model (frozen)
        model_type: "clip" or "llava"
        training_args: HuggingFace TrainingArguments
        train_dataset: Dataset with preference pairs
        eval_dataset: Evaluation dataset
        data_collator: Multimodal DPO data collator
        beta: DPO temperature parameter
        **kwargs: Additional trainer arguments

    Returns:
        MultimodalDPOTrainer instance

    Example:
        trainer = create_multimodal_dpo_trainer(
            policy_model=policy_clip,
            ref_model=ref_clip,
            model_type="clip",
            training_args=args,
            train_dataset=preference_data,
            data_collator=collator,
            beta=0.1,
        )
        trainer.train()
    """
    return MultimodalDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_type=model_type,
        beta=beta,
        **kwargs,
    )
