"""
Multimodal SFT Trainer

Extends SFTTrainer to handle vision-language models (CLIP, LLaVA).
Key differences from text-only SFT:
- Handles pixel_values in addition to input_ids
- Computes multimodal-specific metrics (CLIP Score)
- Supports image+text batch processing
"""

from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import TrainingArguments
import numpy as np

from .trainer import SFTTrainer
from .loss import CausalLMLoss


class MultimodalSFTTrainer(SFTTrainer):
    """
    SFT Trainer for vision-language models.

    Extends SFTTrainer to handle multimodal inputs (images + text).
    Works with CLIP, LLaVA, and other vision-language architectures.

    Example:
        >>> from src.models.vision_language import create_vision_language_model
        >>> from src.data.collators.multimodal import CLIPDataCollator
        >>>
        >>> model = create_vision_language_model("clip", "openai/clip-vit-base-patch32")
        >>> collator = CLIPDataCollator(tokenizer=..., image_processor=...)
        >>>
        >>> trainer = MultimodalSFTTrainer(
        ...     model=model.model,
        ...     args=training_args,
        ...     train_dataset=train_data,
        ...     data_collator=collator,
        ...     model_type="clip",
        ... )
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
        model_type: str = "clip",  # "clip" or "llava"
        **kwargs,
    ):
        """
        Initialize multimodal SFT trainer.

        Args:
            model: Vision-language model to train
            args: Training arguments
            train_dataset: Training dataset (with images)
            eval_dataset: Evaluation dataset (with images)
            tokenizer: Tokenizer
            data_collator: Multimodal data collator
            compute_metrics: Custom metrics function
            model_type: "clip" or "llava"
            **kwargs: Additional arguments for SFTTrainer
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs,
        )

        self.model_type = model_type.lower()

        if self.model_type not in ["clip", "llava"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'clip' or 'llava'.")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss for multimodal inputs.

        For CLIP:
            - Inputs: pixel_values, input_ids
            - Loss: Contrastive loss between image and text embeddings

        For LLaVA:
            - Inputs: pixel_values, input_ids, labels
            - Loss: Causal language modeling loss (predict response tokens)

        Args:
            model: The model
            inputs: Input batch with 'pixel_values', 'input_ids', 'attention_mask'
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items (transformers 4.36+)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Compute loss based on model type
        if self.model_type == "clip":
            # CLIP: Call encoders separately to avoid LoRA routing issues
            # CRITICAL: PEFT wrapper adds unexpected kwargs even to text encoder
            # Solution: Call text_model directly, then apply text projection

            # Get image embeddings - vision model is not wrapped with PEFT
            vision_outputs = model.get_image_features(
                pixel_values=inputs['pixel_values']
            )

            # Get text embeddings - manually call text_model and projection
            # Bypass get_text_features() to avoid PEFT adding inputs_embeds
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')

            # Call text_model directly (this has PEFT wrapper)
            text_encoder_outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get the pooled output (same as get_text_features does)
            text_embeds = text_encoder_outputs[1]  # pooled_output

            # Apply text projection (same as get_text_features does)
            text_outputs = model.text_projection(text_embeds)

            # Create outputs object with embeddings
            class CLIPOutputs:
                def __init__(self, image_embeds, text_embeds, logit_scale):
                    self.image_embeds = image_embeds
                    self.text_embeds = text_embeds
                    self.logit_scale = logit_scale
                    self.loss = None

            # Get logit_scale from model (learnable temperature parameter)
            logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else 1.0
            outputs = CLIPOutputs(vision_outputs, text_outputs, logit_scale)

            # Compute contrastive loss
            loss = self._compute_clip_loss(outputs, inputs)

        else:  # llava
            # LLaVA: Full forward pass with pixel_values, input_ids, labels
            model_inputs = {
                'pixel_values': inputs['pixel_values'],
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs.get('attention_mask'),
            }
            if 'labels' in inputs:
                model_inputs['labels'] = inputs['labels']

            outputs = model(**model_inputs)

            # LLaVA: Causal language modeling loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                # Model computed loss (if labels were passed)
                loss = outputs.loss
            else:
                # Manual loss computation
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                labels = inputs['labels']
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

    def _compute_clip_loss(
        self,
        outputs: Any,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss.

        CLIP learns by maximizing similarity between matching image-text pairs
        and minimizing similarity for non-matching pairs.

        Args:
            outputs: Model outputs with image_embeds and text_embeds
            inputs: Input batch

        Returns:
            Contrastive loss
        """
        # Get embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logit_scale = outputs.logit_scale if hasattr(outputs, 'logit_scale') else 1.0
        logits_per_image = logit_scale * (image_embeds @ text_embeds.t())
        logits_per_text = logits_per_image.t()

        # Contrastive loss (cross-entropy with diagonal targets)
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        # Log CLIP-specific metrics
        if self.state.global_step % self.args.logging_steps == 0:
            # Compute accuracy (how often top-1 match is correct)
            with torch.no_grad():
                i2t_acc = (logits_per_image.argmax(dim=1) == labels).float().mean()
                t2i_acc = (logits_per_text.argmax(dim=1) == labels).float().mean()

            self.log({
                'train/loss': loss.item(),
                'train/i2t_accuracy': i2t_acc.item(),
                'train/t2i_accuracy': t2i_acc.item(),
                'train/logit_scale': logit_scale.item() if isinstance(logit_scale, torch.Tensor) else logit_scale,
            })

        return loss

    def _log_predictions(self, prefix: str = "eval"):
        """
        Generate and log sample predictions for multimodal models.

        For CLIP: Log image-text similarity scores
        For LLaVA: Generate captions/responses

        Args:
            prefix: Prefix for logging
        """
        if not hasattr(self, 'eval_dataset') or self.eval_dataset is None:
            return

        try:
            self.model.eval()

            if self.model_type == "llava":
                # For LLaVA, generate text responses
                super()._log_predictions(prefix=prefix)

            elif self.model_type == "clip":
                # For CLIP, compute and log similarity scores
                num_samples = min(self.num_predictions_to_log, len(self.eval_dataset))
                indices = np.random.choice(len(self.eval_dataset), num_samples, replace=False)

                samples = []
                for idx in indices:
                    try:
                        example = self.eval_dataset[int(idx)]

                        # Prepare inputs
                        pixel_values = example['pixel_values'].unsqueeze(0).to(self.args.device)
                        input_ids = example['input_ids'].unsqueeze(0).to(self.args.device)
                        attention_mask = example['attention_mask'].unsqueeze(0).to(self.args.device)

                        # Forward pass
                        with torch.no_grad():
                            outputs = self.model(
                                pixel_values=pixel_values,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )

                        # Compute similarity
                        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                        similarity = (image_embeds * text_embeds).sum().item()

                        # Decode text
                        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

                        samples.append({
                            'text': text,
                            'similarity': similarity,
                        })

                    except Exception as e:
                        continue

                # Log samples
                if self.args.report_to and samples:
                    log_dict = {}
                    for i, s in enumerate(samples):
                        log_dict[f'{prefix}/sample_{i}/text'] = s['text']
                        log_dict[f'{prefix}/sample_{i}/similarity'] = s['similarity']
                    self.log(log_dict)

            self.model.train()

        except Exception as e:
            import warnings
            warnings.warn(f"Multimodal sample logging failed: {e}")
            self.model.train()


def create_multimodal_trainer(
    model: nn.Module,
    model_type: str,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    **kwargs,
):
    """
    Factory function to create appropriate trainer for multimodal model.

    Args:
        model: Vision-language model
        model_type: "clip" or "llava"
        training_args: HuggingFace TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        data_collator: Multimodal data collator
        **kwargs: Additional trainer arguments

    Returns:
        MultimodalSFTTrainer instance

    Example:
        trainer = create_multimodal_trainer(
            model=clip_model,
            model_type="clip",
            training_args=training_args,
            train_dataset=train_data,
            data_collator=clip_collator,
        )
        trainer.train()
    """
    return MultimodalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_type=model_type,
        **kwargs,
    )
