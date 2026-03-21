#!/usr/bin/env python3
"""
Train Multimodal Models (CLIP, LLaVA)

Supports:
- CLIP: Image-text contrastive learning
- LLaVA: Vision-language instruction following

This script uses MultimodalSFTTrainer which properly handles:
- LoRA training with CLIP (avoids PEFT kwargs routing issues)
- Multimodal inputs (pixel_values + input_ids)
- Contrastive loss for CLIP
- Causal LM loss for LLaVA

Usage:
    # Train CLIP on synthetic data
    python scripts/train/train_multimodal.py \
        experiment=clip_image_caption

    # Train CLIP with LoRA on COCO
    python scripts/train/train_multimodal.py \
        experiment=clip_image_caption \
        data.dataset_name=coco \
        model.use_lora=true

    # Train LLaVA on instruction data
    python scripts/train/train_multimodal.py \
        experiment=llava_instruction

    # Train CLIP on custom image-caption data
    python scripts/train/train_multimodal.py \
        experiment=clip_image_caption \
        data=custom_image_caption \
        data.train_file=/path/to/train.json \
        data.image_dir=/path/to/images \
        data.format=json

Note: LoRA + CLIP requires special handling. See docs/known_issues.md for details.
See docs/CUSTOM_DATA_GUIDE.md for custom data format specifications.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import TrainingArguments, Trainer
from transformers import CLIPProcessor
from datasets import Dataset

from src.models.vision_language import create_vision_language_model
from src.data.processors.multimodal import MultimodalDataProcessor, MultimodalExample
from src.data.collators.multimodal import create_multimodal_collator
from src.evaluation.metrics.multimodal import CLIPScoreMetric
from src.utils.logging import setup_logging


def prepare_dataset(
    examples: list,
    model_type: str,
    instruction_template: str = None,
) -> Dataset:
    """
    Convert MultimodalExample objects to HuggingFace Dataset.

    Args:
        examples: List of MultimodalExample objects
        model_type: "clip" or "llava"
        instruction_template: Template for LLaVA instructions

    Returns:
        HuggingFace Dataset
    """
    if model_type.lower() == "llava":
        # LLaVA format: image, instruction, response
        data = {
            'image': [ex.image for ex in examples],
            'instruction': [instruction_template or "Describe this image:" for _ in examples],
            'response': [ex.caption for ex in examples],
        }
    else:
        # CLIP format: image, text
        data = {
            'image': [ex.image for ex in examples],
            'text': [ex.caption for ex in examples],
        }

    return Dataset.from_dict(data)


def compute_multimodal_metrics(model_type: str, processor):
    """
    Create compute_metrics function for multimodal evaluation.

    Args:
        model_type: "clip" or "llava"
        processor: Model processor

    Returns:
        Metrics computation function
    """
    if model_type.lower() == "clip":
        # For CLIP, we can compute CLIP Score
        clip_metric = CLIPScoreMetric()

        def compute_metrics(eval_pred):
            """Compute CLIP Score (requires custom eval loop)."""
            # Note: This is a placeholder. Full implementation requires
            # access to images, which aren't available in standard eval_pred.
            # For proper evaluation, use evaluate.py script.
            return {
                "eval_loss": 0.0,  # Computed by Trainer
            }

        return compute_metrics
    else:
        # For LLaVA, use standard language modeling metrics
        def compute_metrics(eval_pred):
            """Compute perplexity."""
            predictions, labels = eval_pred
            # Standard metrics computed by Trainer
            return {}

        return compute_metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function for multimodal models.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("MULTIMODAL MODEL TRAINING")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup logging
    logger = setup_logging(cfg)
    logger.info("Starting multimodal training")

    # Device setup
    device = cfg.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # macOS fork safety workaround (using compat layer)
    from src.utils.compat import is_macos
    macos_detected = is_macos()
    if macos_detected:
        logger.warning("⚠️  macOS detected - applying fork safety workarounds")
        logger.warning("   Setting dataloader_num_workers=0 to avoid bus errors")
        logger.warning("   Disabling periodic evaluation (runs once at end)")

    # Verify model type
    model_arch = cfg.model.get('architecture', '').lower()
    if model_arch not in ['clip', 'llava']:
        raise ValueError(f"Unsupported model architecture: {model_arch}. Use 'clip' or 'llava'.")

    logger.info(f"Training {model_arch.upper()} model")

    # Load processor (handles both tokenizer and image processor)
    logger.info(f"Loading processor: {cfg.model.model_name_or_path}")
    if model_arch == "clip":
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(cfg.model.model_name_or_path)
    else:  # llava
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg.model.model_name_or_path)

    # Load model
    logger.info("Creating model...")
    model_wrapper = create_vision_language_model(
        model_type=model_arch,
        model_name=cfg.model.model_name_or_path,
        use_lora=cfg.model.get('use_lora', False),
        lora_config=OmegaConf.to_container(cfg.model.get('lora_config', {}), resolve=True),
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        device=device,
    )

    # Get underlying model for Trainer
    model = model_wrapper.model

    # Load data
    logger.info("Loading data...")
    data_processor = MultimodalDataProcessor()

    dataset_name = cfg.data.get('dataset_name', 'synthetic')
    max_train_samples = cfg.data.get('max_train_samples', 1000)
    max_eval_samples = cfg.data.get('max_eval_samples', 200)

    if dataset_name == "synthetic":
        logger.info(f"Generating synthetic data: {max_train_samples} train, {max_eval_samples} eval")
        train_examples = data_processor.create_synthetic_data(num_examples=max_train_samples)
        eval_examples = data_processor.create_synthetic_data(num_examples=max_eval_samples)
    elif dataset_name == "coco":
        logger.info("Loading COCO Captions dataset...")
        train_examples = data_processor.load_coco_captions(
            split="train",
            num_examples=max_train_samples
        )
        eval_examples = data_processor.load_coco_captions(
            split="validation",
            num_examples=max_eval_samples
        )
    elif dataset_name == "flickr30k":
        logger.info("Loading Flickr30k dataset...")
        train_examples = data_processor.load_flickr30k(
            split="train",
            num_examples=max_train_samples
        )
        eval_examples = data_processor.load_flickr30k(
            split="validation",
            num_examples=max_eval_samples
        )
    elif dataset_name == "custom" or cfg.data.get('train_file'):
        # Custom data loading
        logger.info("Loading custom image-caption data...")
        from src.data.loaders.custom import load_custom_image_caption_data
        from PIL import Image

        train_file = cfg.data.get('train_file')
        val_file = cfg.data.get('val_file')
        image_dir = cfg.data.get('image_dir')
        data_format = cfg.data.get('format', 'json')
        validate = cfg.data.get('validate_on_load', True)

        if not train_file:
            raise ValueError("For custom data, you must specify data.train_file")

        logger.info(f"  Train file: {train_file}")
        if val_file:
            logger.info(f"  Val file: {val_file}")
        logger.info(f"  Image dir: {image_dir or 'Using absolute paths'}")
        logger.info(f"  Format: {data_format}")

        # Load custom data
        if val_file:
            train_data, val_data = load_custom_image_caption_data(
                train_file=train_file,
                val_file=val_file,
                image_dir=image_dir,
                format=data_format,
                max_train_samples=max_train_samples,
                max_val_samples=max_eval_samples,
                validate=validate
            )
        else:
            train_data = load_custom_image_caption_data(
                train_file=train_file,
                val_file=None,
                image_dir=image_dir,
                format=data_format,
                max_train_samples=max_train_samples,
                validate=validate
            )
            # Use a small portion of train data for validation
            val_split = int(len(train_data) * 0.1)
            val_data = train_data[:val_split]
            train_data = train_data[val_split:]
            logger.info(f"  Split train data: {len(train_data)} train, {len(val_data)} val")

        # Convert to MultimodalExample objects
        logger.info(f"Converting {len(train_data)} train examples...")
        train_examples = []
        for item in train_data:
            img = Image.open(item['image_path'])
            train_examples.append(MultimodalExample(
                image=img,
                caption=item['caption']
            ))

        logger.info(f"Converting {len(val_data)} val examples...")
        eval_examples = []
        for item in val_data:
            img = Image.open(item['image_path'])
            eval_examples.append(MultimodalExample(
                image=img,
                caption=item['caption']
            ))

        logger.info("✓ Custom data loaded successfully")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Loaded {len(train_examples)} train examples, {len(eval_examples)} eval examples")

    # Convert to HuggingFace Dataset format
    instruction_template = cfg.data.get('instruction_template', None)
    train_dataset = prepare_dataset(train_examples, model_arch, instruction_template)
    eval_dataset = prepare_dataset(eval_examples, model_arch, instruction_template)

    # Create data collator
    logger.info(f"Creating {model_arch.upper()} data collator...")
    data_collator = create_multimodal_collator(
        model_type=model_arch,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        image_processor=processor.image_processor if hasattr(processor, 'image_processor') else processor,
        max_length=cfg.data.get('max_length', 512),
        instruction_template=instruction_template,
    )

    # Training arguments
    training_cfg = cfg.training
    output_dir = training_cfg.get('output_dir', './outputs/multimodal')
    os.makedirs(output_dir, exist_ok=True)

    # Use compatibility layer for API differences across transformers versions
    from src.utils.compat import TRANSFORMERS_VERSION
    from packaging import version
    TRANSFORMERS_4_36 = version.parse("4.36.0")

    # Build training arguments dict with version compatibility
    training_args_dict = {
        'output_dir': output_dir,
        'num_train_epochs': training_cfg.get('num_epochs', 3),
        'max_steps': training_cfg.get('max_steps', -1),  # Support max_steps override
        'per_device_train_batch_size': training_cfg.get('per_device_train_batch_size', 8),
        'per_device_eval_batch_size': training_cfg.get('per_device_eval_batch_size', 16),
        'gradient_accumulation_steps': training_cfg.get('gradient_accumulation_steps', 1),
        'learning_rate': training_cfg.get('learning_rate', 5e-5),
        'weight_decay': training_cfg.get('weight_decay', 0.01),
        'warmup_steps': training_cfg.get('warmup_steps', 100),
        'max_grad_norm': training_cfg.get('max_grad_norm', 1.0),
        'logging_steps': training_cfg.get('logging_steps', 10),
        'save_strategy': "steps" if training_cfg.get('save_steps') else "epoch",
        'save_steps': training_cfg.get('save_steps', None),
        'save_total_limit': training_cfg.get('save_total_limit', 2),
        'load_best_model_at_end': True,
        'metric_for_best_model': "eval_loss",
        'greater_is_better': False,
        'fp16': training_cfg.get('fp16', False) and device == 'cuda',
        'bf16': training_cfg.get('bf16', False) and device == 'cuda',
        'report_to': ["tensorboard"] if cfg.logging.get('use_tensorboard', True) else [],
        'remove_unused_columns': False,  # Important: keep images!
        'dataloader_pin_memory': device == 'cuda',  # Only useful on GPU
        'dataloader_num_workers': training_cfg.get('dataloader_num_workers', 0),
    }

    # macOS fork safety: Apply workarounds using compat layer
    # Evaluation will still run at the end if do_eval=True
    from src.utils.compat import apply_macos_training_workarounds
    training_args_dict = apply_macos_training_workarounds(training_args_dict)

    # Disable periodic evaluation on macOS (run only at end)
    eval_enabled_override = False if macos_detected else None

    # Handle eval_strategy vs evaluation_strategy (version-dependent)
    eval_enabled = training_cfg.get('eval_steps') is not None
    if eval_enabled_override is not None:
        eval_enabled = eval_enabled_override
    if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
        training_args_dict['eval_strategy'] = "steps" if eval_enabled else "epoch"
        # Don't include logging_dir (deprecated in 4.36+)
    else:
        training_args_dict['evaluation_strategy'] = "steps" if eval_enabled else "epoch"
        training_args_dict['logging_dir'] = os.path.join(output_dir, "logs")

    # Add eval_steps if present
    if eval_enabled:
        training_args_dict['eval_steps'] = training_cfg.get('eval_steps')

    # macOS: Adjust save strategy and load_best_model_at_end to match eval strategy
    # When eval_strategy="epoch", save_strategy must also be "epoch" if load_best_model_at_end=True
    if macos_detected:
        training_args_dict['save_strategy'] = "epoch"  # Match eval_strategy
        training_args_dict['save_steps'] = None  # Not used with "epoch" strategy
        training_args_dict['load_best_model_at_end'] = False  # No periodic eval to compare
        logger.warning("   Setting save_strategy='epoch' to match eval_strategy")
        logger.warning("   Disabling load_best_model_at_end (no periodic eval to compare)")

    training_args = TrainingArguments(**training_args_dict)

    # Create trainer - use MultimodalSFTTrainer for proper CLIP+LoRA handling
    logger.info("Creating MultimodalSFTTrainer...")
    from src.core.sft.multimodal_trainer import MultimodalSFTTrainer

    trainer = MultimodalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        compute_metrics=compute_multimodal_metrics(model_arch, processor),
        model_type=model_arch,  # "clip" or "llava"
        log_predictions=False,  # Disable sample generation for multimodal (complex)
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80 + "\n")

    train_result = trainer.train()

    # Save model
    logger.info("\nSaving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Log training metrics
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nTrain metrics:")
    for key, value in train_result.metrics.items():
        logger.info(f"  {key}: {value}")

    # Evaluate
    if training_cfg.get('do_eval', True):
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING")
        logger.info("=" * 80 + "\n")

        eval_result = trainer.evaluate()

        logger.info(f"\nEval metrics:")
        for key, value in eval_result.items():
            logger.info(f"  {key}: {value}")

        # For CLIP, compute CLIP Score
        if model_arch == "clip":
            logger.info("\nComputing CLIP Score on evaluation set...")
            clip_metric = CLIPScoreMetric(
                model_name=cfg.model.model_name_or_path,
                device=device,
            )

            # Get images and texts from eval dataset
            images = eval_dataset['image'][:100]  # Sample 100 for speed
            texts = eval_dataset['text'][:100] if 'text' in eval_dataset.features else eval_dataset['caption'][:100]

            scores = clip_metric.compute(images, texts)
            logger.info(f"\nCLIP Score Results:")
            logger.info(f"  Average: {scores['clip_score']:.2f}")
            logger.info(f"  Std Dev: {scores['clip_score_std']:.2f}")
            logger.info(f"  Range: [{scores['clip_score_min']:.2f}, {scores['clip_score_max']:.2f}]")

    logger.info(f"\n✓ Training complete! Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
