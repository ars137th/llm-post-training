#!/usr/bin/env python3
"""
Train Multimodal Models with DPO

Direct Preference Optimization for vision-language models.
Learns from preference pairs: (image, chosen_text, rejected_text)

Supports:
- CLIP: Learns preferred image-text alignments
- LLaVA: Generates preferred captions

Usage:
    # Train CLIP with DPO on synthetic preferences
    python scripts/train/train_multimodal_dpo.py \
        experiment=clip_dpo

    # Train LLaVA with DPO on real preferences
    python scripts/train/train_multimodal_dpo.py \
        experiment=llava_dpo \
        data.dataset_name=coco

    # Train CLIP with DPO on custom preferences
    python scripts/train/train_multimodal_dpo.py \
        experiment=clip_dpo \
        data=custom_preferences \
        data.train_file=/path/to/preferences.json \
        data.image_dir=/path/to/images \
        data.format=json

Note: Requires SFT model as starting point. Train SFT first!
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
from transformers import TrainingArguments, CLIPProcessor, AutoProcessor
from datasets import Dataset
from copy import deepcopy

from src.models.vision_language import create_vision_language_model
from src.data.processors.multimodal import MultimodalDataProcessor
from src.data.collators.multimodal import MultimodalDPODataCollator
from src.core.dpo.multimodal_trainer import MultimodalDPOTrainer
from src.evaluation.metrics.multimodal import CLIPScoreMetric
from src.utils.logging import setup_logging


def prepare_dpo_dataset(
    examples: list,
    model_type: str,
) -> Dataset:
    """
    Convert preference pairs to HuggingFace Dataset for DPO.

    Args:
        examples: List of preference pair dicts with:
            - chosen_image, chosen_text
            - rejected_image, rejected_text
        model_type: "clip" or "llava"

    Returns:
        HuggingFace Dataset
    """
    # Preference pairs already have the right structure
    data = {
        'chosen_image': [ex['chosen_image'] for ex in examples],
        'chosen_text': [ex['chosen_text'] for ex in examples],
        'rejected_image': [ex['rejected_image'] for ex in examples],
        'rejected_text': [ex['rejected_text'] for ex in examples],
    }

    return Dataset.from_dict(data)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function for multimodal DPO.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("MULTIMODAL DPO TRAINING")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup logging
    logger = setup_logging(cfg)
    logger.info("Starting multimodal DPO training")

    # Device setup
    device = cfg.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Verify model type
    model_arch = cfg.model.get('architecture', '').lower()
    if model_arch not in ['clip', 'llava']:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    logger.info(f"Training {model_arch.upper()} model with DPO")

    # Load processor
    logger.info(f"Loading processor: {cfg.model.model_name_or_path}")
    if model_arch == "clip":
        processor = CLIPProcessor.from_pretrained(cfg.model.model_name_or_path)
    else:  # llava
        processor = AutoProcessor.from_pretrained(cfg.model.model_name_or_path)

    # Load policy model (to be trained)
    logger.info("Creating policy model...")
    policy_wrapper = create_vision_language_model(
        model_type=model_arch,
        model_name=cfg.model.model_name_or_path,
        use_lora=cfg.model.get('use_lora', False),
        lora_config=OmegaConf.to_container(cfg.model.get('lora_config', {}), resolve=True),
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        device=device,
    )

    # Load reference model (frozen copy for DPO)
    logger.info("Creating reference model (frozen)...")
    ref_wrapper = create_vision_language_model(
        model_type=model_arch,
        model_name=cfg.model.model_name_or_path,
        use_lora=False,  # Reference model doesn't need LoRA
        use_4bit=cfg.model.get('use_4bit', False),
        use_8bit=cfg.model.get('use_8bit', False),
        device=device,
    )

    # Freeze reference model
    ref_wrapper.model.eval()
    for param in ref_wrapper.model.parameters():
        param.requires_grad = False

    logger.info("✓ Policy and reference models loaded")

    # Load data
    logger.info("Loading preference data...")
    data_processor = MultimodalDataProcessor()

    dataset_name = cfg.data.get('dataset_name', 'synthetic')
    max_train_samples = cfg.data.get('max_train_samples', 500)
    max_eval_samples = cfg.data.get('max_eval_samples', 100)

    if dataset_name == "synthetic":
        logger.info(f"Generating synthetic preference data...")
        # Create base examples
        base_train = data_processor.create_synthetic_data(num_examples=max_train_samples)
        base_eval = data_processor.create_synthetic_data(num_examples=max_eval_samples)

        # Create preference pairs (chosen vs rejected captions)
        train_preferences = data_processor.create_preference_pairs(
            base_train,
            augment_negatives=True,
        )
        eval_preferences = data_processor.create_preference_pairs(
            base_eval,
            augment_negatives=True,
        )

    elif dataset_name == "coco":
        logger.info("Loading COCO Captions...")
        base_train = data_processor.load_coco_captions(
            split="train",
            num_examples=max_train_samples
        )
        base_eval = data_processor.load_coco_captions(
            split="validation",
            num_examples=max_eval_samples
        )

        # Create preference pairs
        train_preferences = data_processor.create_preference_pairs(
            base_train,
            augment_negatives=True,
        )
        eval_preferences = data_processor.create_preference_pairs(
            base_eval,
            augment_negatives=True,
        )

    elif dataset_name == "custom" or cfg.data.get('train_file'):
        # Custom preference data loading
        logger.info("Loading custom preference data...")
        from src.data.loaders.custom import load_custom_preference_data
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

        # Load training preferences
        train_prefs = load_custom_preference_data(
            data_file=train_file,
            image_dir=image_dir,
            format=data_format,
            max_samples=max_train_samples,
            validate=validate
        )

        # Load validation preferences
        if val_file:
            eval_prefs = load_custom_preference_data(
                data_file=val_file,
                image_dir=image_dir,
                format=data_format,
                max_samples=max_eval_samples,
                validate=validate
            )
        else:
            # Split train data
            val_split = int(len(train_prefs) * 0.1)
            eval_prefs = train_prefs[:val_split]
            train_prefs = train_prefs[val_split:]
            logger.info(f"  Split data: {len(train_prefs)} train, {len(eval_prefs)} val")

        # Convert to preference pair format expected by trainer
        logger.info(f"Converting {len(train_prefs)} train preferences...")
        train_preferences = []
        for pref in train_prefs:
            img = Image.open(pref['image_path'])
            train_preferences.append({
                'chosen_image': img,
                'chosen_text': pref['chosen'],
                'rejected_image': img,  # Same image, different captions
                'rejected_text': pref['rejected']
            })

        logger.info(f"Converting {len(eval_prefs)} eval preferences...")
        eval_preferences = []
        for pref in eval_prefs:
            img = Image.open(pref['image_path'])
            eval_preferences.append({
                'chosen_image': img,
                'chosen_text': pref['chosen'],
                'rejected_image': img,
                'rejected_text': pref['rejected']
            })

        logger.info("✓ Custom preference data loaded successfully")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Created {len(train_preferences)} train preference pairs")
    logger.info(f"Created {len(eval_preferences)} eval preference pairs")

    # Convert to HuggingFace Dataset
    train_dataset = prepare_dpo_dataset(train_preferences, model_arch)
    eval_dataset = prepare_dpo_dataset(eval_preferences, model_arch)

    # Create DPO data collator
    logger.info("Creating DPO data collator...")
    data_collator = MultimodalDPODataCollator(
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        image_processor=processor.image_processor if hasattr(processor, 'image_processor') else processor,
        max_length=cfg.data.get('max_length', 512),
    )

    # Training arguments
    training_cfg = cfg.training
    output_dir = training_cfg.get('output_dir', './outputs/multimodal_dpo')
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get('num_epochs', 3),
        per_device_train_batch_size=training_cfg.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_cfg.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        learning_rate=training_cfg.get('learning_rate', 1e-5),
        weight_decay=training_cfg.get('weight_decay', 0.01),
        warmup_steps=training_cfg.get('warmup_steps', 50),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        logging_steps=training_cfg.get('logging_steps', 10),
        eval_strategy="steps" if training_cfg.get('eval_steps') else "epoch",
        eval_steps=training_cfg.get('eval_steps', None),
        save_strategy="steps" if training_cfg.get('save_steps') else "epoch",
        save_steps=training_cfg.get('save_steps', None),
        save_total_limit=training_cfg.get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=training_cfg.get('fp16', False) and device == 'cuda',
        bf16=training_cfg.get('bf16', False) and device == 'cuda',
        report_to=["tensorboard"] if cfg.logging.get('use_tensorboard', True) else [],
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,  # Important: keep images!
        dataloader_pin_memory=True,
        dataloader_num_workers=training_cfg.get('dataloader_num_workers', 0),
    )

    # Create DPO trainer
    logger.info("Creating MultimodalDPOTrainer...")
    beta = cfg.technique.get('beta', 0.1)
    logger.info(f"DPO beta (temperature): {beta}")

    trainer = MultimodalDPOTrainer(
        model=policy_wrapper.model,
        ref_model=ref_wrapper.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        model_type=model_arch,
        beta=beta,
        log_rewards=True,
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING DPO TRAINING")
    logger.info("=" * 80 + "\n")

    train_result = trainer.train()

    # Save model
    logger.info("\nSaving policy model...")
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

    logger.info(f"\n✓ DPO training complete! Model saved to: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Compare with SFT baseline")
    logger.info(f"  2. Evaluate on preference accuracy")
    logger.info(f"  3. Generate samples to verify alignment")


if __name__ == "__main__":
    main()
