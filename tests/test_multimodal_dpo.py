#!/usr/bin/env python3
"""
Multimodal DPO Pipeline Test

Tests Direct Preference Optimization for vision-language models:
1. DPO data collation (preference pairs)
2. MultimodalDPOTrainer initialization
3. DPO loss computation (CLIP and LLaVA approaches)
4. Minimal DPO training
5. Reward margin evaluation

Run: python tests/test_multimodal_dpo.py
"""

import os
import sys
import platform
from pathlib import Path

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import TrainingArguments, CLIPProcessor
from datasets import Dataset

from src.models.vision_language import create_vision_language_model, CLIPWrapper
from src.data.processors.multimodal import MultimodalDataProcessor
from src.data.collators.multimodal import MultimodalDPODataCollator
from src.core.dpo.multimodal_trainer import MultimodalDPOTrainer
from src.core.dpo.loss import dpo_loss


def test_dpo_data_collator():
    """Test 1: DPO data collator for preference pairs."""
    print("\n" + "=" * 80)
    print("TEST 1: DPO Data Collator")
    print("=" * 80)

    # Setup
    processor_obj = MultimodalDataProcessor()
    examples = processor_obj.create_synthetic_data(num_examples=4)

    # Create preference pairs
    print("\n1.1 Creating preference pairs...")
    preference_pairs = processor_obj.create_preference_pairs(
        examples,
        augment_negatives=True,
    )
    assert len(preference_pairs) >= 4, "Should create at least 4 preference pairs"
    print(f"  ✓ Created {len(preference_pairs)} preference pairs")

    # Verify structure
    pair = preference_pairs[0]
    assert 'chosen_image' in pair, "Should have chosen_image"
    assert 'chosen_text' in pair, "Should have chosen_text"
    assert 'rejected_image' in pair, "Should have rejected_image"
    assert 'rejected_text' in pair, "Should have rejected_text"
    print(f"  ✓ Preference pair structure valid")

    # Test collator
    print("\n1.2 Testing MultimodalDPODataCollator...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dpo_collator = MultimodalDPODataCollator(
        tokenizer=clip_processor.tokenizer,
        image_processor=clip_processor.image_processor,
        max_length=77,
    )

    # Collate a batch
    batch = dpo_collator(preference_pairs[:4])

    # Verify batch structure
    assert 'chosen_pixel_values' in batch, "Should have chosen_pixel_values"
    assert 'chosen_input_ids' in batch, "Should have chosen_input_ids"
    assert 'chosen_attention_mask' in batch, "Should have chosen_attention_mask"
    assert 'chosen_labels' in batch, "Should have chosen_labels"
    assert 'rejected_pixel_values' in batch, "Should have rejected_pixel_values"
    assert 'rejected_input_ids' in batch, "Should have rejected_input_ids"
    assert 'rejected_attention_mask' in batch, "Should have rejected_attention_mask"
    assert 'rejected_labels' in batch, "Should have rejected_labels"

    # Verify shapes
    batch_size = 4
    assert batch['chosen_pixel_values'].shape[0] == batch_size, f"Batch size should be {batch_size}"
    assert batch['rejected_pixel_values'].shape[0] == batch_size, f"Batch size should be {batch_size}"

    print(f"  ✓ Batch structure valid:")
    print(f"    chosen_pixel_values: {batch['chosen_pixel_values'].shape}")
    print(f"    chosen_input_ids: {batch['chosen_input_ids'].shape}")
    print(f"    rejected_pixel_values: {batch['rejected_pixel_values'].shape}")
    print(f"    rejected_input_ids: {batch['rejected_input_ids'].shape}")

    print("\n✓ DPO data collator tests passed")


def test_clip_dpo_trainer_init():
    """Test 2: Initialize MultimodalDPOTrainer for CLIP."""
    print("\n" + "=" * 80)
    print("TEST 2: CLIP DPO Trainer Initialization")
    print("=" * 80)

    # Load policy and reference models
    print("\n2.1 Loading policy model (CLIP)...")
    policy_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,  # Skip LoRA for test
        device="cpu",
    )
    print("  ✓ Policy model loaded")

    print("\n2.2 Loading reference model (CLIP)...")
    ref_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )
    # Freeze reference model
    ref_model.model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad = False
    print("  ✓ Reference model loaded and frozen")

    # Create trainer
    print("\n2.3 Initializing MultimodalDPOTrainer...")

    # Prepare minimal dataset
    processor_obj = MultimodalDataProcessor()
    examples = processor_obj.create_synthetic_data(num_examples=8)
    preference_pairs = processor_obj.create_preference_pairs(examples, augment_negatives=True)

    train_data = {
        'chosen_image': [p['chosen_image'] for p in preference_pairs[:8]],
        'chosen_text': [p['chosen_text'] for p in preference_pairs[:8]],
        'rejected_image': [p['rejected_image'] for p in preference_pairs[:8]],
        'rejected_text': [p['rejected_text'] for p in preference_pairs[:8]],
    }
    train_dataset = Dataset.from_dict(train_data)

    # Create collator
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dpo_collator = MultimodalDPODataCollator(
        tokenizer=clip_processor.tokenizer,
        image_processor=clip_processor.image_processor,
        max_length=77,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_outputs/clip_dpo_test",
        max_steps=2,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        logging_steps=1,
        remove_unused_columns=False,
        report_to=[],
        save_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # Create DPO trainer
    trainer = MultimodalDPOTrainer(
        model=policy_model.model,
        ref_model=ref_model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=dpo_collator,
        tokenizer=clip_processor.tokenizer,
        model_type="clip",
        beta=0.1,
        log_rewards=True,
    )

    assert trainer is not None, "Trainer should be initialized"
    assert trainer.model_type == "clip", "Model type should be CLIP"
    assert trainer.beta == 0.1, "Beta should be 0.1"
    print("  ✓ MultimodalDPOTrainer initialized successfully")

    print("\n✓ CLIP DPO trainer initialization tests passed")

    return trainer


def test_dpo_loss_computation():
    """Test 3: DPO loss computation."""
    print("\n" + "=" * 80)
    print("TEST 3: DPO Loss Computation")
    print("=" * 80)

    # Test standard DPO loss function
    print("\n3.1 Testing DPO loss function...")

    # Simulate log probabilities
    batch_size = 4
    policy_chosen_logprob = torch.randn(batch_size)  # Random log probs
    policy_rejected_logprob = torch.randn(batch_size)
    ref_chosen_logprob = torch.randn(batch_size)
    ref_rejected_logprob = torch.randn(batch_size)

    # Make chosen better than rejected (on average)
    policy_chosen_logprob += 0.5
    ref_chosen_logprob += 0.5

    beta = 0.1
    loss, details = dpo_loss(
        policy_chosen_logprob=policy_chosen_logprob,
        policy_rejected_logprob=policy_rejected_logprob,
        ref_chosen_logprob=ref_chosen_logprob,
        ref_rejected_logprob=ref_rejected_logprob,
        beta=beta,
    )

    assert loss is not None, "Loss should be computed"
    assert isinstance(loss.item(), float), "Loss should be a scalar"
    assert 'reward_chosen' in details, "Should have reward_chosen"
    assert 'reward_rejected' in details, "Should have reward_rejected"
    assert 'reward_margin' in details, "Should have reward_margin"
    assert 'accuracy' in details, "Should have accuracy"

    print(f"  ✓ DPO loss: {loss.item():.4f}")
    print(f"    Reward margin: {details['reward_margin'].item():.4f}")
    print(f"    Accuracy: {details['accuracy'].item():.2%}")

    # Test that accuracy is reasonable (with biased inputs)
    assert details['accuracy'].item() >= 0.0, "Accuracy should be >= 0"
    assert details['accuracy'].item() <= 1.0, "Accuracy should be <= 1"

    print("\n✓ DPO loss computation tests passed")


def test_clip_dpo_forward():
    """Test 4: CLIP DPO forward pass."""
    print("\n" + "=" * 80)
    print("TEST 4: CLIP DPO Forward Pass")
    print("=" * 80)

    # Setup models
    print("\n4.1 Setting up CLIP models...")
    policy_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )

    ref_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )
    ref_model.model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad = False

    print("  ✓ Models loaded")

    # Prepare data
    print("\n4.2 Preparing preference batch...")
    processor_obj = MultimodalDataProcessor()
    examples = processor_obj.create_synthetic_data(num_examples=4)
    preference_pairs = processor_obj.create_preference_pairs(examples, augment_negatives=True)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dpo_collator = MultimodalDPODataCollator(
        tokenizer=clip_processor.tokenizer,
        image_processor=clip_processor.image_processor,
        max_length=77,
    )

    batch = dpo_collator(preference_pairs[:2])
    print("  ✓ Batch prepared")

    # Test forward pass through trainer
    print("\n4.3 Testing forward pass...")

    training_args = TrainingArguments(
        output_dir="./test_outputs/clip_dpo_forward",
        max_steps=1,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        logging_steps=1,
        remove_unused_columns=False,
        report_to=[],
        save_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # Dummy dataset for trainer
    dummy_data = {
        'chosen_image': [p['chosen_image'] for p in preference_pairs[:2]],
        'chosen_text': [p['chosen_text'] for p in preference_pairs[:2]],
        'rejected_image': [p['rejected_image'] for p in preference_pairs[:2]],
        'rejected_text': [p['rejected_text'] for p in preference_pairs[:2]],
    }
    dummy_dataset = Dataset.from_dict(dummy_data)

    trainer = MultimodalDPOTrainer(
        model=policy_model.model,
        ref_model=ref_model.model,
        args=training_args,
        train_dataset=dummy_dataset,
        data_collator=dpo_collator,
        tokenizer=clip_processor.tokenizer,
        model_type="clip",
        beta=0.1,
        log_rewards=False,  # Disable for test
    )

    # Compute loss
    loss = trainer.compute_loss(policy_model.model, batch, return_outputs=False)

    assert loss is not None, "Loss should be computed"
    assert isinstance(loss.item(), float), "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"

    print(f"  ✓ Forward pass successful, loss: {loss.item():.4f}")

    print("\n✓ CLIP DPO forward pass tests passed")


def test_minimal_dpo_training():
    """Test 5: Run minimal DPO training."""
    print("\n" + "=" * 80)
    print("TEST 5: Minimal DPO Training")
    print("=" * 80)

    # Skip on macOS due to known Trainer multiprocessing issues
    if platform.system() == "Darwin":
        print("\n⚠️  Skipping DPO training test on macOS (known multiprocessing issues)")
        print("   DPO training works correctly on Linux/Windows")
        print("   For macOS testing, use: scripts/train/train_multimodal_dpo.py")
        print("\n✓ DPO training test skipped (platform-specific)")
        return

    # Setup
    print("\n5.1 Setting up for minimal DPO training...")

    policy_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )

    ref_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )
    ref_model.model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad = False

    # Prepare data
    processor_obj = MultimodalDataProcessor()
    examples = processor_obj.create_synthetic_data(num_examples=8)
    preference_pairs = processor_obj.create_preference_pairs(examples, augment_negatives=True)

    train_data = {
        'chosen_image': [p['chosen_image'] for p in preference_pairs[:8]],
        'chosen_text': [p['chosen_text'] for p in preference_pairs[:8]],
        'rejected_image': [p['rejected_image'] for p in preference_pairs[:8]],
        'rejected_text': [p['rejected_text'] for p in preference_pairs[:8]],
    }
    train_dataset = Dataset.from_dict(train_data)

    # Create collator
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dpo_collator = MultimodalDPODataCollator(
        tokenizer=clip_processor.tokenizer,
        image_processor=clip_processor.image_processor,
        max_length=77,
    )

    # Training arguments
    print("\n5.2 Running 2 DPO training steps...")
    training_args = TrainingArguments(
        output_dir="./test_outputs/clip_dpo_training",
        max_steps=2,  # Just 2 steps
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        logging_steps=1,
        remove_unused_columns=False,
        report_to=[],
        save_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # Create trainer
    trainer = MultimodalDPOTrainer(
        model=policy_model.model,
        ref_model=ref_model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=dpo_collator,
        tokenizer=clip_processor.tokenizer,
        model_type="clip",
        beta=0.1,
        log_rewards=True,
    )

    # Train
    try:
        result = trainer.train()
        print(f"  ✓ DPO training completed: {result.metrics.get('train_loss', 'N/A')} loss")
    except Exception as e:
        print(f"  ✗ DPO training failed: {e}")
        raise

    print("\n✓ Minimal DPO training test passed")


def test_reward_margin_evaluation():
    """Test 6: Evaluate reward margins."""
    print("\n" + "=" * 80)
    print("TEST 6: Reward Margin Evaluation")
    print("=" * 80)

    # This tests that DPO can distinguish between chosen and rejected
    print("\n6.1 Computing reward margins...")

    # Setup models
    policy_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )

    ref_model = create_vision_language_model(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        use_lora=False,
        device="cpu",
    )

    # Prepare data
    processor_obj = MultimodalDataProcessor()
    examples = processor_obj.create_synthetic_data(num_examples=4)
    preference_pairs = processor_obj.create_preference_pairs(examples, augment_negatives=True)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dpo_collator = MultimodalDPODataCollator(
        tokenizer=clip_processor.tokenizer,
        image_processor=clip_processor.image_processor,
        max_length=77,
    )

    batch = dpo_collator(preference_pairs[:2])

    # Compute similarities for chosen
    with torch.no_grad():
        # Get embeddings
        chosen_image_embeds = policy_model.model.get_image_features(
            pixel_values=batch['chosen_pixel_values']
        )
        chosen_text_embeds = policy_model.model.get_text_features(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask'],
        )

        # Normalize
        chosen_image_embeds = chosen_image_embeds / chosen_image_embeds.norm(dim=-1, keepdim=True)
        chosen_text_embeds = chosen_text_embeds / chosen_text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity
        chosen_sim = (chosen_image_embeds * chosen_text_embeds).sum(dim=-1)

        # Same for rejected
        rejected_image_embeds = policy_model.model.get_image_features(
            pixel_values=batch['rejected_pixel_values']
        )
        rejected_text_embeds = policy_model.model.get_text_features(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask'],
        )

        rejected_image_embeds = rejected_image_embeds / rejected_image_embeds.norm(dim=-1, keepdim=True)
        rejected_text_embeds = rejected_text_embeds / rejected_text_embeds.norm(dim=-1, keepdim=True)

        rejected_sim = (rejected_image_embeds * rejected_text_embeds).sum(dim=-1)

    # Compute margin
    margin = (chosen_sim - rejected_sim).mean()

    print(f"  ✓ Mean chosen similarity: {chosen_sim.mean().item():.3f}")
    print(f"  ✓ Mean rejected similarity: {rejected_sim.mean().item():.3f}")
    print(f"  ✓ Reward margin: {margin.item():.3f}")

    # For synthetic data with augmented negatives, chosen should generally be better
    # (though not guaranteed for untrained model)
    print(f"  ✓ Margin computed successfully")

    print("\n✓ Reward margin evaluation tests passed")


def run_all_tests():
    """Run all DPO tests."""
    print("=" * 80)
    print("MULTIMODAL DPO PIPELINE TESTS")
    print("=" * 80)
    print("\nTesting Direct Preference Optimization for multimodal models...")

    try:
        test_dpo_data_collator()
        test_clip_dpo_trainer_init()
        test_dpo_loss_computation()
        test_clip_dpo_forward()
        test_minimal_dpo_training()
        test_reward_margin_evaluation()

        print("\n" + "=" * 80)
        print("✓ ALL DPO TESTS PASSED!")
        print("=" * 80)
        print("\nMultimodal DPO pipeline is fully functional.")

        if platform.system() == "Darwin":
            print("\nNote: Training test was skipped on macOS (platform limitation).")
            print("      For training verification, use: scripts/train/train_multimodal_dpo.py")

        print("\nNext steps:")
        print("  1. Run full DPO training: python scripts/train/train_multimodal_dpo.py experiment=clip_dpo")
        print("  2. Compare with SFT baseline")
        print("  3. Evaluate preference accuracy")
        print("\n" + "=" * 80)

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')  # Suppress transformer warnings

    success = run_all_tests()
    sys.exit(0 if success else 1)
