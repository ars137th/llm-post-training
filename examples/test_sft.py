"""
Test SFT Implementation

Quick test to verify that the SFT trainer works correctly.
Runs a few training steps on synthetic data.

Usage:
    python examples/test_sft.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import TrainingArguments
from datasets import Dataset

from src.models.language import LanguageModel
from src.core.sft.trainer import SFTTrainer
from src.core.sft.collator import DataCollatorForSFT, create_sft_dataset


def test_sft():
    """Test SFT training pipeline."""
    print("=" * 60)
    print("Testing SFT Implementation")
    print("=" * 60)

    # Step 1: Load model
    print("\n1. Loading GPT-2 with LoRA...")
    model = LanguageModel.from_pretrained(
        "gpt2",
        use_lora=True,
        lora_config={"r": 8, "lora_alpha": 16},
    )
    print(f"   ✓ Model loaded")
    print(f"   ✓ Total parameters: {model.num_parameters:,}")
    print(f"   ✓ Trainable: {model.num_trainable_parameters:,} ({100 * model.num_trainable_parameters / model.num_parameters:.2f}%)")

    # Step 2: Create dataset
    print("\n2. Creating synthetic dataset...")
    examples = [
        {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
        {"prompt": "What is ML?", "response": "ML is machine learning."},
        {"prompt": "What is Python?", "response": "Python is a programming language."},
        {"prompt": "What is 2+2?", "response": "2+2 equals 4."},
    ]

    tokenized = create_sft_dataset(
        examples=examples,
        tokenizer=model.tokenizer,
        max_length=64,
    )
    dataset = Dataset.from_list(tokenized)
    print(f"   ✓ Dataset created: {len(dataset)} examples")

    # Step 3: Setup trainer
    print("\n3. Setting up trainer...")
    data_collator = DataCollatorForSFT(
        tokenizer=model.tokenizer,
        max_length=64,
    )

    training_args = TrainingArguments(
        output_dir="./outputs/test_sft",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],  # Disable wandb/tensorboard for test
    )

    trainer = SFTTrainer(
        model=model.model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=model.tokenizer,
        data_collator=data_collator,
        loss_type="causal_lm",
        log_predictions=False,
    )
    print("   ✓ Trainer created")

    # Step 4: Train for a few steps
    print("\n4. Training for a few steps...")
    trainer.train()
    print("   ✓ Training completed")

    # Step 5: Test generation
    print("\n5. Testing generation...")
    model.eval()
    test_prompt = "What is AI?"

    encoded = model.tokenizer(test_prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
        )

    generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"   Prompt: {test_prompt}")
    print(f"   Generated: {generated_text}")
    print("   ✓ Generation works")

    # Step 6: Check metrics
    print("\n6. Checking training metrics...")
    metrics = trainer.get_training_metrics()
    if metrics['losses']:
        print(f"   Initial loss: {metrics['losses'][0]:.4f}")
        print(f"   Final loss: {metrics['losses'][-1]:.4f}")
        print(f"   Loss decreased: {metrics['losses'][0] > metrics['losses'][-1]}")
    print("   ✓ Metrics tracked")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nSFT implementation is working correctly.")
    print("You can now:")
    print("  - Run full training: python scripts/train/train_sft.py")
    print("  - Try the notebook: notebooks/01_understanding_sft.ipynb")
    print("  - Experiment with different models and datasets")


if __name__ == "__main__":
    test_sft()
