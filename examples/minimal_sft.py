"""
Minimal Supervised Fine-Tuning Example

A complete working example of SFT in under 50 lines.
This demonstrates the basic usage of the LLM post-training repository.

Usage:
    python examples/minimal_sft.py
"""

# CRITICAL: Set these BEFORE any imports to avoid multiprocessing issues on macOS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import TrainingArguments, Trainer
from src.models.language import LanguageModel
from src.data.processors.text import TextProcessor


def main():
    print("🚀 Minimal SFT Example")
    print("=" * 50)

    # 1. Load a small model with LoRA
    print("\n📦 Loading model...")
    model_wrapper = LanguageModel.from_pretrained(
        "gpt2",
        use_lora=True,
        lora_config={"r": 8, "lora_alpha": 16},
    )

    # 2. Create text processor
    processor = TextProcessor(
        tokenizer=model_wrapper.tokenizer,
        max_length=128,
    )

    # 3. Create a tiny synthetic dataset
    print("📚 Creating dataset...")
    train_data = [
        {"prompt": "What is the capital of France?", "response": "Paris"},
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "Who wrote Romeo and Juliet?", "response": "Shakespeare"},
    ]

    # 4. Process data for SFT
    print("🔧 Processing data...")
    processed_data = []
    for example in train_data:
        processed = processor.process_for_sft(
            prompt=example["prompt"],
            response=example["response"],
            mask_prompt=True,
        )
        processed_data.append(processed)

    # 5. Setup training
    print("⚙️  Setting up training...")
    training_args = TrainingArguments(
        output_dir="./outputs/minimal_sft",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
    )

    # Note: For a real training loop, you'd use DataLoader or HF Dataset
    # This is just a minimal example showing the components

    print("\n✅ Setup complete!")
    print(f"Model parameters: {model_wrapper.num_parameters:,}")
    print(f"Trainable parameters: {model_wrapper.num_trainable_parameters:,}")
    print(f"Training examples: {len(processed_data)}")

    # 6. Test generation
    print("\n🎯 Testing generation...")
    model_wrapper.eval()  # Set to evaluation mode
    test_prompt = "What is the capital of France?"

    # Tokenize and move to correct device
    encoded = processor.tokenize(test_prompt, return_tensors="pt")
    encoded = {k: v.to(model_wrapper.device) for k, v in encoded.items()}

    # Generate with attention_mask
    generated = model_wrapper.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=10,
        temperature=0.7,
        pad_token_id=model_wrapper.tokenizer.pad_token_id,
    )
    output = processor.decode(generated[0])
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {output}")

    print("\n" + "=" * 50)
    print("✨ Example complete!")
    print("\nFor full training, see scripts/train/train_sft.py")


if __name__ == "__main__":
    main()
