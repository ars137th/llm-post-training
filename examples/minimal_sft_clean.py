"""
Minimal SFT Example - Clean Output

This version suppresses the harmless semaphore warning if everything works.

Usage:
    python examples/minimal_sft_clean.py
"""

# Disable warnings and multiprocessing FIRST
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
import sys

# Suppress the semaphore warning specifically
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.language import LanguageModel
from src.data.processors.text import TextProcessor


def main():
    print("🚀 Minimal SFT Example")
    print("=" * 50)

    # 1. Load a small model with LoRA
    print("\n📦 Loading model...")
    try:
        model_wrapper = LanguageModel.from_pretrained(
            "gpt2",
            use_lora=True,
            lora_config={"r": 8, "lora_alpha": 16},
            device="cpu",  # Force CPU for stability
        )
        print("   ✓ Model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

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

    print("\n✅ Setup complete!")
    print(f"Model parameters: {model_wrapper.num_parameters:,}")
    print(f"Trainable parameters: {model_wrapper.num_trainable_parameters:,}")
    print(f"Training examples: {len(processed_data)}")

    # 5. Test generation
    print("\n🎯 Testing generation...")
    model_wrapper.eval()
    test_prompt = "What is the capital of France?"

    try:
        # Tokenize and move to correct device
        encoded = processor.tokenize(test_prompt, return_tensors="pt")
        encoded = {k: v.to(model_wrapper.device) for k, v in encoded.items()}

        # Generate with attention_mask
        with torch.no_grad():
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
        print("   ✓ Generation successful")

    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✨ Example complete!")
    print("\nNext steps:")
    print("  - Full training: python scripts/train/train_sft.py")
    print("  - Tutorial: notebooks/01_understanding_sft.ipynb")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
