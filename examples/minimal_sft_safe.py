"""
Minimal SFT Example (Safe Version)

A safer version that handles device issues better and forces CPU mode if needed.

Usage:
    python examples/minimal_sft_safe.py

    # Force CPU mode
    python examples/minimal_sft_safe.py --cpu
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from src.models.language import LanguageModel
from src.data.processors.text import TextProcessor


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--no-lora', action='store_true', help='Disable LoRA')
    args = parser.parse_args()

    print("🚀 Minimal SFT Example (Safe Mode)")
    print("=" * 50)

    # Determine device
    if args.cpu:
        device = 'cpu'
        print("\n⚠️  Forced CPU mode")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'cpu'  # MPS can be unstable, use CPU for safety
        print("\n⚠️  MPS detected but using CPU for stability")
    else:
        device = 'cpu'
        print("\n✓ Using CPU")

    # 1. Load model
    print("\n📦 Loading model...")
    use_lora = not args.no_lora

    try:
        model_wrapper = LanguageModel.from_pretrained(
            "gpt2",
            use_lora=use_lora,
            lora_config={"r": 8, "lora_alpha": 16} if use_lora else None,
            device=device,
        )

        # Force to correct device
        if device == 'cpu':
            model_wrapper.model = model_wrapper.model.to('cpu')
            model_wrapper.device = 'cpu'

        print(f"   ✓ Model loaded on {model_wrapper.device}")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        print("\n   Trying without LoRA...")
        model_wrapper = LanguageModel.from_pretrained(
            "gpt2",
            use_lora=False,
            device='cpu',
        )
        model_wrapper.model = model_wrapper.model.to('cpu')
        model_wrapper.device = 'cpu'
        print(f"   ✓ Model loaded without LoRA on CPU")

    # 2. Create processor
    processor = TextProcessor(
        tokenizer=model_wrapper.tokenizer,
        max_length=128,
    )

    # 3. Create dataset
    print("📚 Creating dataset...")
    train_data = [
        {"prompt": "What is the capital of France?", "response": "Paris"},
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "Who wrote Romeo and Juliet?", "response": "Shakespeare"},
    ]

    # 4. Process data
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

    test_prompts = [
        "What is the capital of France?",
        "Hello",
    ]

    for test_prompt in test_prompts:
        try:
            print(f"\n  Testing: '{test_prompt}'")

            # Tokenize
            encoded = processor.tokenize(test_prompt, return_tensors="pt")

            # Move to device
            encoded = {k: v.to(model_wrapper.device) for k, v in encoded.items()}

            # Verify device match
            model_device = next(model_wrapper.model.parameters()).device
            input_device = encoded["input_ids"].device
            if model_device != input_device:
                print(f"    ⚠️  Device mismatch! Model: {model_device}, Input: {input_device}")
                encoded = {k: v.to(model_device) for k, v in encoded.items()}

            # Generate (try greedy first, safer)
            print("    Attempting greedy generation...")
            with torch.no_grad():
                output = model_wrapper.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=5,
                    do_sample=False,  # Greedy is more stable
                    pad_token_id=model_wrapper.tokenizer.pad_token_id,
                    eos_token_id=model_wrapper.tokenizer.eos_token_id,
                )

            # Decode
            generated_text = processor.decode(output[0], skip_special_tokens=True)
            print(f"    ✓ Generated: '{generated_text}'")

            # Try sampling
            print("    Attempting sampling...")
            with torch.no_grad():
                output = model_wrapper.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=model_wrapper.tokenizer.pad_token_id,
                    eos_token_id=model_wrapper.tokenizer.eos_token_id,
                )

            generated_text = processor.decode(output[0], skip_special_tokens=True)
            print(f"    ✓ Generated: '{generated_text}'")

        except Exception as e:
            print(f"    ✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n    This is where the bus error likely occurred.")
            print("    Try running: python examples/debug_sft.py")
            return

    print("\n" + "=" * 50)
    print("✨ Example complete!")
    print("\nNext steps:")
    print("  - Run diagnostics: python examples/debug_sft.py")
    print("  - Full training: python scripts/train/train_sft.py")
    print("  - Tutorial: notebooks/01_understanding_sft.ipynb")


if __name__ == "__main__":
    main()
