"""
Test minimal SFT without LoRA to isolate the issue.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from src.models.language import LanguageModel
from src.data.processors.text import TextProcessor


def main():
    print("🧪 Testing WITHOUT LoRA...")
    print("=" * 50)

    # Load model WITHOUT LoRA
    print("\n[1] Loading model without LoRA...")
    try:
        model_wrapper = LanguageModel.from_pretrained(
            "gpt2",
            use_lora=False,  # No LoRA
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create processor
    print("\n[2] Creating text processor...")
    try:
        processor = TextProcessor(
            tokenizer=model_wrapper.tokenizer,
            max_length=128,
        )
        print("✓ Processor created")
    except Exception as e:
        print(f"✗ Failed to create processor: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test generation
    print("\n[3] Testing generation...")
    model_wrapper.eval()
    test_prompt = "What is the capital of France?"

    try:
        encoded = processor.tokenize(test_prompt, return_tensors="pt")
        encoded = {k: v.to(model_wrapper.device) for k, v in encoded.items()}

        generated = model_wrapper.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=10,
            temperature=0.7,
            pad_token_id=model_wrapper.tokenizer.pad_token_id,
        )
        output = processor.decode(generated[0])
        print(f"✓ Generation succeeded")
        print(f"  Prompt: {test_prompt}")
        print(f"  Generated: {output}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 50)
    print("✅ Test PASSED - Issue is with LoRA")


if __name__ == "__main__":
    main()
