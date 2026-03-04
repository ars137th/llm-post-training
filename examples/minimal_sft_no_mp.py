"""
Minimal SFT Example - No Multiprocessing

This version completely disables multiprocessing to avoid bus errors on macOS.

Usage:
    python examples/minimal_sft_no_mp.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Disable all multiprocessing BEFORE importing anything else
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["OMP_NUM_THREADS"] = "1"  # Disable OpenMP threading
os.environ["MKL_NUM_THREADS"] = "1"  # Disable MKL threading

import torch
import warnings

# Disable MPS if on Apple Silicon (can be unstable)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print("⚠️  MPS detected - using CPU for stability")

# Import after setting environment variables
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def main():
    print("🚀 Minimal SFT Example (No Multiprocessing)")
    print("=" * 60)

    # Force CPU to avoid device issues
    device = 'cpu'
    print(f"\n✓ Using device: {device}")

    # 1. Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("   ✓ Tokenizer loaded")

    # 2. Load model (CPU only, no device_map)
    print("\n📦 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        low_cpu_mem_usage=True,  # Efficient loading without multiprocessing
        torch_dtype=torch.float32,  # Explicit dtype
    )
    model = model.to(device)
    model.eval()
    print(f"   ✓ Model loaded on {device}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Apply LoRA
    print("\n🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("   ✓ LoRA applied")

    # 4. Create sample data
    print("\n📚 Creating dataset...")
    train_data = [
        {"prompt": "What is the capital of France?", "response": "Paris"},
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "Who wrote Romeo and Juliet?", "response": "Shakespeare"},
    ]
    print(f"   ✓ Dataset created: {len(train_data)} examples")

    # 5. Test generation (the critical part)
    print("\n🎯 Testing generation...")
    print("   This is where the bus error typically occurs...")

    test_prompts = [
        "What is the capital",
        "Hello",
    ]

    for i, test_prompt in enumerate(test_prompts, 1):
        print(f"\n   Test {i}: '{test_prompt}'")

        try:
            # Tokenize
            inputs = tokenizer(
                test_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Verify device
            input_device = inputs["input_ids"].device
            model_device = next(model.parameters()).device
            print(f"      Input device: {input_device}")
            print(f"      Model device: {model_device}")

            # Generate - GREEDY first (most stable)
            print("      Generating (greedy)...")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=5,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"      ✓ Generated: '{generated_text}'")

            # Try sampling
            print("      Generating (sampling)...")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"      ✓ Generated: '{generated_text}'")

        except Exception as e:
            print(f"      ✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "!" * 60)
            print("ERROR DETAILS:")
            print("!" * 60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nIf you see 'bus error', this is a system-level issue.")
            print("Possible causes:")
            print("  1. macOS security/permissions issue")
            print("  2. Corrupted PyTorch installation")
            print("  3. Memory corruption")
            print("  4. Homebrew installation issue")
            print("\nTry:")
            print("  pip uninstall torch transformers peft")
            print("  pip install torch transformers peft")
            return False

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nGeneration works correctly!")
    print("You can now use:")
    print("  - scripts/train/train_sft.py for full training")
    print("  - notebooks/01_understanding_sft.ipynb for tutorials")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
