"""
Step-by-Step Test to Find Exact Crash Point

Run this to identify exactly where the bus error occurs.

Usage:
    python examples/test_step_by_step.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

def test_step(step_num, description, test_func):
    """Test a single step and report result."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)
    try:
        test_func()
        print(f"✓ SUCCESS")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def step1_import_torch():
    """Test PyTorch import."""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch location: {torch.__file__}")


def step2_torch_operation():
    """Test basic PyTorch operation."""
    import torch
    x = torch.randn(10, 10)
    y = torch.matmul(x, x.T)
    print(f"Created tensor: {y.shape}")


def step3_import_transformers():
    """Test transformers import."""
    from transformers import __version__
    print(f"Transformers version: {__version__}")


def step4_load_tokenizer():
    """Test tokenizer loading."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


def step5_tokenize():
    """Test tokenization."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inputs = tokenizer("Hello world", return_tensors="pt")
    print(f"Tokenized shape: {inputs['input_ids'].shape}")


def step6_load_model():
    """Test model loading."""
    import torch
    from transformers import AutoModelForCausalLM
    print("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    )
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")


def step7_forward_pass():
    """Test forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inputs = tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Output shape: {outputs.logits.shape}")


def step8_generate_greedy():
    """Test greedy generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing input...")
    inputs = tokenizer("Hello", return_tensors="pt")

    print("Generating (greedy)...")
    print("THIS IS WHERE BUS ERRORS TYPICALLY OCCUR")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: '{text}'")


def main():
    print("=" * 60)
    print("STEP-BY-STEP DIAGNOSTIC TEST")
    print("=" * 60)
    print("\nThis will identify exactly where the bus error occurs.")
    print("Watch the output carefully!")

    steps = [
        (1, "Import PyTorch", step1_import_torch),
        (2, "Test PyTorch operation", step2_torch_operation),
        (3, "Import transformers", step3_import_transformers),
        (4, "Load tokenizer", step4_load_tokenizer),
        (5, "Tokenize text", step5_tokenize),
        (6, "Load GPT-2 model", step6_load_model),
        (7, "Forward pass", step7_forward_pass),
        (8, "Generate text (CRITICAL)", step8_generate_greedy),
    ]

    failed_at = None

    for step_num, description, test_func in steps:
        success = test_step(step_num, description, test_func)
        if not success:
            failed_at = step_num
            break

    print("\n" + "=" * 60)
    if failed_at:
        print(f"❌ FAILED AT STEP {failed_at}")
        print("=" * 60)

        if failed_at <= 2:
            print("\nPyTorch itself is broken!")
            print("Fix: Reinstall PyTorch")
            print("  pip uninstall torch")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")

        elif failed_at <= 5:
            print("\nTransformers/tokenizers is broken!")
            print("Fix: Reinstall transformers")
            print("  pip uninstall transformers tokenizers")
            print("  pip install transformers")

        elif failed_at <= 7:
            print("\nModel loading or forward pass is broken!")
            print("This is unusual. Your installation may be corrupted.")
            print("Fix: Complete reinstall")
            print("  See CLEAN_REINSTALL.md")

        elif failed_at == 8:
            print("\nGeneration causes bus error!")
            print("This is the most common issue on macOS.")
            print("\nYour installation is corrupted OR")
            print("There's a system-level incompatibility.")
            print("\nFixes to try:")
            print("  1. Complete clean reinstall (see CLEAN_REINSTALL.md)")
            print("  2. Use virtualenv instead of conda")
            print("  3. Try x86_64 Python (if on Apple Silicon)")
            print("  4. Use Docker or Google Colab instead")
    else:
        print("✅ ALL STEPS PASSED")
        print("=" * 60)
        print("\nIf you're still seeing bus errors in other scripts,")
        print("the issue might be specific to those scripts.")
        print("But raw transformers works correctly!")

    return failed_at is None


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
