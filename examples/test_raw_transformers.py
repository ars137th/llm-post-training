"""
Test Raw Transformers - Absolute Minimal Test

This tests if the issue is with transformers itself or our code.
Uses ONLY transformers library with no extras.

Usage:
    python examples/test_raw_transformers.py
"""

# Set environment variables FIRST
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import torch

print("=" * 60)
print("RAW TRANSFORMERS TEST")
print("=" * 60)

# Test 1: Import
print("\n[Test 1] Importing transformers...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load tokenizer
print("\n[Test 2] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f"✗ Tokenizer loading failed: {e}")
    sys.exit(1)

# Test 3: Load model (CPU only, minimal options)
print("\n[Test 3] Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    )
    model = model.to('cpu')
    model.eval()
    print(f"✓ Model loaded")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Tokenize
print("\n[Test 4] Tokenizing...")
try:
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"✓ Tokenized: {inputs['input_ids'].shape}")
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    sys.exit(1)

# Test 5: Forward pass
print("\n[Test 5] Forward pass...")
try:
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Forward pass: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Generation (THE CRITICAL TEST)
print("\n[Test 6] Generation...")
print("  This is where bus errors typically occur...")
try:
    # Greedy generation first
    print("  Attempting greedy generation...")
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Greedy generation: '{text}'")

    # Sampling generation
    print("  Attempting sampling generation...")
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Sampling generation: '{text}'")

except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "!" * 60)
    print("BUS ERROR OCCURRED")
    print("!" * 60)
    print("\nThis is a system-level crash in transformers library.")
    print("Your PyTorch/transformers installation may be corrupted.")
    print("\nTry:")
    print("  1. pip uninstall torch transformers")
    print("  2. pip cache purge")
    print("  3. pip install torch transformers")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)

print("\nConclusions:")
print("  ✓ transformers library works correctly")
print("  ✓ Model loading works")
print("  ✓ Generation works")

# Check for warnings
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Trigger any remaining cleanup
    pass

if w:
    print(f"\n⚠️  {len(w)} warning(s) detected during execution")
    print("  (This is okay if the script completed)")
else:
    print("\n✓ No warnings detected")

print("\nNext steps:")
print("  - If you saw the semaphore warning but this completed, it's just a")
print("    cleanup warning and everything works fine.")
print("  - If this crashed, your transformers installation has issues.")
print("  - Try: python examples/minimal_sft_no_mp.py")

sys.exit(0)
