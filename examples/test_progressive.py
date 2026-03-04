"""
Progressive test to find exactly what breaks.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_raw_generation():
    """Test 1: Raw transformers (exactly like test_step_by_step.py Step 8)"""
    print("\n[Test 1] Raw transformers generation (like Step 8)")
    print("-" * 50)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Hello", return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Generated: '{text}'")
    return model, tokenizer


def test_with_device_move(model, tokenizer):
    """Test 2: Move model to device (like our wrapper does)"""
    print("\n[Test 2] With explicit device move")
    print("-" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = tokenizer("Hello", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Generated: '{text}'")


def test_with_left_padding():
    """Test 3: With left padding (like we added)"""
    print("\n[Test 3] With left padding_side")
    print("-" * 50)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # THIS IS WHAT WE ADDED

    inputs = tokenizer("Hello", return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Generated: '{text}'")


def test_with_sampling():
    """Test 4: With temperature (sampling) instead of greedy"""
    print("\n[Test 4] With temperature/sampling")
    print("-" * 50)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Hello", return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=True,  # Changed from False
            temperature=0.7,  # Added temperature
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Generated: '{text}'")


def test_with_low_cpu_mem():
    """Test 5: With low_cpu_mem_usage=True (like our wrapper)"""
    print("\n[Test 5] With low_cpu_mem_usage=True")
    print("-" * 50)

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,  # THIS IS WHAT WE USE
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Hello", return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Generated: '{text}'")


def test_with_attention_mask():
    """Test 6: Explicitly passing attention_mask"""
    print("\n[Test 6] With explicit attention_mask")
    print("-" * 50)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Hello", return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Explicitly pass this
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"✓ Generated: '{text}'")


def main():
    print("=" * 60)
    print("PROGRESSIVE TEST - Find What Breaks")
    print("=" * 60)

    tests = [
        ("Raw generation", test_raw_generation),
        ("With left padding", test_with_left_padding),
        ("With sampling", test_with_sampling),
        ("With low_cpu_mem_usage", test_with_low_cpu_mem),
        ("With attention_mask", test_with_attention_mask),
    ]

    model, tokenizer = None, None
    failed_at = None

    for i, (name, test_func) in enumerate(tests, 1):
        try:
            if i == 1:
                model, tokenizer = test_func()
            elif i == 2:
                test_with_device_move(model, tokenizer)
            else:
                test_func()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed_at = name
            break

    print("\n" + "=" * 60)
    if failed_at:
        print(f"❌ FAILED AT: {failed_at}")
    else:
        print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
