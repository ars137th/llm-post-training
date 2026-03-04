"""
Debug script to diagnose SFT issues

This script runs various checks to identify the cause of bus errors.

Usage:
    python examples/debug_sft.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import platform


def check_environment():
    """Check system environment."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"\nSystem: {platform.system()}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("  (Apple Silicon detected)")

    print(f"\nDefault device: {'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'}")


def test_basic_operations():
    """Test basic PyTorch operations."""
    print("\n" + "=" * 60)
    print("BASIC OPERATIONS TEST")
    print("=" * 60)

    try:
        # Test CPU operations
        print("\n1. Testing CPU operations...")
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = torch.matmul(x, y)
        print("   ✓ CPU operations work")

        # Test GPU operations if available
        if torch.cuda.is_available():
            print("\n2. Testing CUDA operations...")
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print("   ✓ CUDA operations work")

        # Test MPS operations if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n2. Testing MPS operations...")
            try:
                x_mps = x.to('mps')
                y_mps = y.to('mps')
                z_mps = torch.matmul(x_mps, y_mps)
                print("   ✓ MPS operations work")
            except Exception as e:
                print(f"   ✗ MPS operations failed: {e}")
                print("   → Will use CPU instead")

        return True
    except Exception as e:
        print(f"   ✗ Basic operations failed: {e}")
        return False


def test_model_loading():
    """Test loading GPT-2 model."""
    print("\n" + "=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)

    try:
        print("\n1. Testing transformers import...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("   ✓ Transformers imported")

        print("\n2. Loading GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("   ✓ Tokenizer loaded")

        print("\n3. Loading GPT-2 model (CPU)...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        print("   ✓ Model loaded on CPU")

        print("\n4. Testing forward pass...")
        input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids)
        print("   ✓ Forward pass works")

        return True, model, tokenizer
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_generation(model, tokenizer):
    """Test text generation."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    try:
        print("\n1. Setting up for generation...")
        model.eval()

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   → Set pad_token = eos_token")

        print("\n2. Encoding test prompt...")
        test_prompt = "Hello"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Device: {inputs['input_ids'].device}")

        print("\n3. Generating (greedy decoding)...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.pad_token_id,
            )
        print(f"   Output shape: {outputs.shape}")

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text}'")
        print("   ✓ Generation works (greedy)")

        print("\n4. Generating (sampling)...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text}'")
        print("   ✓ Generation works (sampling)")

        return True
    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_loading():
    """Test loading model with LoRA."""
    print("\n" + "=" * 60)
    print("LORA LOADING TEST")
    print("=" * 60)

    try:
        print("\n1. Testing PEFT import...")
        from peft import LoraConfig, get_peft_model
        print("   ✓ PEFT imported")

        print("\n2. Loading base model...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        print("   ✓ Base model loaded")

        print("\n3. Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("   ✓ LoRA applied")

        model.print_trainable_parameters()

        return True
    except Exception as e:
        print(f"   ✗ LoRA loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_our_wrapper():
    """Test our LanguageModel wrapper."""
    print("\n" + "=" * 60)
    print("LANGUAGEMODEL WRAPPER TEST")
    print("=" * 60)

    try:
        print("\n1. Importing LanguageModel...")
        from src.models.language import LanguageModel
        print("   ✓ Import successful")

        print("\n2. Loading model WITHOUT LoRA...")
        model = LanguageModel.from_pretrained("gpt2", use_lora=False)
        print("   ✓ Model loaded")
        print(f"   Device: {model.device}")

        print("\n3. Testing generation...")
        from src.data.processors.text import TextProcessor
        processor = TextProcessor(model.tokenizer, max_length=128)

        test_prompt = "Hello"
        encoded = processor.tokenize(test_prompt, return_tensors="pt")

        # Force CPU
        encoded = {k: v.to('cpu') for k, v in encoded.items()}
        if hasattr(model.model, 'to'):
            model.model = model.model.to('cpu')
        model.device = 'cpu'

        print(f"   Model device: {model.device}")
        print(f"   Input device: {encoded['input_ids'].device}")

        model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
            )

        generated = processor.decode(output[0])
        print(f"   Generated: '{generated}'")
        print("   ✓ Generation works WITHOUT LoRA")

        print("\n4. Loading model WITH LoRA...")
        model_lora = LanguageModel.from_pretrained("gpt2", use_lora=True)
        print("   ✓ Model with LoRA loaded")

        # Force CPU
        if hasattr(model_lora.model, 'to'):
            model_lora.model = model_lora.model.to('cpu')
        model_lora.device = 'cpu'

        print("\n5. Testing generation with LoRA...")
        encoded = processor.tokenize(test_prompt, return_tensors="pt")
        encoded = {k: v.to('cpu') for k, v in encoded.items()}

        model_lora.eval()
        with torch.no_grad():
            output = model_lora.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=model_lora.tokenizer.pad_token_id,
            )

        generated = processor.decode(output[0])
        print(f"   Generated: '{generated}'")
        print("   ✓ Generation works WITH LoRA")

        return True
    except Exception as e:
        print(f"   ✗ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "🔍 " * 20)
    print("LLM POST-TRAINING DIAGNOSTIC SCRIPT")
    print("🔍 " * 20)

    results = {}

    # Check environment
    check_environment()

    # Test basic operations
    results['basic_ops'] = test_basic_operations()

    if not results['basic_ops']:
        print("\n" + "!" * 60)
        print("CRITICAL: Basic PyTorch operations failed!")
        print("Please reinstall PyTorch or check your installation.")
        print("!" * 60)
        return

    # Test model loading
    success, model, tokenizer = test_model_loading()
    results['model_loading'] = success

    if not success:
        print("\n" + "!" * 60)
        print("CRITICAL: Could not load GPT-2 model!")
        print("!" * 60)
        return

    # Test generation
    results['generation'] = test_generation(model, tokenizer)

    if not results['generation']:
        print("\n" + "!" * 60)
        print("CRITICAL: Text generation failed!")
        print("This is where the bus error likely occurs.")
        print("!" * 60)

    # Test LoRA
    results['lora'] = test_lora_loading()

    # Test our wrapper
    results['wrapper'] = test_our_wrapper()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    if all(results.values()):
        print("\n✅ All tests passed! The issue may be specific to minimal_sft.py")
        print("   Try running: python examples/minimal_sft.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        print("\nCommon fixes:")
        print("  1. Reinstall PyTorch: pip install --upgrade torch")
        print("  2. If on Mac, make sure not using CUDA/bitsandbytes")
        print("  3. Try CPU-only mode")
        print("  4. Check memory: Activity Monitor (Mac) or top (Linux)")


if __name__ == "__main__":
    main()
