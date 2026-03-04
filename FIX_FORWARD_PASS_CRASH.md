# Fix Forward Pass Crash (Step 7)

## The Issue

Your system passes:
- ✅ PyTorch import (steps 1-2)
- ✅ Transformers import (step 3)
- ✅ Tokenizer (steps 4-5)
- ✅ Model loading (step 6)
- ❌ Forward pass (step 7) **← CRASHES HERE**

This means **specific PyTorch operations are broken**, not the installation itself.

## Most Likely Causes

1. **Specific BLAS/LAPACK operations broken** (matrix multiplication, attention)
2. **Apple Accelerate framework conflict** (on macOS)
3. **MKL threading issues**
4. **Corrupted NumPy or specific PyTorch operations**

## Fix 1: Reinstall with Different BLAS Backend

```bash
# Deactivate current environment
conda deactivate

# Create new environment WITHOUT MKL
conda create -n llmpt-nomkl python=3.10 nomkl -y
conda activate llmpt-nomkl

# Install PyTorch (forces OpenBLAS instead of MKL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transformers
pip install transformers peft

# Test
python examples/test_step_by_step.py
```

The `nomkl` flag forces use of OpenBLAS instead of Intel MKL, which often fixes matrix operation crashes on macOS.

## Fix 2: Install Specific PyTorch Version

Some PyTorch versions have bugs on macOS. Try a known-good version:

```bash
conda activate llmpt-py311  # or your environment

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install specific stable version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Test
python examples/test_step_by_step.py
```

## Fix 3: Force Single-Threaded Operations

The crash might be due to threading in matrix operations:

```bash
# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Test
python examples/test_step_by_step.py
```

Or add to your script:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
# Continue with imports...
```

## Fix 4: Reinstall NumPy

NumPy underlies PyTorch operations. A corrupted NumPy can cause forward pass crashes:

```bash
conda activate llmpt-py311

# Reinstall NumPy with specific version
pip uninstall numpy
pip install numpy==1.24.3

# Test
python examples/test_step_by_step.py
```

## Fix 5: Use OpenBLAS Explicitly

```bash
# Install OpenBLAS
conda install -c conda-forge openblas

# Reinstall PyTorch
pip uninstall torch
pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# Test
python examples/test_step_by_step.py
```

## Fix 6: Disable Apple Accelerate (macOS Specific)

macOS's Accelerate framework can conflict with PyTorch:

```bash
# Create environment without Accelerate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export BLAS=None
export LAPACK=None

# Reinstall PyTorch
pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# Test
python examples/test_step_by_step.py
```

## Test Script for Forward Pass Only

Let me create a focused test:

```python
# test_forward_only.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Tokenizing...")
inputs = tokenizer("Hello", return_tensors="pt")

print("Forward pass attempt 1: model(**inputs)")
try:
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Forward pass successful: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\nForward pass attempt 2: model(inputs['input_ids'])")
try:
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
    print(f"✓ Forward pass successful: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
```

## Quick Test Commands

Try each of these in order:

```bash
# Test 1: Single-threaded
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/test_step_by_step.py

# Test 2: Force CPU tensor operations
python -c "import torch; torch.set_num_threads(1); from transformers import AutoModelForCausalLM, AutoTokenizer; m = AutoModelForCausalLM.from_pretrained('gpt2'); t = AutoTokenizer.from_pretrained('gpt2'); i = t('Hello', return_tensors='pt'); o = m(**i); print('OK')"

# Test 3: Different dtype
python -c "import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; m = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float16); t = AutoTokenizer.from_pretrained('gpt2'); i = t('Hello', return_tensors='pt'); o = m(**i); print('OK')"
```

## Recommended Solution

Based on Step 7 failure, I recommend:

### Solution A (Most Likely to Work):

```bash
# Create new environment WITHOUT MKL
conda create -n llmpt-nomkl python=3.10 nomkl numpy=1.24.3 -y
conda activate llmpt-nomkl

# Install PyTorch
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install other packages
pip install transformers==4.35.0 peft==0.7.1

# Navigate to repo
cd /path/to/llm-post-training
pip install -e .

# Test
python examples/test_step_by_step.py
```

### Solution B (If A doesn't work):

```bash
# Use pip in virtualenv instead of conda
python3 -m venv ~/venv-llmpt-clean
source ~/venv-llmpt-clean/bin/activate

# Install with specific versions
pip install numpy==1.24.3
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.35.0 peft==0.7.1

# Test
python examples/test_step_by_step.py
```

## System Information to Check

Please also run and share:

```bash
# Check BLAS backend
python -c "import numpy; numpy.show_config()"

# Check PyTorch build
python -c "import torch; print(torch.__config__.show())"

# Check which BLAS PyTorch is using
python -c "import torch; print(torch.__config__.parallel_info())"
```

This will help identify which linear algebra library is causing the crash.

## Expected Result After Fix

After applying the fix:

```
[Step 7] Forward pass
--------------------------------------------------
Output shape: torch.Size([1, 2, 50257])
✓ SUCCESS

[Step 8] Generate text (CRITICAL)
--------------------------------------------------
...
✓ SUCCESS

============================================================
✅ ALL STEPS PASSED
```

## If Still Failing

If the forward pass still crashes after trying these fixes:

1. **Your Mac might have hardware/system issues**
2. **Try Docker** (bypasses all macOS issues):
   ```bash
   docker run -it -v $(pwd):/work python:3.10-slim bash
   cd /work
   pip install torch transformers peft
   python examples/test_step_by_step.py
   ```

3. **Use Google Colab** (guaranteed to work)
4. **Use a different machine** (Linux or different Mac)

The forward pass crash at Step 7 specifically points to a **linear algebra library issue**, so the `nomkl` solution (Solution A) should fix it.
