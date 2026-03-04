# Complete macOS Installation Fix - All Issues Solved

## The Problem

Bus errors when running any code that uses this repository on macOS, even though raw transformers tests (like `test_step_by_step.py`) work fine.

## Root Cause

macOS has **BLAS library issues** with certain PyTorch/NumPy operations. The default Intel MKL (Math Kernel Library) or Apple Accelerate framework can cause bus errors during specific operations like:
- Forward pass (matrix multiplications in transformer layers)
- Model loading with `low_cpu_mem_usage=True`
- Certain memory management operations

## Complete Solution (All 5 Required Fixes)

After extensive testing, we identified **5 critical fixes** needed for stable operation on macOS:

### 1. ✅ Use `nomkl` Environment (OpenBLAS instead of Intel MKL)

**Why**: Intel MKL's BLAS implementation has bugs on some macOS systems, especially with certain CPU architectures. OpenBLAS is more stable.

```bash
# Create environment with nomkl flag
conda create -n llmpt-nomkl python=3.10 nomkl numpy=1.24.3 -y
conda activate llmpt-nomkl
```

The `nomkl` flag forces conda to use OpenBLAS instead of Intel MKL for linear algebra operations.

### 2. ✅ Use PyTorch 2.0.1 (NOT 2.4+)

**Why**: PyTorch 2.0.1 is tested and stable on macOS with nomkl/OpenBLAS. Later versions may have compatibility issues.

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 3. ✅ Use transformers <4.36.0 (Compatible with PyTorch 2.0.1)

**Why**: transformers 4.36+ **requires PyTorch >= 2.4**, which conflicts with Fix #2. We need transformers 4.35.x.

The repository's `requirements/base.txt` and `pyproject.toml` already have this constraint:

```
transformers>=4.35.0,<4.36.0  # 4.36+ requires PyTorch 2.4+
```

```bash
pip install "transformers>=4.35.0,<4.36.0"
```

### 4. ✅ Do NOT Use `low_cpu_mem_usage=True`

**Why**: This flag triggers internal memory operations that hit corrupted BLAS paths on macOS with nomkl. Ironically, it was meant to avoid multiprocessing issues but causes BLAS bugs instead.

**Already fixed** in `src/models/language.py`:

```python
# Before (BROKEN on macOS):
load_kwargs = {
    "trust_remote_code": trust_remote_code,
    "low_cpu_mem_usage": True,  # ← Triggers BLAS bug on macOS
    **model_kwargs,
}

# After (WORKS on macOS):
load_kwargs = {
    "trust_remote_code": trust_remote_code,
    **model_kwargs,
}
```

### 5. ✅ Set `padding_side="left"` for Decoder-Only Models

**Why**: Decoder-only models (GPT-2, LLaMA) need left padding for proper generation. Right padding causes attention issues.

**Already fixed** in `src/models/language.py`:

```python
# For decoder-only models, use left padding for generation
tokenizer.padding_side = "left"
```

## Complete Installation Steps

Follow these steps to set up a working environment on macOS:

### Step 1: Create Fresh Environment

```bash
# Deactivate any existing environment
conda deactivate

# Remove old broken environment (if exists)
conda env remove -n llmpt-nomkl

# Create new environment with nomkl and Python 3.10
conda create -n llmpt-nomkl python=3.10 nomkl numpy=1.24.3 -y

# Activate
conda activate llmpt-nomkl
```

### Step 2: Install PyTorch 2.0.1

```bash
# Install specific PyTorch version with matching torchvision/torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Repository

```bash
# Navigate to repository
cd /Users/akhil.shah/code/claude_sandbox/llm-post-training

# Install in editable mode (includes transformers <4.36.0 constraint)
pip install -e .
```

### Step 4: Verify Installation

```bash
# Test 1: PyTorch works
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Test 2: Transformers recognizes PyTorch
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"

# Test 3: All 8 diagnostic steps pass
python examples/test_step_by_step.py

# Test 4: Minimal SFT example works
python examples/minimal_sft.py
```

### Expected Output

If everything works, you should see:

```
[Step 1] Import PyTorch
✓ SUCCESS

[Step 2] Test PyTorch operation
✓ SUCCESS

[Step 3] Import transformers
✓ SUCCESS

[Step 4] Load tokenizer
✓ SUCCESS

[Step 5] Tokenize text
✓ SUCCESS

[Step 6] Load GPT-2 model
✓ SUCCESS

[Step 7] Forward pass
✓ SUCCESS

[Step 8] Generate text (CRITICAL)
✓ SUCCESS

============================================================
✅ ALL STEPS PASSED
```

## Troubleshooting

### If Step 7 Still Fails (Forward Pass)

Your BLAS library is still corrupted. Try single-threaded mode:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python examples/test_step_by_step.py
```

### If transformers Says "PyTorch Not Found"

You have transformers 4.36+ installed. Uninstall and reinstall correct version:

```bash
pip uninstall -y transformers tokenizers
pip install "transformers>=4.35.0,<4.36.0"
```

### If PyTorch Version Conflicts

Your environment has mismatched versions. Reinstall from scratch:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

## Why These Specific Versions?

| Component | Version | Reason |
|-----------|---------|--------|
| Python | 3.10 | Best compatibility with PyTorch 2.0.1 and transformers 4.35.x |
| nomkl flag | Required | Forces OpenBLAS instead of broken Intel MKL |
| NumPy | 1.24.3 | Stable with PyTorch 2.0.1 and OpenBLAS |
| PyTorch | 2.0.1 | Stable on macOS with nomkl/OpenBLAS |
| transformers | 4.35.x | Last version supporting PyTorch 2.0.x |
| torchvision | 0.15.2 | Compatible with torch 2.0.1 |
| torchaudio | 2.0.2 | Compatible with torch 2.0.1 |

## Alternative: Docker (If Nothing Works)

If you still have issues, use Docker to bypass all macOS problems:

```bash
# Pull Python image
docker pull python:3.10-slim

# Run container with repository mounted
docker run -it --rm \
  -v /Users/akhil.shah/code/claude_sandbox/llm-post-training:/workspace \
  python:3.10-slim \
  bash

# Inside container
cd /workspace
pip install torch==2.0.1 transformers==4.35.2 peft accelerate datasets
python examples/test_step_by_step.py
python examples/minimal_sft.py
```

Docker uses Linux internally, which doesn't have the macOS BLAS issues.

## Alternative: Google Colab (Easiest)

If you just want to run the code without local installation hassles:

1. Go to https://colab.research.google.com
2. Create new notebook
3. Clone and run:

```python
# In Colab notebook
!git clone https://github.com/yourusername/llm-post-training
%cd llm-post-training
!pip install -r requirements/base.txt
!python examples/minimal_sft.py
```

Google Colab provides free GPU and doesn't have macOS issues.

## Summary: What We Discovered

Through progressive testing (`examples/test_progressive.py`), we found that:

1. **Raw transformers works** (test_step_by_step.py passes all 8 steps)
2. **Adding LoRA doesn't break it** (PEFT is fine)
3. **Moving to device doesn't break it** (`.to(device)` is fine)
4. **Setting padding_side doesn't break it** (just removes a warning)
5. **Using `low_cpu_mem_usage=True` BREAKS IT** ← The culprit!

The `low_cpu_mem_usage=True` flag triggers specific memory management operations that hit corrupted BLAS library paths on macOS with the nomkl environment. Removing this flag allows everything to work.

## Files Modified

The repository code has been updated with these fixes:

1. **`requirements/base.txt`**: Constrained `transformers>=4.35.0,<4.36.0`
2. **`pyproject.toml`**: Constrained `transformers>=4.35.0,<4.36.0`
3. **`src/models/language.py`**:
   - Removed `low_cpu_mem_usage=True`
   - Added `tokenizer.padding_side = "left"`
   - Added comments explaining macOS BLAS issues

## Testing Tools Created

During debugging, we created these diagnostic scripts:

- **`examples/test_step_by_step.py`**: 8-step progressive test (import → generation)
- **`examples/test_raw_transformers.py`**: Tests raw transformers without custom code
- **`examples/test_progressive.py`**: Tests each configuration change individually
- **`examples/test_minimal_no_lora.py`**: Tests without LoRA to isolate PEFT issues

## References

- **CLEAN_REINSTALL.md**: Complete clean install guide with Docker/Colab alternatives
- **FIX_FORWARD_PASS_CRASH.md**: Details on Step 7 forward pass crashes
- **MACOS_FIX.md**: macOS multiprocessing and semaphore leak issues
- **TROUBLESHOOTING.md**: General troubleshooting guide

---

## Tested System Configuration

This solution was tested and verified on:

**Hardware**:
- **CPU**: Apple M1 Pro (ARM64 architecture)
- **Architecture**: arm64

**Operating System**:
- **macOS Version**: 26.2 (Build 25C56)
- **Kernel**: Darwin 25.2.0
- **Full Kernel**: Darwin Kernel Version 25.2.0: Tue Nov 18 21:09:40 PST 2025; root:xnu-12377.61.12~1/RELEASE_ARM64_T6000

**Working Environment** (`llmpt-nomkl`):
- **Python**: 3.10
- **NumPy**: 1.24.3
- **PyTorch**: 2.0.1 (CPU version with OpenBLAS via nomkl)
- **torchvision**: 0.15.2
- **torchaudio**: 2.0.2
- **transformers**: 4.35.x (specifically <4.36.0)
- **BLAS Backend**: OpenBLAS (forced via conda `nomkl` flag)

**Key Environment Variables**:
```bash
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

**Verification**:
- ✅ `test_step_by_step.py`: All 8 steps passed
- ✅ `minimal_sft.py`: Successful execution with generation
- ✅ No bus errors, no BLAS crashes, no semaphore leaks

**Notes**:
- Base conda environment uses Python 3.9.13, but working environment uses Python 3.10
- The `nomkl` flag is **critical** on Apple Silicon (M1 Pro) to avoid Intel MKL BLAS bugs
- Standard `device_map="auto"` and `low_cpu_mem_usage=True` both trigger BLAS crashes on this system

---

**Last Updated**: March 3, 2026
**Status**: ✅ All issues resolved, `minimal_sft.py` runs successfully on macOS
**Tested On**: Apple M1 Pro, macOS 26.2, Darwin 25.2.0
