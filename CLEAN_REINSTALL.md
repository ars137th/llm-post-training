# Clean Reinstall Guide - Fix Corrupted Installation

## The Issue

If `test_raw_transformers.py` crashes with a bus error, your PyTorch/transformers installation is corrupted or incompatible. This is NOT a code issue - it's an installation issue.

## Nuclear Option: Complete Clean Reinstall

### Step 1: Identify Where It Crashes

Run this to see exactly where it fails:

```bash
python examples/test_raw_transformers.py 2>&1 | tee crash_log.txt
```

Look at the last line before the crash. Does it crash at:
- Import?
- Model loading?
- Generation?

### Step 2: Remove Everything

```bash
# Deactivate current environment
conda deactivate

# Remove the broken environment completely
conda env remove -n llmpt-py311

# Clear all caches
pip cache purge
conda clean --all --yes

# On macOS, also clear Homebrew cache
brew cleanup
```

### Step 3: Create Fresh Environment (Option A - Conda)

```bash
# Create new environment
conda create -n llmpt-fresh python=3.10 -y
conda activate llmpt-fresh

# Install PyTorch (CPU version for macOS)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch works BEFORE installing anything else
python -c "import torch; print(torch.__version__); x = torch.randn(5,5); print('PyTorch OK')"
```

**STOP HERE** - Does the PyTorch test work? If not, continue to Option B.

### Step 3: Create Fresh Environment (Option B - Virtualenv)

If conda has issues, use virtualenv instead:

```bash
# Use system Python (not Homebrew)
python3 -m venv ~/venvs/llmpt-venv

# Activate
source ~/venvs/llmpt-venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Test
python -c "import torch; print(torch.__version__); print('OK')"
```

### Step 4: Install Transformers (Carefully)

```bash
# Install transformers (specific version known to work)
pip install transformers==4.35.0

# Test immediately
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"

# If that works, install peft
pip install peft==0.7.1

# Test
python -c "from peft import LoraConfig; print('PEFT OK')"
```

### Step 5: Test Raw Transformers Again

```bash
cd /path/to/llm-post-training

# Test raw transformers
python examples/test_raw_transformers.py
```

**Does it work now?** If yes, continue. If no, go to Step 6.

### Step 6: If Still Failing - Check System

Your system may have deeper issues. Check:

```bash
# Check which Python you're using
which python
python --version

# Check if it's ARM64 or x86_64
file $(which python)
# Should say: "Mach-O 64-bit executable arm64" for Apple Silicon

# Check architecture consistency
python -c "import platform; print(platform.machine())"
# Should say: "arm64" for Apple Silicon

# Check if PyTorch is correct architecture
python -c "import torch; print(torch.__version__); print(torch.__file__)"
```

### Step 7: Try Intel x86_64 Python (If on Apple Silicon)

Sometimes ARM64 Python has issues. Try x86_64:

```bash
# Install x86_64 Python via Rosetta
arch -x86_64 /bin/bash
conda create -n llmpt-x86 python=3.10
conda activate llmpt-x86

# Install packages
pip install torch torchvision torchaudio transformers peft

# Test
python examples/test_raw_transformers.py
```

## Alternative: Use Official PyTorch Installation

### Option 1: Official PyTorch Wheels

```bash
# Create fresh environment
conda create -n llmpt-official python=3.10
conda activate llmpt-official

# Install from PyTorch.org official instructions
# For macOS (CPU):
pip3 install torch torchvision torchaudio

# Test
python -c "import torch; print(torch.__version__)"
python examples/test_raw_transformers.py
```

### Option 2: Install from Source (Last Resort)

If binary wheels are incompatible:

```bash
# Install dependencies
conda install cmake ninja

# Install PyTorch from source (takes ~30 min)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Debugging Specific Crashes

### If Crashes at Import

```bash
# Test PyTorch import
python -c "import torch"
# If this crashes, PyTorch itself is broken

# Reinstall PyTorch
pip uninstall torch
pip install torch --no-cache-dir
```

### If Crashes at Model Loading

```bash
# Test model loading
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype='auto'); print('OK')"

# Try with explicit dtype
python -c "import torch; from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32); print('OK')"
```

### If Crashes at Generation

This is the most common. Try:

```bash
# Test generation with no sampling
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
inputs = tokenizer('Hello', return_tensors='pt')
with torch.no_grad():
    output = model.generate(inputs['input_ids'], max_new_tokens=5, do_sample=False)
print('OK')
"
```

## System-Level Fixes

### Check macOS Security Settings

```bash
# Check if Python has necessary permissions
spctl --assess --verbose /opt/homebrew/anaconda3/envs/llmpt-py311/bin/python

# If blocked, allow
sudo spctl --add /opt/homebrew/anaconda3/envs/llmpt-py311/bin/python
```

### Check for Conflicting Libraries

```bash
# Check what's in your environment
conda list | grep -E "torch|numpy|mkl"

# Look for conflicts (multiple versions, incompatible builds)
```

### Increase System Limits

```bash
# Check limits
ulimit -a

# Increase if needed
ulimit -s 65532  # Stack size
ulimit -n 4096   # Open files
```

## The Nuclear Nuclear Option

If NOTHING works, there may be a fundamental system issue:

### Option 1: Use Google Colab Instead

```python
# In Google Colab notebook:
!git clone https://github.com/yourusername/llm-post-training
%cd llm-post-training
!pip install -r requirements/base.txt
!python examples/test_raw_transformers.py
```

### Option 2: Use Docker

```bash
# Install Docker Desktop for Mac
# https://www.docker.com/products/docker-desktop

# Run in container
docker run -it --rm -v $(pwd):/workspace python:3.10-slim bash

# Inside container
cd /workspace/llm-post-training
pip install torch transformers peft
python examples/test_raw_transformers.py
```

### Option 3: Use Remote Development

- Use GitHub Codespaces
- Use AWS Cloud9
- Use JupyterHub

## Verification Checklist

After reinstall, verify each step:

```bash
# 1. Python works
python --version

# 2. PyTorch works
python -c "import torch; print(torch.__version__)"

# 3. Basic tensor operations work
python -c "import torch; x = torch.randn(10,10); y = x @ x.T; print('OK')"

# 4. Transformers works
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2'); print('OK')"

# 5. Model loading works
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('gpt2'); print('OK')"

# 6. Generation works
python examples/test_raw_transformers.py

# 7. Our code works
python examples/minimal_sft_clean.py
```

## What to Report

If still failing after clean reinstall, report:

1. **Exact crash location**: Which test in `test_raw_transformers.py` crashes?
2. **System info**:
   ```bash
   uname -a
   sw_vers
   python --version
   pip list | grep -E "torch|transformers"
   ```
3. **Architecture info**:
   ```bash
   arch
   file $(which python)
   ```
4. **Crash details**: Any other error messages besides "bus error"?

## Expected Success

After clean reinstall, you should see:

```
[Test 1] Importing transformers...
✓ Import successful
[Test 2] Loading tokenizer...
✓ Tokenizer loaded
[Test 3] Loading model...
✓ Model loaded
[Test 4] Tokenizing...
✓ Tokenized: torch.Size([1, 2])
[Test 5] Forward pass...
✓ Forward pass: torch.Size([1, 2, 50257])
[Test 6] Generation...
  Attempting greedy generation...
✓ Greedy generation: 'Hello world, I'
  Attempting sampling generation...
✓ Sampling generation: 'Hello, there.'

============================================================
✅ ALL TESTS PASSED
```

No bus error, no semaphore warnings, everything works!
