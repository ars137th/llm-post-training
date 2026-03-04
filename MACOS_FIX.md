# macOS Bus Error & Multiprocessing Fix

## The Issue

On macOS (especially with Homebrew and Apple Silicon), you may encounter:
```
resource_tracker: There appear to be 1 leaked semaphore objects
zsh: bus error
```

This happens because:
1. **Tokenizers library** spawns parallel processes by default
2. **macOS uses spawn instead of fork** for multiprocessing (more strict)
3. **Homebrew Python** may have different multiprocessing behavior
4. **Apple MPS backend** can be unstable with multiprocessing

## Quick Fix: Try This First

```bash
# Run the no-multiprocessing version
python examples/minimal_sft_no_mp.py
```

This version completely disables multiprocessing and should work.

## Permanent Fix: Set Environment Variables

Add these to your shell profile (`~/.zshrc` or `~/.bash_profile`):

```bash
# Disable multiprocessing issues
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bash_profile
```

## Fix Method 1: Environment Variables Before Running

Set these every time you run:

```bash
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python examples/minimal_sft.py
```

## Fix Method 2: Add to Your Python Scripts

Put this **at the very top** of your script, before any imports:

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Now import other libraries
import torch
from transformers import ...
```

## Fix Method 3: Reinstall PyTorch from Scratch

Sometimes Homebrew Python has issues with PyTorch. Reinstall cleanly:

```bash
# Completely remove PyTorch
pip uninstall torch torchvision torchaudio
pip cache purge

# Reinstall (CPU version for macOS)
pip install torch torchvision torchaudio

# Reinstall transformers and peft
pip install --upgrade transformers peft accelerate

# Test
python -c "import torch; print(torch.__version__)"
```

## Fix Method 4: Use Miniforge Instead of Anaconda

Anaconda/Homebrew can have conflicts. Miniforge is better for M1/M2/M3 Macs:

```bash
# Install Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh

# Create new environment
conda create -n llmpt python=3.11
conda activate llmpt

# Install packages
pip install torch torchvision torchaudio
pip install transformers peft accelerate datasets

# Navigate to repo
cd /path/to/llm-post-training
pip install -e .

# Test
python examples/minimal_sft_no_mp.py
```

## Understanding the Semaphore Leak

The warning:
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

Means a multiprocessing semaphore wasn't properly cleaned up. This happens when:
1. Tokenizers library creates parallel processes
2. Process crashes before cleanup
3. Semaphore remains in system memory

This is harmless but indicates the multiprocessing issue.

## Why macOS is Different

### macOS Multiprocessing Behavior

| System | Default Method | Issue |
|--------|---------------|-------|
| Linux | `fork` | Fast, copies memory |
| macOS | `spawn` | Slow, creates new process |
| Windows | `spawn` | Slow, creates new process |

macOS uses `spawn` which:
- Creates entirely new Python interpreter
- Doesn't share memory with parent
- More likely to have semaphore leaks
- More strict about cleanup

### Apple Silicon Considerations

M1/M2/M3 Macs have additional issues:
- MPS (Metal Performance Shaders) backend is newer
- Some PyTorch operations not optimized for MPS
- Fallback to CPU can cause device mismatches

## Verify Your Fix

After applying fixes, test:

```bash
# Test 1: No multiprocessing version
python examples/minimal_sft_no_mp.py

# Test 2: Original version (should work now)
python examples/minimal_sft.py

# Test 3: Check for warnings
python examples/minimal_sft.py 2>&1 | grep -i "semaphore\|bus error"
# Should see no output
```

## If Still Failing

### Check System Limits

macOS has strict limits on semaphores:

```bash
# Check current limits
sysctl kern.sysv.semmni
sysctl kern.sysv.semmns

# These should be at least:
# kern.sysv.semmni: 87381
# kern.sysv.semmns: 87381
```

### Check for Corrupted Installation

```bash
# Check if PyTorch works at all
python -c "import torch; x = torch.randn(10, 10); print('OK')"

# Check if transformers works
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2'); print('OK')"

# Check if PEFT works
python -c "from peft import LoraConfig; print('OK')"
```

### Check Homebrew Python Issues

```bash
# Check which Python you're using
which python
# Should be: /opt/homebrew/anaconda3/envs/llmpt-py311/bin/python

# Check if it's Homebrew
python -c "import sys; print(sys.prefix)"

# If using Homebrew Python directly, consider switching to conda
```

## Nuclear Option: Clean Reinstall

If nothing works:

```bash
# 1. Remove conda environment
conda deactivate
conda env remove -n llmpt-py311

# 2. Clear pip cache
pip cache purge

# 3. Clear conda cache
conda clean --all

# 4. Create fresh environment
conda create -n llmpt-fresh python=3.11
conda activate llmpt-fresh

# 5. Install ONLY what's needed
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft accelerate datasets
pip install rich pyyaml omegaconf

# 6. Test immediately (before installing anything else)
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('gpt2'); print('OK')"

# 7. If that works, install the repo
cd /path/to/llm-post-training
pip install -e .

# 8. Test
python examples/minimal_sft_no_mp.py
```

## Alternative: Use Docker

If macOS continues to have issues, use Docker:

```bash
# Pull PyTorch image
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Run container
docker run -it -v $(pwd):/workspace pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash

# Inside container
cd /workspace/llm-post-training
pip install -r requirements/base.txt
pip install -e .
python examples/minimal_sft.py
```

## Debugging Commands

```bash
# Check for leaked semaphores
ipcs -s

# Clean up leaked semaphores (if any)
ipcs -s | grep $(whoami) | awk '{print $2}' | xargs -n1 ipcrm -s

# Monitor during execution
python examples/minimal_sft.py &
PID=$!
watch -n 1 "ipcs -s | grep $(whoami)"
```

## Expected Behavior After Fix

You should see:
```
🚀 Minimal SFT Example
==================================================

📦 Loading model...
   ✓ Model loaded

📚 Creating dataset...
🔧 Processing data...

✅ Setup complete!
Model parameters: 124,439,808
Trainable parameters: 294,912

🎯 Testing generation...
   ✓ Generated: What is the capital of France? Paris...

==================================================
✨ Example complete!
```

**NO semaphore warnings, NO bus errors!**

## Summary of Fixes

1. ✅ Use `minimal_sft_no_mp.py` - Disables all multiprocessing
2. ✅ Set `TOKENIZERS_PARALLELISM=false` - Disables tokenizer parallelism
3. ✅ Set `OMP_NUM_THREADS=1` - Disables OpenMP threading
4. ✅ Reinstall PyTorch cleanly
5. ✅ Use Miniforge instead of Anaconda
6. ✅ Force CPU mode to avoid MPS issues

Pick the fix that works for your setup!
