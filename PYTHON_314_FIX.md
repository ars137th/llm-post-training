# Python 3.14 Compatibility Fix

## The Issue

You're using **Python 3.14** (currently in alpha/beta), which has multiprocessing changes that are incompatible with PyTorch's model loading and text generation. This causes:
- Semaphore leaks: `resource_tracker: There appear to be 1 leaked semaphore objects`
- Bus errors (segmentation faults) during generation

## The Solution

**Downgrade to Python 3.10 or 3.11** (recommended for best compatibility)

### Option 1: Create New Conda Environment (Recommended)

```bash
# Create new environment with Python 3.11
conda create -n llmpt-py311 python=3.11

# Activate it
conda activate llmpt-py311

# Navigate to repo
cd /path/to/llm-post-training

# Install dependencies
pip install -r requirements/base.txt
pip install -e .

# Test
python examples/minimal_sft.py
```

### Option 2: Update Existing Environment

```bash
# Downgrade Python in current environment
conda install python=3.11

# Reinstall PyTorch (important!)
pip install --force-reinstall torch torchvision torchaudio

# Reinstall other packages
pip install --force-reinstall transformers peft accelerate

# Test
python examples/minimal_sft.py
```

### Option 3: Use pyenv (Alternative to Conda)

```bash
# Install Python 3.11
pyenv install 3.11.7

# Create virtual environment
pyenv virtualenv 3.11.7 llmpt

# Activate
pyenv activate llmpt

# Install dependencies
pip install -r requirements/base.txt
pip install -e .
```

## Temporary Workaround (If You Must Use Python 3.14)

While not recommended, you can try forcing CPU mode and different multiprocessing methods:

```bash
# Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Force spawn method for multiprocessing
python -c "import multiprocessing; multiprocessing.set_start_method('spawn', force=True)"

# Run with CPU mode
python examples/minimal_sft_safe.py --cpu
```

Or add to your script:

```python
import multiprocessing
import os

# Force spawn method (more compatible)
multiprocessing.set_start_method('spawn', force=True)

# Disable threading
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Then run your code
```

## Why Python 3.14 Causes Issues

Python 3.14 (PEP 703) introduces changes to the GIL (Global Interpreter Lock) and multiprocessing that affect:
1. **PyTorch's C++ extensions** - May not be compiled for Python 3.14 yet
2. **Multiprocessing spawn behavior** - Changed defaults and behavior
3. **Transformers' model parallelism** - Uses accelerate which spawns processes
4. **CUDA/MPS backends** - May have race conditions with new multiprocessing

## Verify Your Fix

After downgrading, verify:

```bash
# Check Python version
python --version
# Should show: Python 3.10.x or 3.11.x

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Test basic operations
python -c "import torch; x = torch.randn(10, 10); print('OK')"

# Test model loading
python examples/debug_sft.py

# Test generation
python examples/minimal_sft.py
```

## Expected Output (After Fix)

```
🚀 Minimal SFT Example
==================================================

📦 Loading model...
📚 Creating dataset...
🔧 Processing data...
⚙️  Setting up training...

✅ Setup complete!
Model parameters: 124,439,808
Trainable parameters: 294,912
Training examples: 3

🎯 Testing generation...
Prompt: What is the capital of France?
Generated: What is the capital of France? Paris is the capital...

==================================================
✨ Example complete!
```

## Additional Resources

- PyTorch Compatibility: https://pytorch.org/get-started/previous-versions/
- Python Version Support: Most ML libraries officially support Python 3.8-3.11
- Transformers Compatibility: https://github.com/huggingface/transformers#installation

## Still Having Issues?

If you still get bus errors after downgrading:

1. **Check your conda environment:**
   ```bash
   conda list | grep python
   which python
   ```

2. **Completely remove and recreate:**
   ```bash
   conda deactivate
   conda env remove -n llmpt
   conda create -n llmpt python=3.11
   conda activate llmpt
   # Reinstall everything fresh
   ```

3. **Try without LoRA:**
   ```bash
   python examples/minimal_sft_safe.py --no-lora
   ```

4. **Run diagnostics:**
   ```bash
   python examples/debug_sft.py
   ```

5. **Check system resources:**
   - Activity Monitor (Mac): Check available memory
   - Close other applications
   - Restart your terminal/IDE

## Why We Fixed the Code Too

Even though the root cause is Python 3.14, we improved the code to:
1. Use `low_cpu_mem_usage=True` - Avoids unnecessary multiprocessing
2. Only use `device_map="auto"` when needed (quantization)
3. Add Python version warning
4. Better device handling

These changes improve stability even on Python 3.10-3.11!
