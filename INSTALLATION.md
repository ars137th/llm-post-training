# Installation Guide

## Quick Install by Platform

### macOS (Apple Silicon M1/M2/M3)
```bash
# Create nomkl environment
conda create -n llmpt-nomkl python=3.10 nomkl numpy=1.24.3 -y
conda activate llmpt-nomkl

# Install PyTorch 2.0.1 (compatible with nomkl)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install repository (uses macOS-compatible versions)
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e .
```

### Linux/Google Colab/Cloud (GPU)
```bash
# Clone and install with latest versions
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e ".[gpu]"  # Uses PyTorch 2.4+, transformers 4.36+
```

### Alternative: macOS Specific Install
```bash
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e ".[macos]"  # Uses PyTorch 2.0.x, transformers 4.35.x
```

---

## Detailed Installation Options

We provide **three ways** to install, depending on your platform:

### Method 1: Direct pip install (Flexible - Adapts to Platform)

Uses flexible version constraints, lets pip choose compatible versions:

```bash
pip install -e .
```

**Versions installed**:
- PyTorch: 2.0.0+ (whatever's available on your platform)
- transformers: 4.35.0+ (whatever's compatible with your PyTorch)

**Works on**: All platforms, but may get newer versions than needed on macOS
**Recommended**: Use Method 2 or 3 for explicit platform control

---

### Method 2: Platform-specific extras (Recommended)

Uses platform-specific version constraints:

```bash
# For macOS
pip install -e ".[macos]"

# For GPU platforms
pip install -e ".[gpu]"
```

**Versions installed**:
- **[macos]**: PyTorch 2.0.x, transformers 4.35.x (BLAS-compatible)
- **[gpu]**: PyTorch 2.4.0+, transformers 4.36.0+ (latest)

**Recommended**: Use this method for explicit control

---

### Method 3: Manual requirements files

#### For macOS:
```bash
pip install -r requirements/base.txt
```

Uses PyTorch 2.0.1, transformers <4.36.0 (macOS-compatible)

#### For GPU platforms:
```bash
pip install -r requirements/gpu.txt
```

Uses PyTorch 2.4+, transformers 4.36+ (latest versions)

---

## Installation by Platform

### macOS (Detailed)

**Why special setup needed?**: macOS has BLAS library bugs that cause crashes. See [MACOS_COMPLETE_FIX.md](MACOS_COMPLETE_FIX.md) for details.

```bash
# Step 1: Create nomkl environment (forces OpenBLAS)
conda create -n llmpt-nomkl python=3.10 nomkl numpy=1.24.3 -y
conda activate llmpt-nomkl

# Step 2: Install PyTorch 2.0.1 (compatible with nomkl)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Step 3: Clone repository
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training

# Step 4: Install (uses conservative versions)
pip install -e .

# Step 5: Verify
python examples/test_step_by_step.py
```

---

### Google Colab

**Why it's different**: Colab runs Linux with GPU, so latest versions work perfectly!

```python
# In a Colab cell

# Enable GPU
# Runtime → Change runtime type → T4 GPU

# Clone repository
!git clone https://github.com/yourusername/llm-post-training.git
%cd llm-post-training

# Option 1: Minimal install (for examples/minimal_sft.py only)
!pip install -e ".[gpu]"

# Option 2: Full install (RECOMMENDED - includes all features)
!pip install -e ".[all-gpu]"

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Run examples
!python examples/minimal_sft.py

# Run full training scripts (requires [experiment] extras)
!python scripts/train/train_sft.py
```

**Recommendation**: Use `[all-gpu]` to avoid import errors when running training scripts.

---

### Linux Desktop with GPU

```bash
# Step 1: Create environment
conda create -n llmpt-gpu python=3.10 -y
conda activate llmpt-gpu

# Step 2: Install PyTorch with CUDA (replace cu118 with your CUDA version)
pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: Clone and install with GPU extras
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e ".[gpu]"

# Step 4: Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python examples/test_step_by_step.py
```

---

### Databricks

```python
# In a Databricks notebook

# Install from GitHub
%sh
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e ".[gpu]"  # Latest versions

# Verify
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## Optional Dependencies

The repository has modular optional dependencies for different features.

### Available Extras

| Extra | Includes | Use Case | Required For |
|-------|----------|----------|--------------|
| `[gpu]` | PyTorch 2.4+, transformers 4.36+ | GPU platforms | Latest versions on Linux/Colab/Cloud |
| `[macos]` | PyTorch 2.0.x, transformers 4.35.x | macOS | BLAS-compatible versions |
| `[experiment]` | **hydra-core**, wandb, tensorboard | Config & tracking | **`scripts/train/*.py`** |
| `[rlhf]` | trl | RLHF training | PPO/DPO with TRL |
| `[multimodal]` | Pillow, torchvision, clip-score | Vision models | CLIP, LLaVA |
| `[quantization]` | bitsandbytes | 4-bit/8-bit | QLoRA, memory optimization |
| `[dev]` | pytest, black, mypy, etc. | Development | Testing, formatting |
| `[all]` | All above (base versions) | Everything | All features (macOS versions) |
| `[all-gpu]` | All above (GPU versions) | Everything | All features (GPU versions) |

### Installation Examples

```bash
# Install base + experiment extras (for scripts/train/*.py)
pip install -e ".[gpu,experiment]"

# Install base + multiple extras
pip install -e ".[gpu,experiment,quantization]"

# Install everything with GPU optimization (recommended for Colab)
pip install -e ".[all-gpu]"

# Install everything with macOS versions
pip install -e ".[all]"
```

### Which Extras Do I Need?

**For `examples/minimal_sft.py`**:
- ✅ Base install is enough: `pip install -e ".[gpu]"`

**For `scripts/train/train_sft.py`**:
- ⚠️ Needs `[experiment]`: `pip install -e ".[gpu,experiment]"`
- Requires **hydra-core** for configuration management

**For Google Colab** (recommended):
- ✅ Install everything: `pip install -e ".[all-gpu]"`
- Includes all features (W&B, quantization, etc.)

**For macOS**:
- ✅ Use `[macos]` or `[all]`: `pip install -e ".[macos,experiment]"`

---

## Comparison: What Gets Installed?

### `pip install -e .` (Base - macOS Compatible)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.0.0-2.2.0 | macOS nomkl compatible |
| transformers | 4.35.0-4.36.0 | Compatible with PyTorch 2.0.x |
| peft | 0.7.0+ | Latest |
| accelerate | 0.25.0+ | Latest |
| datasets | 2.16.0+ | Latest |

**Use when**: Running on macOS or want maximum compatibility

---

### `pip install -e ".[gpu]"` (GPU - Latest Versions)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.4.0+ | Latest stable |
| transformers | 4.36.0+ | Latest with new features |
| peft | 0.7.0+ | Latest |
| accelerate | 0.25.0+ | Latest |
| datasets | 2.16.0+ | Latest |

**Use when**: Running on Linux/Colab/Cloud with GPU

---

## Requirements File Reference

We provide three requirements files:

### `requirements/base.txt` - macOS Compatible
```txt
torch>=2.0.0,<2.2.0
transformers>=4.35.0,<4.36.0
# ... other packages
```
**Use on**: macOS, or anywhere you want conservative versions

### `requirements/gpu.txt` - GPU Optimized
```txt
torch>=2.4.0
transformers>=4.36.0
# ... other packages
```
**Use on**: Linux, Colab, Cloud with GPU

### `requirements/rlhf.txt`, `requirements/multimodal.txt`, etc.
Optional dependencies for specific features

---

## Environment Variables (macOS)

On macOS, set these before running:

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Or in your Python script:
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

**Not needed on**: Linux, Colab, Cloud

---

## Verification

After installation, verify everything works:

```bash
# Test all components
python examples/test_step_by_step.py

# Should see:
# ✓ Step 1-8 all pass
# ✅ ALL STEPS PASSED

# Test training
python examples/minimal_sft.py

# Should see:
# ✅ Setup complete!
# 🎯 Testing generation...
# ✨ Example complete!
```

---

## Troubleshooting

### "Bus error" on macOS
**Solution**: Follow the [MACOS_COMPLETE_FIX.md](MACOS_COMPLETE_FIX.md) guide. You need the nomkl environment.

### "PyTorch >= 2.4 is required but found 2.0.1"
**Problem**: Using GPU-optimized install on macOS
**Solution**: Use `pip install -e .` instead of `pip install -e ".[gpu]"`

### "CUDA not available" on Linux
**Problem**: PyTorch installed without CUDA support
**Solution**: Reinstall PyTorch with correct CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Version conflicts
**Solution**: Start fresh:
```bash
pip uninstall -y torch transformers
# Then follow platform-specific instructions above
```

### Import Errors (ModuleNotFoundError)

Common import errors and their solutions:

#### `ModuleNotFoundError: No module named 'hydra'`

**When**: Running `scripts/train/train_sft.py` or other training scripts
**Problem**: Missing `[experiment]` extras
**Solution**:
```bash
# Add experiment extras
pip install -e ".[gpu,experiment]"

# Or install everything
pip install -e ".[all-gpu]"
```

**Why**: Training scripts use Hydra for configuration management, which is in optional `[experiment]` dependencies.

#### `ModuleNotFoundError: No module named 'trl'`

**When**: Running RLHF/PPO/DPO scripts
**Problem**: Missing `[rlhf]` extras
**Solution**:
```bash
pip install -e ".[gpu,rlhf]"
```

#### `ModuleNotFoundError: No module named 'bitsandbytes'`

**When**: Using 4-bit or 8-bit quantization
**Problem**: Missing `[quantization]` extras
**Solution**:
```bash
pip install -e ".[gpu,quantization]"
```

#### `ModuleNotFoundError: No module named 'PIL'` or `'torchvision'`

**When**: Using vision-language models (CLIP, LLaVA)
**Problem**: Missing `[multimodal]` extras
**Solution**:
```bash
pip install -e ".[gpu,multimodal]"
```

#### Quick Fix for All Import Errors

Install everything:
```bash
# On GPU platforms
pip install -e ".[all-gpu]"

# On macOS
pip install -e ".[all]"
```

### Configuration Errors (Hydra/OmegaConf)

#### `ConfigAttributeError: Key 'name' is not in struct`

**When**: Running `scripts/train/train_sft.py`
**Error message**:
```
omegaconf.errors.ConfigAttributeError: Key 'name' is not in struct
    full_key: model.name
    object_type=dict
```

**Problem**: Hydra can't find the config files
**Cause**: You're running from a directory where the relative path to configs doesn't work
**Solution**: This is now fixed in the latest version. Pull the latest changes:

```bash
git pull origin master
```

**What was fixed**: The script now uses absolute paths to find configs, so it works from any directory (local, Colab, Databricks).

**Alternative**: If still having issues, run from repository root:
```bash
cd /path/to/llm-post-training
python scripts/train/train_sft.py
```

### Dataset Loading Errors

#### `RuntimeError: Dataset scripts are no longer supported`

**When**: Loading certain HuggingFace datasets
**Error message**:
```
RuntimeError: Dataset scripts are no longer supported, but found daily_dialog.py
```

**Problem**: The dataset uses a loading script (.py file) which is deprecated for security reasons in newer `datasets` library versions.

**Solution**: This is now fixed. The default config uses `wikitext` which doesn't require scripts.

**If using other datasets**: Use datasets that don't require loading scripts, or provide local data files:

```bash
# Override with a script-free dataset
python scripts/train/train_sft.py data.dataset_name=wikitext data.dataset_config=wikitext-2-raw-v1

# Or use your own data
python scripts/train/train_sft.py data.data_files=/path/to/your/data.jsonl
```

**Working datasets** (no scripts needed):
- `wikitext` (text completion)
- `openwebtext` (web text)
- `c4` (Common Crawl)
- `wikipedia` (Wikipedia articles)

#### `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`

**When**: Running training scripts with transformers 4.36+
**Problem**: API change in transformers 4.36+

**Solution**: This is now fixed. The script uses `eval_strategy` instead of `evaluation_strategy`.

```bash
git pull origin master  # Get the fix
```

**Why this happened**: transformers 4.36+ renamed `evaluation_strategy` to `eval_strategy` for consistency. This only affects GPU platforms using transformers 4.36+ (Colab, Linux). macOS with transformers 4.35.x is unaffected.

#### `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`

**When**: Running training scripts with transformers 4.36+
**Problem**: API change - Trainer no longer accepts tokenizer parameter

**Solution**: This is now fixed. The SFTTrainer stores tokenizer separately.

```bash
git pull origin master  # Get the fix
```

**Why this happened**: transformers 4.36+ removed the `tokenizer` parameter from `Trainer.__init__()`. Our custom trainer now stores it as an instance variable instead of passing it to the parent class.

#### `TypeError: SFTTrainer.training_step() takes 3 positional arguments but 4 were given`

**When**: Training starts but fails on first batch with transformers 4.36+
**Problem**: API change - `training_step()` now requires `num_items_in_batch` parameter

**Solution**: This is now fixed. The method signature has been updated.

```bash
git pull origin master  # Get the fix
```

**Why this happened**: transformers 4.36+ added a `num_items_in_batch` parameter to `training_step()`. Our custom trainer now accepts this parameter with a default value for backwards compatibility.

---

## Quick Reference

| Platform | Command | PyTorch | transformers |
|----------|---------|---------|--------------|
| macOS | `pip install -e ".[macos]"` | 2.0.x | 4.35.x |
| Colab | `pip install -e ".[gpu]"` | 2.4+ | 4.36+ |
| Linux GPU | `pip install -e ".[gpu]"` | 2.4+ | 4.36+ |
| Databricks | `pip install -e ".[gpu]"` | 2.4+ | 4.36+ |
| Flexible | `pip install -e .` | Latest available | Latest compatible |

---

## What's the Difference?

### Why two installation methods?

**macOS has BLAS bugs** with:
- Intel MKL (Math Kernel Library)
- PyTorch 2.4+ with certain operations
- transformers 4.36+ (requires PyTorch 2.4+)

**GPU platforms work fine** with:
- Latest PyTorch 2.4+
- Latest transformers 4.36+
- All features and optimizations

### Should I use `[gpu]` on macOS?
**❌ NO** - It will cause bus errors. Use `pip install -e ".[macos]"` or requirements/base.txt

### Can I use base install on GPU platforms?
**✅ YES** - It will work, but versions will depend on what's available on your platform. Use `pip install -e ".[gpu]"` for explicit latest versions.

### Best practice?
- **macOS**: Use `pip install -e ".[macos]"` or `pip install -r requirements/base.txt && pip install -e .`
- **GPU platforms**: Use `pip install -e ".[gpu]"`

---

## Summary

We provide flexible installation to support all platforms:

**One-line install**:
```bash
# macOS
pip install -e ".[macos]"

# GPU platforms
pip install -e ".[gpu]"
```

That's it! The repository automatically handles platform differences. 🚀

For more details:
- **macOS issues**: See [MACOS_COMPLETE_FIX.md](MACOS_COMPLETE_FIX.md)
- **Platform guide**: See [PLATFORM_GUIDE.md](PLATFORM_GUIDE.md)
- **Setup guide**: See [SETUP.md](SETUP.md)
