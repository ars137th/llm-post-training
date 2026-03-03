# Setup Guide

Detailed installation and setup instructions for the LLM Post-Training repository.

## Prerequisites

- Python 3.9 or higher
- pip or conda
- (Optional) NVIDIA GPU with CUDA support for faster training
- (Optional) 16GB+ GPU memory for small models, 24GB+ for larger models

## Installation Methods

### Method 1: pip (Recommended for most users)

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
pip install -r requirements/base.txt

# Install the package in development mode
pip install -e .

# Optional: Install additional dependencies
pip install -r requirements/rlhf.txt         # For RLHF/PPO/DPO
pip install -r requirements/multimodal.txt   # For vision-language models
pip install -r requirements/dev.txt          # For development/testing
```

### Method 2: conda

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training

# Create conda environment
conda create -n llm-post-training python=3.10
conda activate llm-post-training

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install -r requirements/base.txt
pip install -e .
```

### Method 3: Docker (Coming soon)

## Verification

Test your installation:

```bash
# Test imports
python -c "import torch; import transformers; import peft; print('✓ All imports successful!')"

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run minimal example
python examples/minimal_sft.py
```

## GPU Setup

### CUDA Installation

If you have an NVIDIA GPU but CUDA is not installed:

1. Check your GPU: `nvidia-smi`
2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Verify installation: `nvcc --version`

### Memory Requirements

Approximate GPU memory requirements with LoRA:

- **GPT-2 (124M)**: 2-4GB
- **OPT-350M**: 4-6GB
- **LLaMA-1B**: 6-8GB
- **Mistral-7B (with QLoRA)**: 12-16GB

Without LoRA, multiply by ~3-4x.

### Multiple GPUs

For multi-GPU training, install additional dependencies:

```bash
pip install deepspeed>=0.12.0
```

## Optional Dependencies

### Quantization (4-bit/8-bit)

For training larger models on limited GPU memory:

```bash
pip install bitsandbytes>=0.41.0
```

**Note**: Requires CUDA. On Windows, may need to install from source.

### Flash Attention 2

For 2-4x faster training (requires CUDA and compilation):

```bash
pip install flash-attn --no-build-isolation
```

**Note**: This can take 10-15 minutes to compile.

### Experiment Tracking

```bash
# Weights & Biases
pip install wandb
wandb login  # Follow prompts to login

# TensorBoard (included in base requirements)
# No additional setup needed
```

## Troubleshooting

### Issue: `ImportError: cannot import name 'X' from 'transformers'`

**Solution**: Update transformers
```bash
pip install --upgrade transformers
```

### Issue: CUDA out of memory

**Solutions**:
1. Use smaller batch size
2. Enable LoRA (reduces memory by 90%)
3. Use gradient accumulation
4. Enable 4-bit quantization with QLoRA
5. Reduce max sequence length

Example config:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

model:
  use_lora: true
  use_4bit: true
```

### Issue: Slow CPU training

**Solution**: CPU training is slow but works. For faster results:
- Use smaller models (GPT-2, OPT-350M)
- Reduce dataset size
- Consider using Google Colab (free GPU) or cloud GPUs

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution**: Install package in development mode
```bash
pip install -e .
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: bitsandbytes installation fails

**Solution**:
- On Windows: May need to install from source or use WSL
- On Mac: bitsandbytes requires CUDA, not supported on Mac. Remove from requirements.
- On Linux: Ensure CUDA is properly installed

## Next Steps

After installation:

1. **Quick Start**: Run `python examples/minimal_sft.py`
2. **Tutorial**: Open `notebooks/00_setup_and_quickstart.ipynb`
3. **Documentation**: Read `docs/guides/quickstart.md`
4. **Training**: Try `python scripts/train/train_sft.py`

## Getting Help

- Check the troubleshooting section above
- Search existing issues: https://github.com/yourusername/llm-post-training/issues
- Open a new issue with:
  - Your environment (OS, Python version, GPU)
  - Full error message
  - Steps to reproduce

## Development Setup

For contributing to the repository:

```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/
```

See `CONTRIBUTING.md` for more details.
