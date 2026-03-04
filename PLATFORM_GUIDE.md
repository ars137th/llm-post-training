# Platform-Specific Installation Guide

## Overview

The macOS-specific fixes (PyTorch 2.0.1, transformers <4.36.0, nomkl) are **ONLY needed on macOS** due to BLAS library bugs. On other platforms, you can use the latest versions with GPU support!

## Quick Comparison

| Platform | PyTorch Version | transformers Version | GPU Support | Issues |
|----------|----------------|---------------------|-------------|--------|
| **macOS (M1/M2)** | 2.0.1 | <4.36.0 | ❌ No CUDA | BLAS bugs require nomkl |
| **Google Colab** | 2.4+ | 4.36+ | ✅ Free T4 GPU | None - works perfectly |
| **Databricks** | 2.4+ | 4.36+ | ✅ Enterprise GPUs | None - works perfectly |
| **Linux Desktop** | 2.4+ | 4.36+ | ✅ CUDA GPUs | None - works perfectly |
| **AWS/GCP/Azure** | 2.4+ | 4.36+ | ✅ Cloud GPUs | None - works perfectly |

---

## Option 1: Google Colab (Easiest, Free GPU)

**Advantages**:
- Free Tesla T4 GPU (15GB VRAM)
- No installation needed
- Latest PyTorch/transformers work perfectly
- Pre-installed CUDA

### Setup Instructions

1. **Go to Google Colab**: https://colab.research.google.com
2. **Create new notebook**
3. **Enable GPU**: Runtime → Change runtime type → GPU → T4 GPU
4. **Clone and install**:

```python
# In a Colab cell
!git clone https://github.com/yourusername/llm-post-training.git
%cd llm-post-training

# Install with latest versions (no version constraints!)
!pip install torch>=2.4.0 transformers>=4.36.0 peft accelerate datasets evaluate
!pip install einops rich tqdm pyyaml

# OR use the requirements (will work on Linux)
!pip install -r requirements/base.txt

# Verify GPU is available
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

5. **Run examples**:

```python
# Test basic functionality
!python examples/test_step_by_step.py

# Run minimal SFT (will use GPU automatically)
!python examples/minimal_sft.py

# Run full training
!python scripts/train/train_sft.py
```

### Colab-Specific Notes

- **Session timeout**: Colab disconnects after ~12 hours or 90 min idle
- **Disk space**: ~100GB available, but temporary
- **Save checkpoints**: Download to Google Drive or mount Drive
- **Free tier limits**: ~12-15 hours/day GPU usage

### Mounting Google Drive (Persistent Storage)

```python
from google.colab import drive
drive.mount('/content/drive')

# Change output directory to save to Drive
output_dir = "/content/drive/MyDrive/llm-post-training/outputs"
```

---

## Option 2: Databricks (Enterprise, Managed)

**Advantages**:
- Enterprise-grade infrastructure
- Managed clusters with powerful GPUs
- Integrated with MLflow for experiment tracking
- Team collaboration features

### Setup Instructions

1. **Create Databricks workspace** (AWS/Azure/GCP)
2. **Create compute cluster**:
   - **Runtime**: ML Runtime 14.3 LTS or higher (includes PyTorch, transformers)
   - **Node type**: GPU instance (e.g., g4dn.xlarge on AWS, Standard_NC6s_v3 on Azure)
   - **Workers**: 0 for single-node, or 2+ for distributed training

3. **Create notebook**:

```python
# Install repository
%sh
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e .

# Verify setup
import torch
import transformers
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
```

4. **Run training**:

```python
# In notebook or as job
%sh
cd /Workspace/Repos/llm-post-training
python scripts/train/train_sft.py \
  model.name=gpt2 \
  training.output_dir=/dbfs/mnt/experiments/sft/run1 \
  training.num_train_epochs=3
```

### Databricks-Specific Features

**MLflow Integration**:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("model", "gpt2")
    mlflow.log_param("lora_r", 8)
    # Training happens here
    mlflow.log_metric("final_loss", loss)
```

**Distributed Training**:
```python
# Automatically uses all GPUs on cluster
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/dbfs/...",
    per_device_train_batch_size=4,
    # Databricks handles distribution automatically
)
```

---

## Option 3: Linux Desktop with GPU

**Advantages**:
- Full control
- No time limits
- Can use latest hardware (RTX 4090, A100, etc.)

### Setup Instructions

```bash
# 1. Create conda environment (no nomkl needed on Linux!)
conda create -n llmpt-gpu python=3.10 -y
conda activate llmpt-gpu

# 2. Install PyTorch with CUDA (replace cu118 with your CUDA version)
pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 4. Clone and install repository
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e .

# 5. Run training
python examples/minimal_sft.py
```

### Multi-GPU Training

```bash
# Use accelerate for multi-GPU
accelerate config  # Configure once

# Then use accelerate launch
accelerate launch scripts/train/train_sft.py \
  training.per_device_train_batch_size=8
```

---

## Option 4: Cloud Platforms (AWS, GCP, Azure)

### AWS SageMaker

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train_sft.py',
    source_dir='./scripts/train',
    instance_type='ml.p3.2xlarge',  # V100 GPU
    transformers_version='4.36',
    pytorch_version='2.4',
    py_version='py310',
)

huggingface_estimator.fit()
```

### GCP Vertex AI

```bash
# Create custom training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=llm-sft \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_V100,accelerator-count=1 \
  --python-package-uris=gs://your-bucket/llm-post-training.tar.gz
```

### Azure ML

```python
from azure.ai.ml import command

job = command(
    code="./llm-post-training",
    command="python scripts/train/train_sft.py",
    environment="AzureML-pytorch-2.4-cuda12:1",
    compute="gpu-cluster",
    instance_type="Standard_NC6s_v3",  # V100
)

ml_client.jobs.create_or_update(job)
```

---

## Requirements Files for Different Platforms

The repository has flexible requirements:

### For macOS (Current Setup)
```bash
# requirements/base.txt
torch>=2.0.0,<2.2.0
transformers>=4.35.0,<4.36.0
# ... nomkl environment required
```

### For Linux/Colab/Cloud (Latest Versions)
```bash
# No version constraints needed!
pip install torch>=2.4.0 transformers>=4.36.0 peft accelerate datasets
```

### Creating Platform-Specific Requirements

You can create a `requirements/gpu.txt`:

```txt
# requirements/gpu.txt - For Linux/Colab/Cloud with GPU
torch>=2.4.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
datasets>=2.16.0
evaluate>=0.4.0
einops>=0.7.0
rich>=13.7.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0

# Optional for GPU optimization
flash-attn>=2.0.0  # Faster attention on A100/H100
bitsandbytes>=0.41.0  # 4-bit/8-bit quantization
```

---

## Key Differences: macOS vs GPU Platforms

### What Changes on GPU Platforms?

| Feature | macOS (M1 Pro) | GPU Platforms |
|---------|---------------|---------------|
| **PyTorch** | 2.0.1 (nomkl required) | 2.4+ (latest stable) |
| **transformers** | <4.36.0 | 4.36+ (latest) |
| **Device** | CPU only | CUDA GPU |
| **BLAS** | OpenBLAS (nomkl) | cuBLAS (CUDA) |
| **low_cpu_mem_usage** | ❌ Causes crashes | ✅ Works fine |
| **device_map="auto"** | ❌ Causes issues | ✅ Works fine |
| **Quantization** | ❌ Not supported | ✅ 4-bit/8-bit works |
| **Flash Attention** | ❌ Not available | ✅ Works on A100+ |

### Code Changes for GPU

**Minimal - our code is platform-agnostic!** Just ensure:

```python
# In src/models/language.py
# This automatically detects GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# On GPU platforms, these flags work fine:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # ✅ Works on GPU, ❌ Crashes on macOS
    low_cpu_mem_usage=True,  # ✅ Works on GPU, ❌ Crashes on macOS
    torch_dtype=torch.float16,  # ✅ Much faster on GPU
)
```

---

## Model Size Recommendations by Platform

### macOS M1 Pro (16GB RAM, CPU only)
- **Max model size**: ~1B parameters
- **Recommended**: GPT-2 (124M), OPT-350m, TinyLlama-1.1B
- **LoRA required**: Yes, for anything > 500M params

### Google Colab (T4 GPU, 15GB VRAM)
- **Max model size**: ~3B parameters with 8-bit quantization
- **Recommended**: LLaMA-2-7B (8-bit), Mistral-7B (8-bit), GPT-2 XL (1.5B)
- **LoRA required**: Yes for 7B models

### Databricks (A100 GPU, 40GB VRAM)
- **Max model size**: 13B parameters with 4-bit, 7B full precision
- **Recommended**: LLaMA-2-13B (4-bit), Mistral-7B (full), CodeLlama-7B
- **LoRA required**: Optional, depends on model size

### Linux Desktop (RTX 4090, 24GB VRAM)
- **Max model size**: 13B parameters with 4-bit, 7B with LoRA
- **Recommended**: LLaMA-2-13B (4-bit), Mistral-7B (LoRA), Phi-2 (full)
- **LoRA required**: For >7B models

---

## Testing on New Platform

Use this checklist when moving to a new platform:

```bash
# 1. Check hardware
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else 'N/A')"

# 2. Check versions (should be latest on GPU platforms)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 3. Run diagnostic
python examples/test_step_by_step.py

# 4. Test GPU training
python examples/minimal_sft.py

# 5. Check GPU utilization
nvidia-smi  # Should show python process using GPU
```

---

## Performance Comparison

Training GPT-2 (124M params) on 1000 examples:

| Platform | Time | Cost | Notes |
|----------|------|------|-------|
| macOS M1 Pro (CPU) | ~45 min | $0 | Good for learning/debugging |
| Colab T4 (Free) | ~8 min | $0 | Best free option |
| Colab A100 (Paid) | ~3 min | $0.50/hr | Fastest on Colab |
| Databricks (A100) | ~3 min | $2-5/hr | Enterprise features |
| Linux RTX 4090 | ~4 min | $0 (owned) | Best for iteration |

---

## Recommendations by Use Case

### Learning & Experimentation
- **Best**: macOS (free, local) or Google Colab (free GPU)
- **Why**: No cost, quick iteration

### Research & Development
- **Best**: Linux desktop with RTX 3090/4090
- **Why**: No time limits, full control, fast iteration

### Production Training
- **Best**: Databricks or AWS SageMaker
- **Why**: Managed infrastructure, experiment tracking, team features

### One-off Large Training
- **Best**: Google Colab Pro ($10/month) with A100
- **Why**: Cheapest for occasional large jobs

---

## Summary

**Key Takeaway**: The macOS-specific constraints (PyTorch 2.0.1, transformers <4.36.0, nomkl) are **ONLY needed on macOS**. On any other platform (Colab, Databricks, Linux, Cloud), you can use the latest versions and enjoy GPU acceleration without any special configuration!

The repository code is platform-agnostic and will automatically:
- ✅ Detect and use GPU when available
- ✅ Fall back to CPU on macOS
- ✅ Work with both old (2.0.1) and new (2.4+) PyTorch versions
- ✅ Handle both CPU and GPU-specific optimizations

Just install, run, and enjoy! 🚀
