# Platform Compatibility Guide

This guide covers platform-specific considerations for running LLM post-training on different hardware.

## Supported Platforms

| Platform | GPU | Status | Notes |
|----------|-----|--------|-------|
| **Linux** | NVIDIA (CUDA) | ✅ Fully Supported | Best performance, all features work |
| **Google Colab** | T4/V100 (CUDA) | ✅ Fully Supported | Free GPU access, recommended for experiments |
| **macOS** | Apple Silicon (MPS) | ⚠️ Partial Support | Some operations not yet implemented |
| **macOS** | CPU | ✅ Fully Supported | Slower but stable |
| **Windows** | NVIDIA (CUDA) | ✅ Fully Supported | Same as Linux |
| **Windows** | CPU | ✅ Fully Supported | Slower but stable |

---

## Apple Silicon (MPS) - macOS

### What is MPS?

**MPS (Metal Performance Shaders)** is Apple's GPU acceleration framework for Apple Silicon chips (M1, M2, M3, M4).

**Why it exists:**
- Apple Silicon has integrated GPUs with unified memory architecture
- Very efficient for ML workloads when operations are supported
- CUDA only works on NVIDIA GPUs, so Apple needed their own solution

### Current Limitations

PyTorch added MPS support in version 2.0+, but **not all operations are implemented yet**.

#### Known Issues:

1. **`log_sigmoid` not supported** (affects reward modeling)
   - Error: `NotImplementedError: The operator 'aten::log_sigmoid_forward' is not currently implemented for the MPS device`
   - Tracking: https://github.com/pytorch/pytorch/issues/77764
   - **Workaround**: Use CPU or MPS fallback mode

2. **Some reduction operations cast int64 → int32**
   - Warning: `MPS: no support for int64 reduction ops, casting it to int32`
   - Usually harmless but can cause issues with very large sequences

3. **Limited support for advanced operations**
   - Some custom CUDA kernels don't have MPS equivalents
   - Complex attention mechanisms may fall back to CPU

### Workarounds

#### Option 1: Force CPU (Recommended for Development)

Most reliable for small models:

```bash
# Set in command line
python scripts/train/train_reward_model.py device=cpu

# Or set environment variable
export CUDA_VISIBLE_DEVICES=""
python scripts/train/train_reward_model.py
```

**Pros:**
- ✅ Everything works
- ✅ Stable and predictable
- ✅ Good for GPT-2, OPT-350m sized models

**Cons:**
- ❌ Slower than GPU (2-5x)
- ❌ Not practical for large models (> 1B params)

#### Option 2: MPS with CPU Fallback

Use MPS when possible, fall back to CPU for unsupported ops:

```bash
# Enable fallback mode
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train/train_reward_model.py device=mps
```

**Pros:**
- ✅ Uses GPU for supported operations
- ✅ Automatically falls back for unsupported ops
- ✅ More features work

**Cons:**
- ❌ Slower due to device transfers (CPU ↔ GPU)
- ❌ Complex debugging (which ops are falling back?)
- ❌ Still slower than native MPS or CUDA

#### Option 3: Use Cloud GPU (Recommended for Production)

For serious training, use Google Colab or cloud GPUs:

**Google Colab (Free):**
```python
# In Colab notebook
!git clone https://github.com/yourusername/llm-post-training.git
%cd llm-post-training
!pip install -e ".[gpu,experiment]"
!python scripts/train/train_reward_model.py experiment=reward_gpt2_hh_rlhf
```

**Benefits:**
- ✅ Free T4 GPU access
- ✅ All operations supported
- ✅ 10-50x faster than macOS CPU
- ✅ Can train larger models

### Performance Comparison

Training GPT-2 reward model on 1000 examples:

| Platform | Time | Notes |
|----------|------|-------|
| **macOS CPU** (M1/M2) | ~15 min | Stable, all features work |
| **macOS MPS** | ❌ Error | `log_sigmoid` not supported |
| **macOS MPS + Fallback** | ~8 min | Works but has overhead |
| **Google Colab T4** | ~2 min | 7x faster, fully supported |
| **NVIDIA RTX 4090** | ~1 min | 15x faster, best performance |

### When to Use Each Approach:

**Use CPU:**
- ✅ Developing and testing on macOS
- ✅ Small models (< 500M params)
- ✅ Quick experiments and debugging
- ✅ When stability > speed

**Use Colab/Cloud GPU:**
- ✅ Training production models
- ✅ Large models (> 1B params)
- ✅ Long training runs (> 1 hour)
- ✅ When speed matters

---

## NVIDIA GPUs (CUDA)

### Setup

**Linux/WSL:**
```bash
# Install CUDA toolkit (if not already installed)
# See: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project with GPU extras
pip install -e ".[gpu,experiment]"
```

**Verify CUDA:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Performance Tips

1. **Enable fp16/bf16 training:**
   ```bash
   python scripts/train/train_reward_model.py training.fp16=true
   ```
   - 2x faster
   - 2x less memory
   - Minimal accuracy impact

2. **Increase batch size:**
   ```bash
   python scripts/train/train_reward_model.py \
     training.per_device_train_batch_size=16 \
     training.gradient_accumulation_steps=2
   ```
   - Effective batch = 16 × 2 = 32
   - Better GPU utilization

3. **Use multiple GPUs:**
   ```bash
   # Automatic with HuggingFace Accelerate
   accelerate launch scripts/train/train_reward_model.py \
     experiment=reward_gpt2_hh_rlhf
   ```

---

## CPU-Only Training

Works on all platforms but slower.

### When to Use:

- ✅ No GPU available
- ✅ Testing on small datasets
- ✅ Debugging code
- ✅ Very small models (< 100M params)

### Optimization Tips:

1. **Use smaller batch sizes:**
   ```bash
   training.per_device_train_batch_size=2
   ```

2. **Reduce model size with LoRA:**
   ```bash
   model.use_lora=true
   ```

3. **Use fewer workers:**
   ```bash
   num_workers=0  # Avoid multiprocessing overhead
   ```

4. **Limit data:**
   ```bash
   data.num_train_examples=1000  # Train on subset
   ```

---

## Google Colab

### Free Tier

**GPU:** T4 (16GB VRAM)
**Runtime:** 12 hours max
**Cost:** Free

**Recommended for:**
- ✅ Experiments and prototyping
- ✅ Small to medium models (< 1B params)
- ✅ Learning and tutorials

**Limitations:**
- ❌ Session timeout after 12 hours
- ❌ Limited GPU time per day
- ❌ Need to reconnect periodically

### Colab Pro/Pro+

**GPU:** T4, V100, A100
**Runtime:** 24+ hours
**Cost:** $10-50/month

**Recommended for:**
- ✅ Longer training runs
- ✅ Larger models
- ✅ More reliable sessions

### Setup

See `notebooks/02_reward_modeling_colab.ipynb` for complete setup.

Quick start:
```python
# Clone repo
!git clone https://github.com/yourusername/llm-post-training.git
%cd llm-post-training

# Install with GPU support
!pip install -e ".[gpu,experiment]"

# Train
!python scripts/train/train_reward_model.py experiment=reward_gpt2_synthetic
```

---

## Cloud Platforms

### AWS

**Instances:** p3.2xlarge (V100), p4d.24xlarge (A100)

```bash
# On EC2 instance
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e ".[gpu,experiment]"
python scripts/train/train_reward_model.py experiment=reward_gpt2_hh_rlhf
```

### GCP

**Instances:** n1-highmem-8 + T4/V100

Similar setup to AWS.

### Lambda Labs

**GPU:** RTX 4090, A6000, H100
**Cost:** $0.50-2.00/hour

Good for serious training at lower cost than AWS/GCP.

---

## Troubleshooting

### "CUDA out of memory"

**Solutions:**
1. Reduce batch size: `training.per_device_train_batch_size=2`
2. Enable gradient checkpointing: `model.use_gradient_checkpointing=true`
3. Use 8-bit quantization: `model.use_8bit=true`
4. Use smaller model: `model=gpt2` instead of larger

### "MPS not supported"

**Solutions:**
1. Use CPU: `device=cpu`
2. Use Colab: Switch to cloud GPU
3. Enable fallback: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

### "DataLoader worker killed"

On macOS, multiprocessing can cause issues.

**Solution:**
```bash
num_workers=0  # Disable multiprocessing
```

### Slow on CPU

This is expected. CPU is 10-50x slower than GPU.

**Solutions:**
1. Use GPU (Colab or cloud)
2. Reduce dataset size
3. Use LoRA for efficiency
4. Be patient (it's still learning!)

---

## Summary

| Use Case | Recommended Platform | Command |
|----------|---------------------|---------|
| **Quick test** | macOS CPU | `device=cpu` |
| **Development** | macOS CPU | `device=cpu` |
| **Real training** | Google Colab | See Colab notebook |
| **Production** | AWS/GCP GPU | `training.fp16=true` |
| **Large models** | Cloud A100 | Multi-GPU setup |

**Bottom line:** For learning and development on macOS, use CPU. For serious training, use cloud GPUs.
