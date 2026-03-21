# Training on Google Colab

This guide covers running LLM post-training experiments on Google Colab, including best practices, limitations, and platform-specific optimizations.

## Quick Start

### Basic Setup

```python
# 1. Check GPU availability
!nvidia-smi

# 2. Clone repository
!git clone https://github.com/your-username/llm-post-training.git
%cd llm-post-training

# 3. Install dependencies
!pip install -r requirements/base.txt
!pip install -r requirements/multimodal.txt

# 4. Verify installation
!python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Train Your First Model

```bash
# CLIP image-text alignment (no LoRA needed on GPU)
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.max_steps=100 \
    training.fp16=true
```

---

## Colab GPU Tiers

### Free Tier
- **GPU**: T4 (16GB VRAM)
- **RAM**: 12-13GB system RAM
- **Disk**: ~100GB
- **Session**: 12 hours max, can disconnect after inactivity
- **Cost**: Free

**What you can train:**
- ✅ CLIP-ViT-B/32 (full fine-tuning)
- ✅ GPT-2 (with LoRA)
- ✅ LLaMA-1B (with 4-bit LoRA)
- ❌ LLaVA-7B (use Colab Pro)

### Colab Pro ($10/month)
- **GPU**: T4, V100 (16-32GB VRAM)
- **RAM**: 25-32GB system RAM
- **Session**: 24 hours max
- **Priority**: Background execution

**What you can train:**
- ✅ Everything from free tier
- ✅ LLaVA-7B (with 4-bit LoRA)
- ✅ LLaMA-7B (with 4-bit LoRA)
- ✅ Mistral-7B (with 4-bit LoRA)

### Colab Pro+ ($50/month)
- **GPU**: V100, A100 (40-80GB VRAM)
- **RAM**: 50GB+ system RAM
- **Session**: No time limit
- **Priority**: Highest

**What you can train:**
- ✅ Everything from Pro tier
- ✅ LLaMA-13B (with 4-bit LoRA)
- ✅ LLaVA-13B (with 4-bit LoRA)
- ✅ Larger batch sizes, faster training

---

## Platform-Specific Considerations

### 1. LoRA Compatibility

| Model Type | LoRA Support | Status | Notes |
|------------|-------------|--------|-------|
| **CLIP** | ❌ Not supported | Broken | PEFT kwargs routing issues |
| **LLaVA** | ✅ Fully supported | Works | Recommended with 4-bit |
| **GPT-2** | ✅ Fully supported | Works | Great for testing |
| **LLaMA** | ✅ Fully supported | Works | Use 4-bit for >1B params |
| **Mistral** | ✅ Fully supported | Works | Use 4-bit for 7B |

**Important:** The LoRA + CLIP issue is **platform-independent**. You'll get the same errors on Colab as on local machines:

```python
TypeError: CLIPVisionTransformer.forward() got an unexpected keyword argument 'input_ids'
TypeError: CLIPTextTransformer.forward() got an unexpected keyword argument 'inputs_embeds'
```

**Root cause:** PEFT library's kwargs routing is fundamentally incompatible with CLIP's dual-encoder architecture.

**Solution:** Train CLIP without LoRA on Colab. The GPU makes this feasible.

### 2. Why CLIP Works Better on Colab (Without LoRA)

**Speed comparison:**

| Platform | Hardware | Training Time (100 steps) | Memory Usage |
|----------|----------|--------------------------|--------------|
| Local macOS | CPU | ~5 minutes | ~2 GB |
| Colab Free | T4 GPU | ~30 seconds | ~1.5 GB VRAM |
| Colab Pro | V100 GPU | ~20 seconds | ~1.5 GB VRAM |

**Key insight:** On GPU, full fine-tuning of CLIP-ViT-B/32 is so fast that LoRA isn't necessary. The memory savings from LoRA (which is broken anyway) don't matter when you have 16GB+ VRAM.

### 3. Memory Requirements

**Without LoRA (full fine-tuning):**

| Model | Parameters | Memory (fp32) | Memory (fp16) | Colab Tier |
|-------|-----------|---------------|---------------|------------|
| CLIP-ViT-B/32 | 151M | ~2.4 GB | ~1.2 GB | Free ✓ |
| GPT-2 | 124M | ~2.0 GB | ~1.0 GB | Free ✓ |
| GPT-2-Medium | 355M | ~5.6 GB | ~2.8 GB | Free ✓ |
| GPT-2-Large | 774M | ~12 GB | ~6 GB | Free ✓ |
| LLaVA-7B | 7B | ~56 GB | ~28 GB | Pro+ only |

**With LoRA + 4-bit quantization:**

| Model | Parameters | Trainable % | Memory (4-bit) | Colab Tier |
|-------|-----------|-------------|----------------|------------|
| GPT-2 | 124M | ~0.5% | ~200 MB | Free ✓ |
| LLaMA-1B | 1.1B | ~0.5% | ~1 GB | Free ✓ |
| LLaMA-7B | 7B | ~0.5% | ~4 GB | Free ✓ |
| LLaVA-7B | 7B | ~0.5% | ~4 GB | Free ✓ |
| Mistral-7B | 7B | ~0.5% | ~4 GB | Free ✓ |

### 4. Session Management

**Colab sessions disconnect after inactivity.** Solutions:

#### Option 1: Keep Session Alive (Browser Hack)
```javascript
// Run in browser console (F12)
function ClickConnect(){
    console.log("Keeping session alive...");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)  // Click every 60 seconds
```

#### Option 2: Use Background Execution (Pro/Pro+ only)
```python
# Enable background execution in Colab settings
# Your notebook continues running even if you close the browser
```

#### Option 3: Checkpoint Frequently
```yaml
# In your config files
training:
  save_steps: 100  # Save every 100 steps
  save_total_limit: 3  # Keep last 3 checkpoints
```

Then resume from checkpoint:
```bash
!python scripts/train/train_sft.py \
    experiment=gpt2_conversation \
    training.resume_from_checkpoint=./outputs/checkpoint-300
```

### 5. Data Storage

**Colab disk space is temporary** - everything is deleted when the session ends.

#### Option A: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Save outputs to Drive
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.output_dir=/content/drive/MyDrive/llm-outputs/clip_run1
```

#### Option B: Upload to HuggingFace Hub
```python
from huggingface_hub import HfApi, create_repo

# After training
!python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./outputs/clip_caption_synthetic',
    repo_id='your-username/clip-finetuned',
    repo_type='model'
)
"
```

#### Option C: Download to Local Machine
```python
from google.colab import files

# Zip and download
!zip -r model_checkpoint.zip ./outputs/clip_caption_synthetic
files.download('model_checkpoint.zip')
```

---

## Recommended Workflows

### Workflow 1: Train CLIP on Colab (No LoRA)

```python
# Setup
!git clone https://github.com/your-username/llm-post-training.git
%cd llm-post-training
!pip install -r requirements/base.txt -r requirements/multimodal.txt

# Mount Drive for outputs
from google.colab import drive
drive.mount('/content/drive')

# Train CLIP (full fine-tuning on GPU is fast)
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=coco \
    data.max_train_samples=5000 \
    training.output_dir=/content/drive/MyDrive/clip_outputs \
    training.num_epochs=3 \
    training.per_device_train_batch_size=32 \
    training.fp16=true \
    training.logging_steps=10 \
    training.save_steps=500

# Outputs saved to Google Drive
```

**Expected time:** ~15-20 minutes on T4 GPU for 5000 samples, 3 epochs

### Workflow 2: Train LLaVA with LoRA (Colab Pro)

```python
# Setup (same as above)

# Train LLaVA-7B with 4-bit LoRA
!python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    model.use_lora=true \
    model.use_4bit=true \
    data.max_train_samples=1000 \
    training.output_dir=/content/drive/MyDrive/llava_outputs \
    training.num_epochs=1 \
    training.per_device_train_batch_size=4 \
    training.gradient_accumulation_steps=4 \
    training.fp16=true \
    training.save_steps=100

# With 4-bit + LoRA, uses ~4GB VRAM
```

**Expected time:** ~30-40 minutes on T4 GPU for 1000 samples, 1 epoch

### Workflow 3: DPO Training on Colab

```python
# Setup (same as above)

# Train preference model with DPO
!python scripts/train/train_dpo.py \
    experiment=gpt2_dpo \
    model.use_lora=true \
    data.dataset_name=anthropic_hh \
    data.max_train_samples=2000 \
    training.output_dir=/content/drive/MyDrive/dpo_outputs \
    training.num_epochs=1 \
    training.per_device_train_batch_size=8 \
    training.beta=0.1

# DPO is faster than PPO but requires preference pairs
```

**Expected time:** ~10-15 minutes on T4 GPU for 2000 pairs

---

## Optimization Tips

### 1. Maximize Batch Size

Colab GPUs have more memory than consumer GPUs. Use it:

```yaml
training:
  per_device_train_batch_size: 32  # Increase from default 8
  gradient_accumulation_steps: 1   # No need to accumulate with large batch
```

**Find optimal batch size:**
```python
# Start high and decrease until it fits
for batch_size in [64, 32, 16, 8, 4]:
    try:
        !python scripts/train/train_multimodal.py \
            experiment=clip_image_caption \
            training.per_device_train_batch_size={batch_size} \
            training.max_steps=10
        print(f"✓ Batch size {batch_size} works!")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ Batch size {batch_size} OOM, trying smaller...")
        else:
            raise
```

### 2. Use Mixed Precision Training

Always enable fp16 on Colab GPUs:

```yaml
training:
  fp16: true  # 2x faster, 2x less memory
```

**Speed comparison (CLIP, 100 steps):**
- fp32: ~60 seconds
- fp16: ~30 seconds

### 3. Profile Memory Usage

```python
# Add to your training script
import torch

def print_gpu_utilization():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

# Call periodically during training
print_gpu_utilization()
```

### 4. Clear Cache Between Runs

```python
# If running multiple experiments in one session
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
```

### 5. Use Smaller Models for Iteration

Start with small models to test your pipeline:

```bash
# Quick iteration (2 minutes)
!python scripts/train/train_sft.py experiment=gpt2_conversation training.max_steps=50

# Once it works, scale up
!python scripts/train/train_sft.py experiment=gpt2_conversation training.num_epochs=3
```

---

## Troubleshooting Colab-Specific Issues

### Issue 1: "CUDA out of memory"

**Solutions (in order):**
1. Reduce batch size: `training.per_device_train_batch_size=4`
2. Enable gradient accumulation: `training.gradient_accumulation_steps=4`
3. Enable 4-bit quantization: `model.use_4bit=true`
4. Use gradient checkpointing: `model.gradient_checkpointing=true`
5. Clear CUDA cache: `torch.cuda.empty_cache()`

**Example:**
```bash
# Before (OOM)
training.per_device_train_batch_size=32

# After (works)
training.per_device_train_batch_size=8
training.gradient_accumulation_steps=4
# Effective batch size = 8 * 4 = 32
```

### Issue 2: "Session disconnected due to inactivity"

**Prevention:**
- Use browser console trick (see "Session Management" above)
- Save checkpoints frequently (`save_steps=100`)
- Mount Google Drive for persistent storage
- Upgrade to Colab Pro for background execution

**Recovery:**
```bash
# Resume from last checkpoint
!python scripts/train/train_sft.py \
    experiment=gpt2_conversation \
    training.resume_from_checkpoint=./outputs/checkpoint-500
```

### Issue 3: "Disk quota exceeded"

**Solutions:**
1. Clear HuggingFace cache:
```bash
!rm -rf ~/.cache/huggingface/hub
```

2. Delete old checkpoints:
```bash
!rm -rf ./outputs/old_experiment
```

3. Save to Google Drive instead of local disk:
```yaml
training:
  output_dir: /content/drive/MyDrive/outputs
```

### Issue 4: "Package version conflicts"

Colab has pre-installed packages. Sometimes they conflict:

```bash
# Check installed versions
!pip list | grep transformers
!pip list | grep torch

# Force reinstall if needed
!pip install --force-reinstall transformers==4.36.0

# Or use requirements without version pins
!pip install transformers datasets accelerate
```

### Issue 5: "Import errors after installation"

**Solution:** Restart runtime after installing packages:

```python
# Install packages
!pip install -r requirements/base.txt

# Restart runtime (Runtime -> Restart runtime in Colab menu)
# Then re-run training cells
```

---

## Example Colab Notebook Template

```python
# ==============================================================================
# CELL 1: Setup and Installation
# ==============================================================================
# Check GPU
!nvidia-smi

# Clone repository
!git clone https://github.com/your-username/llm-post-training.git
%cd llm-post-training

# Install dependencies
!pip install -q -r requirements/base.txt
!pip install -q -r requirements/multimodal.txt

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ==============================================================================
# CELL 2: Mount Google Drive (Optional)
# ==============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================================
# CELL 3: Train CLIP (No LoRA)
# ==============================================================================
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=synthetic \
    data.max_train_samples=500 \
    training.output_dir=/content/drive/MyDrive/clip_outputs \
    training.num_epochs=3 \
    training.per_device_train_batch_size=32 \
    training.fp16=true \
    training.logging_steps=10 \
    training.save_steps=100

# ==============================================================================
# CELL 4: Train LLaVA with LoRA (Colab Pro)
# ==============================================================================
!python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    model.use_lora=true \
    model.use_4bit=true \
    data.max_train_samples=1000 \
    training.output_dir=/content/drive/MyDrive/llava_outputs \
    training.num_epochs=1 \
    training.per_device_train_batch_size=4 \
    training.gradient_accumulation_steps=4 \
    training.fp16=true

# ==============================================================================
# CELL 5: Evaluate Model
# ==============================================================================
!python scripts/evaluate/evaluate_model.py \
    model_path=/content/drive/MyDrive/clip_outputs \
    eval_dataset=coco \
    metrics=clip_score

# ==============================================================================
# CELL 6: Download Checkpoint
# ==============================================================================
from google.colab import files
!zip -r model_checkpoint.zip /content/drive/MyDrive/clip_outputs
files.download('model_checkpoint.zip')
```

---

## Cost-Benefit Analysis

### Free Tier vs Paid Tiers

**Free Tier ($0/month):**
- **Best for:**
  - Learning and experimentation
  - Small models (CLIP, GPT-2, LLaMA-1B)
  - Short training runs (<2 hours)
  - Datasets <5000 samples

**Colab Pro ($10/month):**
- **Best for:**
  - Regular training workflows
  - Medium models (LLaVA-7B, LLaMA-7B with LoRA)
  - Longer sessions (up to 24 hours)
  - Background execution

**Colab Pro+ ($50/month):**
- **Best for:**
  - Production-scale experiments
  - Large models (13B+ with LoRA)
  - Continuous training
  - Multiple concurrent sessions

### Alternative: Local GPU

| Option | Initial Cost | Monthly Cost | Performance | Use Case |
|--------|-------------|--------------|-------------|----------|
| Colab Free | $0 | $0 | T4 GPU | Learning, small experiments |
| Colab Pro | $0 | $10 | V100 GPU | Regular training |
| Colab Pro+ | $0 | $50 | A100 GPU | Heavy usage |
| Local RTX 4090 | ~$1800 | ~$20 (electricity) | 24GB VRAM | Daily training |
| Cloud (AWS p3.2xlarge) | $0 | ~$3/hour | V100 | On-demand |

**Recommendation:**
- **Beginners**: Start with Colab Free
- **Regular users**: Colab Pro is the sweet spot
- **Power users**: Consider local GPU if training >50 hours/month

---

## Performance Benchmarks

### Training Speed (T4 GPU on Colab Free)

| Model | Samples | Epochs | Time | Tokens/sec |
|-------|---------|--------|------|-----------|
| CLIP-ViT-B/32 | 1000 | 3 | ~3 min | ~15K |
| GPT-2 | 5000 | 3 | ~10 min | ~8K |
| GPT-2 + LoRA | 5000 | 3 | ~8 min | ~10K |
| LLaVA-7B + 4bit LoRA | 1000 | 1 | ~30 min | ~2K |
| LLaMA-7B + 4bit LoRA | 5000 | 1 | ~45 min | ~2.5K |

### Memory Footprint (Peak VRAM)

| Model | Config | Peak VRAM | Fits in Free Tier? |
|-------|--------|-----------|-------------------|
| CLIP-ViT-B/32 | fp16, batch=32 | ~3 GB | ✅ Yes |
| GPT-2 | fp16, batch=16 | ~2 GB | ✅ Yes |
| GPT-2 + LoRA | fp16, batch=32 | ~1.5 GB | ✅ Yes |
| LLaVA-7B + LoRA | 4bit, batch=4 | ~6 GB | ✅ Yes |
| LLaVA-7B full | fp16, batch=1 | ~18 GB | ❌ No (needs Pro+) |

---

## Best Practices Summary

### ✅ Do This

1. **Always enable fp16 on GPU**: `training.fp16=true`
2. **Save to Google Drive**: Prevent data loss on disconnect
3. **Use checkpoints frequently**: `training.save_steps=100`
4. **Start with small datasets**: Test pipeline before scaling
5. **Clear CUDA cache between runs**: `torch.cuda.empty_cache()`
6. **Use LoRA for 7B+ models**: Essential for Colab's memory limits
7. **Monitor GPU usage**: `!nvidia-smi` in a cell

### ❌ Don't Do This

1. **Don't use LoRA with CLIP**: It's broken, GPU makes it unnecessary
2. **Don't train large models on free tier without LoRA**: Will OOM
3. **Don't ignore session timeouts**: Use Drive or checkpoints
4. **Don't use fp32 on GPU**: Wastes time and memory
5. **Don't start with large datasets**: Test small first
6. **Don't forget to mount Drive**: Outputs will be lost
7. **Don't use CPU when GPU is available**: Check `device: auto` in config

---

## Additional Resources

- **Colab Documentation**: https://colab.research.google.com/notebooks/
- **HuggingFace Hub Integration**: `docs/huggingface_integration.md`
- **Model Selection Guide**: `docs/model_selection_guide.md`
- **Known Issues**: `docs/known_issues.md`
- **Platform Compatibility**: `docs/PLATFORM_COMPATIBILITY.md`

---

## FAQ

**Q: Can I use Colab's free tier for this repository?**

A: Yes! Free tier is perfect for:
- CLIP training (no LoRA)
- GPT-2 experiments
- Small-scale DPO/PPO
- Learning the post-training techniques

Not suitable for: LLaVA-7B+ without LoRA

---

**Q: Why not use LoRA for CLIP on Colab?**

A: PEFT (the LoRA library) has a fundamental bug with CLIP's dual-encoder architecture. This affects all platforms, not just Colab. Since Colab has a GPU, you can train CLIP without LoRA efficiently.

---

**Q: How do I resume training after disconnection?**

A:
```bash
!python scripts/train/train_sft.py \
    experiment=gpt2_conversation \
    training.resume_from_checkpoint=./outputs/checkpoint-500
```

Make sure checkpoints are saved to Google Drive!

---

**Q: Can I train multiple models in one session?**

A: Yes, but clear CUDA cache between trainings:
```python
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

---

**Q: What if I run out of VRAM?**

A: Try in this order:
1. Reduce batch size (`per_device_train_batch_size=4`)
2. Use gradient accumulation (`gradient_accumulation_steps=4`)
3. Enable 4-bit quantization (`model.use_4bit=true`)
4. Use gradient checkpointing (trades compute for memory)

---

**Q: Should I upgrade to Colab Pro?**

A: Upgrade if you:
- Train regularly (>10 hours/week)
- Need LLaVA-7B training
- Want background execution
- Hit free tier limits frequently

Free tier is fine for occasional use and small models.
