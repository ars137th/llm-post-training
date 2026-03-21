# ⚠️ macOS Training Limitation

## TL;DR

**HuggingFace Trainer cannot reliably train models on macOS** due to fork safety issues.

✅ **What works on macOS:**
- Model loading and inference
- Data processing
- Evaluation metrics
- Code development and testing

❌ **What doesn't work on macOS:**
- Training with HuggingFace Trainer (bus error)

## The Problem

Even with all workarounds applied:
- ✅ `dataloader_num_workers=0`
- ✅ `dataloader_pin_memory=False`
- ✅ `eval_strategy="epoch"` (no periodic eval)
- ✅ `save_strategy="epoch"`

The training still fails with:
```
zsh: bus error  python scripts/train/train_multimodal.py
resource_tracker: There appear to be 1 leaked semaphore objects
```

**Root cause:** HuggingFace Trainer uses multiprocessing internally in the training loop itself (not just DataLoader), which violates macOS's strict fork safety rules.

## Solutions

### Solution 1: Use Google Colab (Recommended)

Free GPU training that's 10-20x faster than macOS CPU:

```python
# In Google Colab
!git clone https://github.com/your-repo/llm-post-training.git
%cd llm-post-training
!pip install -r requirements/base.txt -r requirements/multimodal.txt

# Train CLIP (no LoRA needed on GPU)
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.num_epochs=3 \
    training.fp16=true
```

**See:** `docs/google_colab_guide.md` for complete setup

### Solution 2: Use Docker with Linux

Run training in a Linux container:

```bash
# Pull official PyTorch image
docker run -it --rm \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
    bash

# Inside container (Linux environment)
cd /workspace
pip install -r requirements/base.txt
python scripts/train/train_multimodal.py experiment=clip_image_caption
```

### Solution 3: Remote Linux Server

SSH into a Linux machine and run training there:

```bash
ssh your-server
git clone <repo>
python scripts/train/train_multimodal.py experiment=clip_image_caption
```

### Solution 4: GitHub Actions / CI

Use GitHub Actions with Linux runners for automated training.

## What macOS IS Good For

macOS is excellent for:
1. **Development** - Writing code, debugging logic
2. **Testing** - Unit tests, integration tests (non-training)
3. **Inference** - Running trained models
4. **Data exploration** - Processing datasets, visualization
5. **Prototyping** - Quick experiments without training

## Comparison

| Task | macOS | Google Colab | Linux Server |
|------|-------|--------------|--------------|
| **Development** | ✅ Best | ❌ Limited | ✅ Good |
| **Testing** | ✅ Good | ❌ Not ideal | ✅ Best |
| **Training** | ❌ Fails | ✅ Fast (GPU) | ✅ Best |
| **Inference** | ✅ Works | ✅ Works | ✅ Works |
| **Cost** | Free (own HW) | Free/Paid | Paid |
| **Speed** | Slow (CPU) | Fast (GPU) | Fast (GPU) |

## Workaround We Tried

Our `scripts/train/train_multimodal.py` attempts to work around this by:

```python
from src.utils.compat import is_macos, apply_macos_training_workarounds

if is_macos():
    logger.warning("⚠️  macOS detected - applying fork safety workarounds")
    training_args = apply_macos_training_workarounds(training_args)
    # Sets dataloader_num_workers=0, disables eval, etc.
```

**Unfortunately, this is not sufficient.** The Trainer's internal training loop still causes bus errors.

## Why This Happens

**Technical details:**

1. macOS uses a `fork()` system call to create child processes
2. macOS has strict "fork safety" rules - certain operations aren't safe after `fork()`
3. PyTorch and NumPy operations often violate these rules
4. HuggingFace Trainer uses multiprocessing internally for:
   - Progress bars
   - Logging
   - Checkpoint management
   - Internal state tracking
5. Even with `num_workers=0`, the Trainer still forks for these operations

**From Apple's documentation:**
> "On Darwin, the use of fork() without an immediate exec() is not safe in multi-threaded programs."

PyTorch and transformers are multi-threaded by nature, so this is unavoidable.

## Alternatives for macOS Users

### If you must train on macOS:

**Option: Simple Training Loop (No Trainer)**

Create a minimal training script that doesn't use HuggingFace Trainer:

```python
# examples/train_clip_simple_macos.py
import torch
from torch.utils.data import DataLoader

# Load model
model = ...

# Create DataLoader (num_workers=0 for macOS)
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(**batch)
        loss = compute_loss(outputs, batch)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
```

This avoids Trainer's multiprocessing entirely.

**Status:** We don't currently provide this in the repo, but it's possible to implement.

## Recommendation

**For this repository:**
- Use macOS for development and testing
- Use Google Colab (free tier) for actual training
- Use Linux/cloud for production training

**Training workflow:**
1. Develop code on macOS
2. Test (non-training) on macOS
3. Train on Colab using `docs/google_colab_guide.md`
4. Download trained model back to macOS for inference

## Related Documentation

- `docs/google_colab_guide.md` - Complete Colab training guide
- `docs/known_issues.md` - Bus error technical details
- `docs/PLATFORM_COMPATIBILITY.md` - Platform comparison
- `src/utils/compat.py` - Platform detection and workarounds

## Status

This is a **known limitation** that cannot be fixed without:
1. HuggingFace removing all multiprocessing from Trainer (unlikely)
2. Apple relaxing fork safety rules (unlikely)
3. PyTorch becoming fork-safe on macOS (unlikely)

**Our recommendation:** Accept this limitation and use Colab/Linux for training.
