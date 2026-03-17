# Running PPO Training on Google Colab

This guide shows how to run full-scale PPO training (`ppo_gpt2_full`) on Google Colab with free GPU access.

## Prerequisites

Before starting, you need:
1. **Trained Reward Model** - From Phase 3 (reward modeling)
2. **Prompts Dataset** - Text file with prompts (one per line)

If you don't have these, see the "Quick Start (Without Prerequisites)" section below.

---

## Step-by-Step Guide

### 1. Setup Colab Notebook

Create a new Colab notebook and enable GPU:
- Go to `Runtime` → `Change runtime type`
- Select `T4 GPU` (free tier)
- Click `Save`

### 2. Clone Repository and Install Dependencies

```python
# Clone the repository
!git clone https://github.com/ars137th/llm-post-training.git
%cd llm-post-training

# Install dependencies (use Colab-specific requirements with relaxed version constraints)
!pip install -q -r requirements/colab.txt
!pip install -q -r requirements/colab-rlhf.txt

# Verify installation
!python -c "from src.core.ppo import *; print('✅ PPO imports work')"
```

### 3. Upload Reward Model (Option A: From Local)

If you have a trained reward model locally:

```python
# Create directory
!mkdir -p outputs/reward_model_hh_rlhf/final_model

# Upload files using Colab's file upload
from google.colab import files
import os

print("Upload your reward model files (model weights, config.json, etc.)")
print("Upload to: outputs/reward_model_hh_rlhf/final_model/")

uploaded = files.upload()

# Move uploaded files to correct location
for filename, content in uploaded.items():
    with open(f'outputs/reward_model_hh_rlhf/final_model/{filename}', 'wb') as f:
        f.write(content)
```

### 3. Upload Reward Model (Option B: From Google Drive)

If your reward model is on Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive to Colab
!cp -r "/content/drive/MyDrive/reward_model_hh_rlhf" outputs/
```

### 3. Upload Reward Model (Option C: Use HuggingFace Hub)

If you uploaded your reward model to HuggingFace:

```python
# Modify the config to load from HF Hub
# We'll override checkpoint_path at runtime
HF_REWARD_MODEL = "your-username/your-reward-model"  # Change this
```

### 4. Prepare Prompts Dataset

**Option A: Upload prompts file**

```python
# Create datasets directory
!mkdir -p datasets

# Upload prompts file
from google.colab import files
print("Upload your prompts file (one prompt per line)")
uploaded = files.upload()

# Move to correct location
for filename, content in uploaded.items():
    with open('datasets/hh_rlhf_prompts.txt', 'wb') as f:
        f.write(content)

# Verify
!head -n 5 datasets/hh_rlhf_prompts.txt
!wc -l datasets/hh_rlhf_prompts.txt
```

**Option B: Extract from HuggingFace dataset**

```python
from datasets import load_dataset

# Load Anthropic HH-RLHF dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# Extract prompts (first message from each conversation)
prompts = []
for example in dataset:
    # Parse conversation and extract first human message
    text = example['chosen']  # or 'rejected', both have same prompt
    # Simple extraction: take text before first "Assistant:"
    if "Assistant:" in text:
        prompt = text.split("Assistant:")[0].strip()
        prompts.append(prompt)

# Save to file
!mkdir -p datasets
with open('datasets/hh_rlhf_prompts.txt', 'w') as f:
    for prompt in prompts[:5000]:  # Use first 5000
        f.write(prompt + '\n')

print(f"✅ Extracted {len(prompts[:5000])} prompts")
!head -n 3 datasets/hh_rlhf_prompts.txt
```

### 5. Configure Training

Create a custom config for Colab (reduced settings to fit in Colab's resources):

```python
%%writefile configs/experiment/ppo_gpt2_colab.yaml
# @package _global_

# PPO Training on Colab (GPU optimized)

defaults:
  - override /model: gpt2
  - override /technique: ppo
  - override /data: prompts
  - _self_

# Model settings
model:
  name: "gpt2"
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: ["c_attn", "c_proj"]
    bias: "none"
    task_type: "CAUSAL_LM"

# Data settings
data:
  use_synthetic: false
  prompt_file: "./datasets/hh_rlhf_prompts.txt"
  num_prompts: 1000  # Reduced for Colab
  max_prompt_length: 256  # Reduced
  max_response_length: 128  # Reduced

# Training settings (Colab optimized)
training:
  output_dir: "./outputs/ppo_gpt2_colab"

  # Rollout (reduced for Colab)
  num_rollouts: 50  # Reduced from 100
  batch_size: 16  # Reduced from 32

  # PPO
  ppo_epochs: 4
  mini_batch_size: 4  # Reduced from 8
  clip_range: 0.2

  # GAE
  gamma: 0.99
  lam: 0.95

  # Loss coefficients
  vf_coef: 0.5
  ent_coef: 0.01
  kl_coef: 0.05

  # Optimization
  learning_rate: 1.0e-6
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Generation
  temperature: 1.0
  top_p: 0.9
  top_k: 50

  # Adaptive KL
  use_adaptive_kl: true
  target_kl: 0.01

  # Normalization
  normalize_advantages: true
  reward_clip: 10.0

  # Logging
  log_every: 1
  save_every: 10

  # Mixed precision (helps with memory)
  fp16: true

# Reward model settings
reward_model:
  from_checkpoint: true
  checkpoint_path: "./outputs/reward_model_hh_rlhf/final_model"

# Logging
logging:
  use_wandb: false  # Set to true if you have W&B account
  wandb_project: "llm-post-training"
  wandb_run_name: "ppo_gpt2_colab"
  use_tensorboard: true

# System
device: "cuda"
seed: 42
```

### 6. Run Training

```python
# Run PPO training with Colab config
!python scripts/train/train_ppo.py experiment=ppo_gpt2_colab

# This will take approximately 3-5 hours on Colab T4 GPU
```

**Monitor Progress:**

```python
# In a separate cell, monitor training logs (run while training)
!tail -f outputs/ppo_gpt2_colab/training.log
```

### 7. Monitor with TensorBoard (Optional)

```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir outputs/ppo_gpt2_colab/tensorboard
```

### 8. Download Results

After training completes:

```python
# Zip the output directory
!zip -r ppo_gpt2_colab.zip outputs/ppo_gpt2_colab/

# Download
from google.colab import files
files.download('ppo_gpt2_colab.zip')
```

---

## Quick Start (Without Prerequisites)

If you don't have a reward model or prompts, run a minimal test:

```python
# Clone repo
!git clone https://github.com/ars137th/llm-post-training.git
%cd llm-post-training

# Install (use Colab-specific requirements)
!pip install -q -r requirements/colab.txt
!pip install -q -r requirements/colab-rlhf.txt

# Run minimal test with synthetic data (no reward model needed)
!python scripts/train/train_ppo.py \
    model=gpt2 \
    data.use_synthetic=true \
    data.num_prompts=20 \
    training.num_rollouts=5 \
    training.batch_size=4 \
    device=cuda

# This will complete in ~10 minutes
```

---

## Resource Management

### Check GPU Memory

```python
# Monitor GPU usage
!nvidia-smi

# Check memory in Python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### If You Run Out of Memory

Reduce these settings in your config:

```yaml
training:
  batch_size: 8  # Reduce from 16
  mini_batch_size: 2  # Reduce from 4

data:
  max_prompt_length: 128  # Reduce from 256
  max_response_length: 64  # Reduce from 128
```

Or use gradient checkpointing:

```python
# Add to training script before creating models
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

---

## Expected Results

For `ppo_gpt2_colab` config:

- **Training Time**: 3-5 hours on T4 GPU
- **GPU Memory**: ~8-10 GB (fits in Colab free tier)
- **Final Reward**: Should improve significantly (0.0 → 0.5+)
- **KL Divergence**: Should stay < 0.01
- **Output**: Trained actor and critic models

---

## Troubleshooting

### Issue: Colab Disconnects

**Solution**: Colab may disconnect after ~12 hours. For long training:

```python
# Add to beginning of notebook to prevent disconnection
import IPython
js_code = '''
function ClickConnect(){
    console.log("Clicked");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
'''
display(IPython.display.Javascript(js_code))
```

### Issue: Out of Memory

**Solutions**:
1. Reduce batch_size: `training.batch_size=8`
2. Reduce sequence length: `data.max_response_length=64`
3. Use gradient accumulation (modify trainer)
4. Use 8-bit models: `model.use_8bit=true`

### Issue: Reward Model Not Found

**Solution**: Either:
1. Train reward model first (see Phase 3)
2. Use synthetic testing: `data.use_synthetic=true reward_model.from_checkpoint=false`
3. Use a pre-trained model from HuggingFace Hub

### Issue: Slow Training

**Solutions**:
1. Verify GPU is enabled: `Runtime → Change runtime type → GPU`
2. Reduce num_rollouts: `training.num_rollouts=20`
3. Use mixed precision: `training.fp16=true`

---

## Using Colab Pro

If you have Colab Pro:

- Use **A100 GPU** for 3-5x faster training
- Increase batch sizes: `batch_size=32, mini_batch_size=8`
- Train larger models: Use `gpt2-medium` or `gpt2-large`
- No timeout issues (up to 24 hours)

---

## Saving to Google Drive

To avoid losing results if Colab disconnects:

```python
# Mount Drive at the beginning
from google.colab import drive
drive.mount('/content/drive')

# Modify output_dir in config
training:
  output_dir: "/content/drive/MyDrive/ppo_outputs/ppo_gpt2_colab"
```

Or periodically sync during training:

```python
# Run in background while training
!rsync -av --progress outputs/ppo_gpt2_colab/ \
  "/content/drive/MyDrive/ppo_outputs/ppo_gpt2_colab/"
```

---

## Complete Colab Notebook Template

Here's a complete notebook you can copy-paste:

```python
# === CELL 1: Setup ===
# Enable GPU: Runtime → Change runtime type → T4 GPU

# Clone and install
!git clone https://github.com/ars137th/llm-post-training.git
%cd llm-post-training
!pip install -q -r requirements/colab.txt
!pip install -q -r requirements/colab-rlhf.txt

# === CELL 2: Mount Drive (optional) ===
from google.colab import drive
drive.mount('/content/drive')

# === CELL 3: Prepare Data ===
# Option A: Use synthetic (easiest)
USE_SYNTHETIC = True

# Option B: Extract from HuggingFace
if not USE_SYNTHETIC:
    from datasets import load_dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    prompts = []
    for example in dataset[:5000]:
        text = example['chosen']
        if "Assistant:" in text:
            prompt = text.split("Assistant:")[0].strip()
            prompts.append(prompt)

    !mkdir -p datasets
    with open('datasets/hh_rlhf_prompts.txt', 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')

# === CELL 4: Run Training ===
if USE_SYNTHETIC:
    # Quick test with synthetic data
    !python scripts/train/train_ppo.py \
        model=gpt2 \
        data.use_synthetic=true \
        data.num_prompts=50 \
        training.num_rollouts=10 \
        training.batch_size=8 \
        device=cuda
else:
    # Full training with real data
    !python scripts/train/train_ppo.py \
        experiment=ppo_gpt2_colab

# === CELL 5: Download Results ===
!zip -r results.zip outputs/
from google.colab import files
files.download('results.zip')
```

---

## Next Steps After Training

1. **Evaluate the model** - Compare with SFT baseline
2. **Test generations** - Try the trained actor on new prompts
3. **Compare with DPO** - See Phase 4 results
4. **Analyze metrics** - Check reward curves, KL divergence
5. **Fine-tune hyperparameters** - Adjust based on results

See `notebooks/05_ppo_rlhf_deep_dive.ipynb` for detailed analysis.
