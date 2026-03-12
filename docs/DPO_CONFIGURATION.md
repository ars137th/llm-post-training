# DPO Configuration Guide

This document explains the configuration files for Direct Preference Optimization (DPO) training, including the Anthropic HH-RLHF dataset and all training parameters.

## Table of Contents
- [Overview](#overview)
- [Configuration File Structure](#configuration-file-structure)
- [The Anthropic HH-RLHF Dataset](#the-anthropic-hh-rlhf-dataset)
- [DPO-Specific Parameters](#dpo-specific-parameters)
- [Training Configuration](#training-configuration)
- [Experiment Configs](#experiment-configs)
- [Parameter Tuning Guide](#parameter-tuning-guide)
- [Common Issues and Solutions](#common-issues-and-solutions)

---

## Overview

DPO (Direct Preference Optimization) is a simpler alternative to traditional RLHF (PPO) that directly optimizes a language model from preference data without needing a separate reward model or complex reinforcement learning.

**Key advantages:**
- **Simpler**: 2 models (policy + reference) instead of 4 (policy + critic + reference + reward)
- **More stable**: Supervised learning instead of RL
- **Better results**: Often matches or exceeds PPO performance
- **Faster training**: No rollout generation or advantage computation

**Configuration files:**
```
configs/
├── config_dpo.yaml                          # Main config (entry point)
├── technique/dpo.yaml                        # DPO-specific hyperparameters
└── experiment/
    ├── dpo_gpt2_synthetic.yaml              # Quick test on synthetic data
    └── dpo_gpt2_hh_rlhf.yaml                # Production training on Anthropic dataset
```

---

## Configuration File Structure

### 1. Main Config: `configs/config_dpo.yaml`

This is the **entry point** for DPO training. It uses Hydra's defaults system to compose settings from multiple files.

```yaml
defaults:
  - model: gpt2              # Base model config
  - technique: dpo           # DPO-specific settings
  - data: preference         # Preference data settings
  - _self_                   # Override with settings below
```

**Key sections:**

#### Training Settings
```yaml
training:
  output_dir: "./outputs/dpo"
  num_epochs: 1                          # Usually 1 epoch is enough
  per_device_train_batch_size: 2         # Small due to (chosen, rejected) pairs
  gradient_accumulation_steps: 8         # Effective batch = 16
  learning_rate: 5e-7                    # MUCH lower than SFT (5e-5)
  weight_decay: 0.0                      # Often 0 for DPO
```

**Why these values?**
- **1 epoch**: DPO can overfit quickly to preferences
- **Batch size 2**: Each example has chosen + rejected responses (2x memory)
- **Learning rate 5e-7**: DPO is very sensitive to LR; too high → instability
- **Weight decay 0**: Preference learning doesn't need regularization like SFT

#### Data Settings
```yaml
data:
  use_synthetic: true                    # Quick test mode
  num_train_examples: 500                # For synthetic
  num_eval_examples: 100

  # For real datasets (set use_synthetic: false)
  # dataset_name: "Anthropic/hh-rlhf"
  # format: "anthropic"
```

#### Model Settings
```yaml
model:
  use_lora: true                         # HIGHLY RECOMMENDED
  lora_config:
    r: 8                                 # Rank
    lora_alpha: 16                       # Scaling
    target_modules: ["c_attn"]           # Which layers to adapt
```

**Why LoRA?**
- More stable training (fewer parameters to update)
- Less memory (50x reduction)
- Easier to merge/deploy
- Better generalization

---

## The Anthropic HH-RLHF Dataset

### What is HH-RLHF?

**Full name:** Anthropic's Human Preference Dataset for Harmless and Helpful AI

**Paper:** "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"
- Authors: Anthropic (Bai et al., 2022)
- Link: https://arxiv.org/abs/2204.05862

**Description:**
A large-scale dataset of human preferences for training AI assistants to be both helpful (answer questions well) and harmless (avoid harmful outputs).

**Dataset Statistics:**
- **Size**: ~161,000 preference pairs
  - Train: ~160,800 examples
  - Test: ~8,552 examples
- **Format**: Conversation pairs with human preference labels
- **Domain**: General assistant conversations
- **Languages**: English
- **License**: MIT

**HuggingFace Dataset:** `Anthropic/hh-rlhf`
**Download size**: ~170 MB compressed

### Dataset Structure

Each example contains a **conversation** with two responses (chosen and rejected):

```python
{
  "chosen": "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris...",
  "rejected": "Human: What is the capital of France?\n\nAssistant: I don't know."
}
```

**Conversation format:**
- Multi-turn dialogues between Human and Assistant
- Last turn is the Assistant's response being evaluated
- Both `chosen` and `rejected` have the same prompt context
- Human annotators preferred `chosen` over `rejected`

### Data Collection Process

1. **Generation**: Two different AI models generate responses to the same prompt
2. **Annotation**: Human annotators choose which response is better based on:
   - **Helpfulness**: Does it answer the question well?
   - **Harmlessness**: Does it avoid harmful/toxic content?
3. **Quality control**: Multiple annotators per example, agreement checks

### Using HH-RLHF in Your Config

```yaml
data:
  use_synthetic: false                   # Use real data
  dataset_name: "Anthropic/hh-rlhf"
  dataset_config: null
  train_split: "train"
  eval_split: "test"
  num_train_examples: 10000              # Subset for faster training
  num_eval_examples: 1000
  format: "anthropic"                    # Parse Anthropic conversation format
```

**Format parser:** Our `parse_anthropic_format()` function extracts:
- **Prompt**: Everything up to the last "Assistant:" marker
- **Chosen response**: Assistant's preferred response
- **Rejected response**: Assistant's rejected response

---

## DPO-Specific Parameters

### Technique Config: `configs/technique/dpo.yaml`

This file contains DPO-specific hyperparameters.

#### Beta (β) - Most Important Parameter

```yaml
beta: 0.1  # Temperature parameter controlling KL divergence
```

**What it does:**
- Controls how far the policy can deviate from the reference model
- Appears in the DPO loss: `loss = -log σ(β * (log π/π_ref)[chosen] - β * (log π/π_ref)[rejected])`

**Effect:**
- **Higher beta (0.3-0.5)**: Stay closer to reference (more conservative)
  - Pros: More stable, safer, less likely to overfit
  - Cons: Slower learning, less adaptation to preferences
- **Lower beta (0.05-0.1)**: More aggressive optimization
  - Pros: Faster learning, stronger preference alignment
  - Cons: Risk of overfitting, instability

**Recommended values:**
- **Start with**: 0.1 (default)
- **Conservative**: 0.2-0.3
- **Aggressive**: 0.05-0.1
- **Very aggressive**: < 0.05 (not recommended without careful tuning)

#### Loss Type

```yaml
loss_type: "dpo"  # or "ipo"
```

**Options:**

1. **DPO** (Standard):
   - Loss: `-log σ(β * logits)`
   - Uses log-sigmoid (numerically stable)
   - Most common, well-tested

2. **IPO** (Identity Preference Optimization):
   - Loss: `(β * logits - 0.5)²`
   - Squared loss instead of log-sigmoid
   - More robust to outliers and noisy preferences
   - Reference: https://arxiv.org/abs/2310.12036

**When to use IPO:**
- Noisy preference data (annotator disagreement)
- Want more robustness
- Overconfident model predictions

---

## Training Configuration

### Learning Rate

```yaml
learning_rate: 5e-7  # Much lower than SFT or reward modeling
```

**Comparison:**
- **SFT**: 5e-5 (100x higher)
- **Reward Model**: 1e-5 (20x higher)
- **DPO**: 5e-7 (most conservative)

**Why so low?**
- DPO is very sensitive to learning rate
- Too high → training instability, divergence
- Too low → no learning, stays at reference

**Tuning guide:**
- Start with: 5e-7
- If loss not decreasing: Try 1e-6
- If training unstable: Try 1e-7
- Monitor: accuracy, reward margins, KL divergence

### Number of Epochs

```yaml
num_epochs: 1  # Typically sufficient
```

**Why only 1 epoch?**
- DPO learns preferences quickly
- More epochs → overfitting to specific preference pairs
- Unlike SFT which benefits from multiple epochs

**Signs you need more epochs:**
- Accuracy still increasing at end of epoch 1
- Reward margins still growing
- Validation loss still decreasing

**Signs of overfitting:**
- Training accuracy high (>80%) but validation plateaus
- Large gap between train and validation metrics
- KL divergence increasing rapidly

### Batch Size

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
# Effective batch size = 2 × 8 = 16
```

**Why small batch size?**
- Each example contains **two sequences** (chosen + rejected)
- Memory usage: 2x a normal SFT example
- DPO benefits from larger effective batch sizes (16-32)

**How to increase effective batch size:**
1. Increase `gradient_accumulation_steps` (no extra memory)
2. Increase `per_device_train_batch_size` (needs more memory)
3. Use multiple GPUs with `per_device_train_batch_size * num_gpus`

**Recommended:**
- Single GPU (24GB): batch=2, accum=8 → effective=16
- Multiple GPUs (4x): batch=4, accum=2 → effective=32
- If OOM: batch=1, accum=16 → effective=16

### Other Training Settings

```yaml
weight_decay: 0.0        # Often 0 for DPO (unlike SFT which uses 0.01)
warmup_steps: 100        # Small warmup
max_grad_norm: 1.0       # Gradient clipping (important for stability)
```

---

## Experiment Configs

### Quick Test: `dpo_gpt2_synthetic.yaml`

**Purpose:** Validate DPO pipeline quickly without downloading datasets

```yaml
data:
  use_synthetic: true
  num_train_examples: 500
  num_eval_examples: 100

training:
  num_epochs: 1
  learning_rate: 5e-7

model:
  name: "gpt2"
  use_lora: true
```

**Run command:**
```bash
python scripts/train/train_dpo.py experiment=dpo_gpt2_synthetic device=cpu
```

**Expected time:** 20-30 minutes on CPU, 5-10 minutes on GPU

**Expected results:**
- Accuracy: 60-75% (synthetic data is easy)
- Loss should decrease steadily
- Reward margins should increase

### Production: `dpo_gpt2_hh_rlhf.yaml`

**Purpose:** Train on real Anthropic preference data

```yaml
data:
  use_synthetic: false
  dataset_name: "Anthropic/hh-rlhf"
  format: "anthropic"
  num_train_examples: 10000      # Subset of full 160K
  num_eval_examples: 1000

training:
  num_epochs: 1
  learning_rate: 5e-7
  fp16: true                      # Enable for GPU

model:
  name: "gpt2"
  use_lora: true
```

**Run command:**
```bash
# GPU training (recommended)
python scripts/train/train_dpo.py experiment=dpo_gpt2_hh_rlhf device=cuda training.fp16=true

# CPU training (slow but works)
python scripts/train/train_dpo.py experiment=dpo_gpt2_hh_rlhf device=cpu
```

**Expected time:** 
- GPU (RTX 3090): 1-2 hours for 10K examples
- CPU (M1 Max): 6-8 hours for 10K examples

**Expected results:**
- Accuracy: 55-65% (real data is harder)
- Reward margin: positive and growing
- Should match or exceed reward model performance

---

## Parameter Tuning Guide

### Step-by-Step Tuning Process

#### 1. Start with Defaults

```bash
python scripts/train/train_dpo.py experiment=dpo_gpt2_synthetic device=cpu
```

**Check:**
- Does training run without errors?
- Is accuracy increasing?
- Are reward margins positive?

#### 2. Tune Beta

**Test different beta values:**

```bash
# Conservative (safe)
python scripts/train/train_dpo.py experiment=dpo_gpt2_synthetic technique.beta=0.3

# Default
python scripts/train/train_dpo.py experiment=dpo_gpt2_synthetic technique.beta=0.1

# Aggressive
python scripts/train/train_dpo.py experiment=dpo_gpt2_synthetic technique.beta=0.05
```

**Compare:**
- Final accuracy
- Training stability (loss curve smoothness)
- KL divergence (should stay reasonable, < 5.0)

#### 3. Tune Learning Rate

**If training is unstable:**
```bash
python scripts/train/train_dpo.py training.learning_rate=1e-7
```

**If not learning enough:**
```bash
python scripts/train/train_dpo.py training.learning_rate=1e-6
```

#### 4. Try IPO Loss

**If data is noisy or model is overconfident:**
```bash
python scripts/train/train_dpo.py technique.loss_type=ipo
```

### Hyperparameter Sensitivity

**Most sensitive (tune first):**
1. **Learning rate**: High impact on stability and learning
2. **Beta**: Controls strength of preference learning
3. **Effective batch size**: Affects gradient quality

**Less sensitive:**
- Warmup steps
- Weight decay (usually keep at 0)
- Max grad norm

**Usually don't change:**
- Number of epochs (almost always 1)
- Optimizer (AdamW works best)

---

## Common Issues and Solutions

### Issue 1: Training is Unstable (Loss Spikes)

**Symptoms:**
- Loss suddenly increases
- NaN or inf values
- Gradient explosions

**Solutions:**
1. **Lower learning rate**: Try 1e-7 instead of 5e-7
2. **Increase beta**: Try 0.2 or 0.3 (more conservative)
3. **Check max_grad_norm**: Ensure it's set to 1.0
4. **Use fp16/bf16**: Can help with numerical stability

```bash
python scripts/train/train_dpo.py \
  training.learning_rate=1e-7 \
  technique.beta=0.2 \
  training.fp16=true
```

### Issue 2: Accuracy Not Improving (Stuck at ~50%)

**Symptoms:**
- Accuracy hovers around 50% (random guessing)
- Reward margins near zero
- Model not learning preferences

**Solutions:**
1. **Check data quality**: Are preferences clear?
2. **Increase learning rate**: Try 1e-6
3. **Lower beta**: Try 0.05 (more aggressive)
4. **Train longer**: Try 2 epochs
5. **Check reference model**: Should be a decent SFT model

```bash
python scripts/train/train_dpo.py \
  training.learning_rate=1e-6 \
  technique.beta=0.05 \
  training.num_epochs=2
```

### Issue 3: Overfitting (Train Good, Validation Bad)

**Symptoms:**
- Training accuracy: 80%+
- Validation accuracy: <60%
- Large train/val gap

**Solutions:**
1. **Use LoRA**: Reduces overfitting
2. **Increase beta**: 0.2-0.3 (stay closer to reference)
3. **More data**: Increase `num_train_examples`
4. **Early stopping**: Monitor validation and stop when it plateaus

```bash
python scripts/train/train_dpo.py \
  model.use_lora=true \
  technique.beta=0.2 \
  checkpoint.load_best_model_at_end=true
```

### Issue 4: Out of Memory (OOM)

**Symptoms:**
- CUDA out of memory error
- Process killed
- Slow swapping

**Solutions:**
1. **Reduce batch size**: Set `per_device_train_batch_size=1`
2. **Increase gradient accumulation**: Keep effective batch size same
3. **Use LoRA**: Reduces memory by ~50%
4. **Use 8-bit quantization**: `model.use_8bit=true`
5. **Reduce max_length**: `tokenizer.max_length=256`

```bash
python scripts/train/train_dpo.py \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=16 \
  model.use_lora=true \
  model.use_8bit=true
```

### Issue 5: MPS (Apple Silicon) Errors

**Symptoms:**
- `NotImplementedError` on MPS device
- Operations not supported

**Solution:** Use CPU instead
```bash
python scripts/train/train_dpo.py device=cpu
```

**Why:** Some PyTorch operations used in DPO (log_sigmoid) not yet implemented on MPS. CPU training works perfectly, just slower.

---

## Monitoring Training

### Key Metrics to Watch

**During training:**

1. **Loss**: Should decrease steadily
   - Good: Smooth decrease
   - Bad: Spikes, instability, NaN

2. **Accuracy**: % of pairs where R(chosen) > R(rejected)
   - Random: ~50%
   - Good: 60-70%
   - Excellent: >70%

3. **Reward Margin**: R(chosen) - R(rejected)
   - Should be positive and increasing
   - Typical: 0.1-1.0
   - If negative: model prefers rejected (bad!)

4. **KL Divergence**: How far from reference
   - Should stay bounded (< 5.0)
   - If too high: increase beta or lower LR

### Logging Examples

**TensorBoard:**
```bash
tensorboard --logdir outputs/dpo/logs
```

**Weights & Biases:**
```yaml
logging:
  use_wandb: true
  wandb_project: "llm-post-training"
  wandb_run_name: "dpo_gpt2_experiment"
```

---

## Advanced Topics

### Using a Custom SFT Model as Reference

```python
from src.models.language import LanguageModel

# Load your SFT model as reference
reference_model = LanguageModel.from_pretrained(
    "./outputs/sft_gpt2/final_model",
    use_lora=False,
)
```

### Combining DPO with Other Techniques

**Pipeline:**
1. **SFT**: Train on demonstrations → SFT model
2. **DPO**: Train SFT model on preferences → Aligned model
3. **Evaluation**: Test on benchmarks

### Scaling to Larger Models

**For models > 1B parameters:**

```yaml
model:
  use_lora: true               # Essential
  use_8bit: true               # Or use_4bit for even larger models
  lora_config:
    r: 16                      # Higher rank for more capacity
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]  # More modules

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32  # Large effective batch
  fp16: true                   # Or bf16 for A100/H100
```

---

## Summary: Quick Reference

### Commands

```bash
# Quick test (synthetic data)
python scripts/train/train_dpo.py experiment=dpo_gpt2_synthetic device=cpu

# Real training (Anthropic HH-RLHF)
python scripts/train/train_dpo.py experiment=dpo_gpt2_hh_rlhf device=cuda training.fp16=true

# Override any parameter
python scripts/train/train_dpo.py technique.beta=0.2 training.learning_rate=1e-6
```

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **beta** | 0.1 | 0.05-0.5 | KL penalty strength |
| **learning_rate** | 5e-7 | 1e-7 to 1e-6 | Learning speed |
| **num_epochs** | 1 | 1-2 | Training iterations |
| **batch_size** | 2 | 1-4 | Memory vs speed |
| **gradient_accum** | 8 | 4-16 | Effective batch size |

### Expected Results

| Dataset | Accuracy | Training Time (GPU) |
|---------|----------|---------------------|
| Synthetic (500) | 60-75% | 5-10 min |
| HH-RLHF (10K) | 55-65% | 1-2 hours |
| HH-RLHF (full) | 60-70% | 10-15 hours |

---

## References

**Papers:**
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [IPO: Identity Preference Optimization](https://arxiv.org/abs/2310.12036)
- [Anthropic HH-RLHF Dataset](https://arxiv.org/abs/2204.05862)

**Code:**
- This implementation: `src/core/dpo/`
- HuggingFace TRL: https://github.com/huggingface/trl

**Datasets:**
- Anthropic HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Stanford SHP: https://huggingface.co/datasets/stanfordnlp/SHP
- OpenAssistant: https://huggingface.co/datasets/OpenAssistant/oasst1

**Related Docs:**
- `docs/DPO_THEORY.md`: Mathematical explanation of DPO
- `docs/techniques/dpo.md`: Technical deep dive
- `notebooks/04_dpo_simplified_rlhf.ipynb`: Interactive tutorial
