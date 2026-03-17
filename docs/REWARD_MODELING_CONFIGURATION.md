# Reward Modeling Configuration Guide

This document explains the configuration files for reward model training, including the base config structure, experiment overrides, and the Anthropic HH-RLHF dataset.

## Table of Contents
- [Overview](#overview)
- [Configuration File Structure](#configuration-file-structure)
- [Base Config: config_reward.yaml](#base-config-config_rewardyaml)
- [The Anthropic HH-RLHF Dataset](#the-anthropic-hh-rlhf-dataset)
- [Experiment Configs](#experiment-configs)
- [Parameter Tuning Guide](#parameter-tuning-guide)
- [Common Issues and Solutions](#common-issues-and-solutions)

---

## Overview

Reward modeling is the second phase of RLHF (Reinforcement Learning from Human Feedback). The reward model learns to score responses based on human preferences.

**Key concept:** The reward model predicts which of two responses humans prefer, trained using the Bradley-Terry ranking loss.

**Configuration files:**
```
configs/
├── config_reward.yaml                       # Base config (starting point)
├── technique/reward_model.yaml              # Reward-specific hyperparameters
└── experiment/
    ├── reward_gpt2_synthetic.yaml           # Quick test on synthetic data
    └── reward_gpt2_hh_rlhf.yaml            # Production training on Anthropic dataset
```

---

## Configuration File Structure

### The Hierarchy

Hydra composes configurations in this order:

```
1. config_reward.yaml         (Base config)
   ├─ loads model/gpt2.yaml
   ├─ loads technique/reward_model.yaml
   └─ loads data/preference.yaml

2. experiment/reward_gpt2_hh_rlhf.yaml  (Overrides base)
   └─ Overrides data.use_synthetic, training params, etc.
```

**Key insight:** `config_reward.yaml` is the **base/default** configuration. Experiment configs **override** specific settings from the base.

### The `_self_` Directive

The `_self_` directive in Hydra controls **when** a config file's own values are applied:

```yaml
# In experiment config
defaults:
  - override /model: gpt2
  - override /technique: reward_model
  - override /data: preference
  - _self_  # ← This is CRITICAL!
```

**Without `_self_`:**
- Order: model → technique → data → **experiment** → base
- Result: Base config overrides experiment config ❌

**With `_self_`:**
- Order: model → technique → data → base → **experiment**
- Result: Experiment config overrides base config ✅

**Real example of the problem:**
```yaml
# experiment/reward_gpt2_hh_rlhf.yaml (without _self_)
data:
  use_synthetic: false  # I want real data!

# But config_reward.yaml has:
data:
  use_synthetic: true  # Base default

# Without _self_, you get: use_synthetic: true (base wins)
# With _self_, you get: use_synthetic: false (experiment wins)
```

---

## Base Config: config_reward.yaml

### Purpose

`config_reward.yaml` serves as the **default starting point** for all reward model training:

- ✅ Provides sensible defaults that work out-of-the-box
- ✅ Can be used directly for quick tests
- ✅ Gets overridden by experiment configs for specific use cases

**Think of it as:** A template that experiment configs customize.

### File Structure

```yaml
# configs/config_reward.yaml

defaults:
  - model: gpt2                    # Load model config
  - technique: reward_model        # Load reward-specific settings
  - data: preference               # Load preference data config
  - optional experiment: null      # Allow experiment overrides
  - _self_                         # Apply this file's settings

# Training settings (optimized for reward modeling)
training:
  output_dir: "./outputs/reward_model"
  num_epochs: 1              # Usually 1 epoch is sufficient
  per_device_train_batch_size: 2  # Small (processes pairs)
  gradient_accumulation_steps: 4  # Effective batch = 8
  learning_rate: 1e-5        # Lower than SFT
  # ... more settings

# Data settings (default to synthetic for quick tests)
data:
  use_synthetic: true        # ← Overridden by experiments
  num_train_examples: 500
  num_eval_examples: 100

  # For real datasets (experiments enable this)
  # dataset_name: "Anthropic/hh-rlhf"
  # format: "anthropic"
```

### Key Sections

#### Training Settings

```yaml
training:
  num_epochs: 1                       # 1 epoch usually sufficient
  per_device_train_batch_size: 2      # Small: each example = (chosen, rejected) pair
  gradient_accumulation_steps: 4      # Effective batch = 2 × 4 = 8
  learning_rate: 1e-5                 # Lower than SFT (5e-5)
  eval_steps: 250                     # Frequent evaluation
```

**Why these values?**
- **1 epoch:** Reward models can overfit quickly to preferences
- **Batch size 2:** Each example has 2 forward passes (chosen + rejected)
- **Learning rate 1e-5:** More conservative than SFT to prevent overfitting

#### Data Settings

```yaml
data:
  use_synthetic: true       # Default: synthetic for testing
  num_train_examples: 500   # Quick test size
  num_eval_examples: 100

  # Real data (enabled by experiments)
  # dataset_name: "Anthropic/hh-rlhf"
  # format: "anthropic"
```

**Default behavior:** Uses synthetic data for quick validation of the pipeline.

**Why synthetic by default?**
- ✅ No download needed
- ✅ Fast to generate
- ✅ Tests pipeline end-to-end
- ✅ Good for development

**Real data:** Enabled by experiment configs like `reward_gpt2_hh_rlhf.yaml`.

---

## The Anthropic HH-RLHF Dataset

### What is HH-RLHF?

**Full name:** Anthropic's Human Preference Dataset for Harmless and Helpful AI

**Paper:** "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"
- Authors: Bai et al. (Anthropic, 2022)
- Link: https://arxiv.org/abs/2204.05862

**Purpose:** Train AI assistants to be both helpful (answer questions well) and harmless (avoid harmful outputs).

### Dataset Statistics

- **Total size:** ~169,000 preference pairs
  - Train split: ~161,000 examples
  - Test split: ~8,500 examples
- **Format:** Conversational preference pairs
- **Domain:** General assistant conversations
- **Languages:** English
- **License:** MIT

**HuggingFace:** `Anthropic/hh-rlhf`
**Download size:** ~170 MB

### Data Structure

Each example contains a conversation with two responses:

```python
{
  "chosen": "Human: What's the capital of France?\n\nAssistant: The capital of France is Paris. It's a beautiful city known for...",
  "rejected": "Human: What's the capital of France?\n\nAssistant: I don't know."
}
```

**Key points:**
- Multi-turn dialogues between Human and Assistant
- Both responses share the same prompt context
- Human annotators preferred "chosen" over "rejected"
- Last turn is what's being evaluated

### Data Collection Process

1. **Response Generation**
   - Two different AI models generate responses to the same prompt
   - Or: Same model with different parameters/prompts

2. **Human Annotation**
   - Annotators choose which response is better based on:
     - **Helpfulness:** Does it answer the question well?
     - **Harmlessness:** Does it avoid harmful/toxic content?
   - Multiple annotators per example for quality control

3. **Quality Filtering**
   - Agreement checks between annotators
   - Removing ambiguous examples
   - Balancing dataset

### Using HH-RLHF in Config

```yaml
# In experiment config
data:
  use_synthetic: false
  dataset_name: "Anthropic/hh-rlhf"
  dataset_config: null
  train_split: "train"
  eval_split: "test"
  num_train_examples: 10000      # Use subset for faster training
  num_eval_examples: 1000
  format: "anthropic"            # Parse conversation format
```

**Format parser:** Our code extracts:
- **Prompt:** Everything up to the last "Assistant:" marker
- **Chosen response:** The preferred Assistant response
- **Rejected response:** The rejected Assistant response

---

## Experiment Configs

### Quick Test: reward_gpt2_synthetic.yaml

**Purpose:** Fast validation of reward modeling pipeline

```yaml
# configs/experiment/reward_gpt2_synthetic.yaml
# @package _global_

defaults:
  - override /model: gpt2
  - override /technique: reward_model
  - override /data: preference
  - _self_  # ← Overrides base config

data:
  use_synthetic: true       # Generate synthetic preferences
  num_train_examples: 500
  num_eval_examples: 100

training:
  output_dir: "./outputs/reward_gpt2_synthetic"
  num_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 1e-5

model:
  name: "gpt2"
  use_lora: false           # Full fine-tuning for small test
```

**Usage:**
```bash
python scripts/train/train_reward_model.py experiment=reward_gpt2_synthetic device=cpu
```

**Expected time:** 5-10 minutes on CPU, 2-3 minutes on GPU

**Expected results:**
- Accuracy: 70-85% (synthetic data has clear patterns)
- Reward margin: > 0.5
- Good for testing the pipeline works

### Production: reward_gpt2_hh_rlhf.yaml

**Purpose:** Train on real human preference data

```yaml
# configs/experiment/reward_gpt2_hh_rlhf.yaml
# @package _global_

defaults:
  - override /model: gpt2
  - override /technique: reward_model
  - override /data: preference
  - _self_  # ← Critical for overriding base!

data:
  use_synthetic: false           # Use real data
  dataset_name: "Anthropic/hh-rlhf"
  train_split: "train"
  eval_split: "test"
  num_train_examples: 10000      # Subset (full = 161K)
  num_eval_examples: 1000
  format: "anthropic"

training:
  output_dir: "./outputs/reward_gpt2_hh_rlhf"
  num_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  fp16: false                    # Set to true for GPU

model:
  name: "gpt2"
  use_lora: true                 # Recommended for efficiency
  lora_config:
    r: 16
    lora_alpha: 32
```

**Usage:**
```bash
# GPU training (recommended)
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  device=cuda \
  training.fp16=true

# CPU training (slow but works)
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  device=cpu
```

**Expected time:**
- GPU (T4): 1-2 hours for 10K examples
- CPU (M1 Max): 6-8 hours for 10K examples

**Expected results:**
- Accuracy: 60-70% (real data is harder)
- Reward margin: positive and increasing
- Ready for use in PPO/RLHF

---

## Parameter Tuning Guide

### Key Parameters

#### Learning Rate

```yaml
training:
  learning_rate: 1e-5  # Conservative default
```

**Comparison:**
- **SFT:** 5e-5 (higher)
- **Reward Model:** 1e-5 (default)
- **DPO:** 5e-7 (much lower)

**Tuning:**
- If underfitting: Try 5e-5
- If overfitting: Try 5e-6
- Monitor: accuracy, reward margins

#### Number of Epochs

```yaml
training:
  num_epochs: 1  # Usually sufficient
```

**Why only 1 epoch?**
- Reward models learn rankings quickly
- More epochs → overfitting to specific pairs
- Monitor validation accuracy to decide

**Signs you need more epochs:**
- Accuracy still increasing at end of epoch 1
- Training loss not converged
- Validation accuracy still improving

**Signs of overfitting:**
- Training accuracy > 80%, validation < 65%
- Large gap between train and validation
- Reward margins too large (overconfident)

#### Batch Size

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  # Effective batch = 2 × 4 = 8
```

**Why small batch size?**
- Each example = 2 forward passes (chosen + rejected)
- Memory usage: ~2× compared to SFT
- Larger effective batch → more stable gradients

**Recommendations:**
- Single GPU (16GB): batch=2, accum=4 → effective=8
- Single GPU (24GB): batch=4, accum=2 → effective=8
- If OOM: batch=1, accum=8 → effective=8

#### LoRA Settings

```yaml
model:
  use_lora: true
  lora_config:
    r: 16              # Rank
    lora_alpha: 32     # Scaling factor
    lora_dropout: 0.05
    target_modules: ["c_attn", "c_proj"]
```

**Why LoRA?**
- ✅ Reduces trainable params by ~50×
- ✅ Faster training
- ✅ Less memory
- ✅ Reduces overfitting

**Tuning:**
- Smaller model: r=8, alpha=16 (faster, less capacity)
- Larger model: r=32, alpha=64 (more capacity)

---

## Common Issues and Solutions

### Issue 1: Using Synthetic Data Instead of Real

**Symptom:** Experiment with `hh_rlhf` in name prints "Generating synthetic data"

**Cause:** Missing `_self_` in experiment config defaults

**Solution:**
```yaml
# In experiment config
defaults:
  - override /model: gpt2
  - override /technique: reward_model
  - override /data: preference
  - _self_  # ← Add this!
```

Without `_self_`, base config overrides experiment config.

### Issue 2: Low Accuracy (<55%)

**Symptoms:**
- Accuracy hovers around 50% (random guessing)
- Reward margins near zero
- Model not learning preferences

**Solutions:**
1. **Check data quality:** Are preferences clear and consistent?
2. **Increase learning rate:** Try 5e-5
3. **Train longer:** Try 2 epochs
4. **Check tokenization:** Max length might be too short
5. **Verify loss:** Should be decreasing

```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=5e-5 \
  training.num_epochs=2
```

### Issue 3: Overfitting

**Symptoms:**
- Training accuracy: >80%
- Validation accuracy: <60%
- Large train/val gap

**Solutions:**
1. **Use LoRA:** `model.use_lora=true`
2. **More data:** Increase `num_train_examples`
3. **Regularization:** Increase `weight_decay`
4. **Early stopping:** Monitor validation, stop when plateaus

```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  model.use_lora=true \
  training.weight_decay=0.01 \
  checkpoint.load_best_model_at_end=true
```

### Issue 4: Out of Memory

**Symptoms:**
- CUDA out of memory error
- Process killed
- Slow swapping

**Solutions:**
1. **Reduce batch size:**
   ```bash
   training.per_device_train_batch_size=1 \
   training.gradient_accumulation_steps=8
   ```

2. **Use LoRA:** `model.use_lora=true`

3. **Use 8-bit:** `model.use_8bit=true`

4. **Reduce max length:**
   ```bash
   tokenizer.max_length=256
   ```

5. **Enable gradient checkpointing:** (if supported)

### Issue 5: Safetensors Error (Tied Weights)

**Symptom:**
```
RuntimeError: Some tensors share memory
[{'model.lm_head.weight', 'model.transformer.wte.weight'}]
```

**Cause:** GPT-2 has tied weights (word embeddings and LM head share memory)

**Solution:** Already fixed in our codebase! We override `_save()` to use PyTorch format.

**See:** `docs/TIED_WEIGHTS_GUIDE.md` for full explanation

---

## Advanced Configuration

### Using a Custom Base Model

```bash
python scripts/train/train_reward_model.py \
  model=opt-350m \
  experiment=reward_gpt2_hh_rlhf
```

### Training from an SFT Model

```yaml
# In experiment config
model:
  name: "./outputs/sft_gpt2/final_model"  # Path to SFT checkpoint
  use_lora: true
```

**Why this is good:**
- SFT model already understands instruction format
- Better starting point than base GPT-2
- Usually achieves higher accuracy

### Freezing Base Model

```yaml
technique:
  freeze_base_model: true  # Only train value head
```

**When to use:**
- You have a good pre-trained model
- Want very fast training
- Testing the pipeline

**Trade-off:** Lower accuracy but much faster

### Multiple Evaluation Splits

```yaml
data:
  eval_split: "test"
  # Can also use validation split
  # eval_split: "validation"
```

---

## Monitoring Training

### Key Metrics

**During training, watch:**

1. **Loss:** Should decrease steadily
   - Good: Smooth downward curve
   - Bad: Spikes, instability

2. **Accuracy:** % of pairs where R(chosen) > R(rejected)
   - Random: 50%
   - Good: 60-70%
   - Excellent: >70%
   - Too high (>85%): Possible overfitting

3. **Reward Margin:** R(chosen) - R(rejected)
   - Should be positive and increasing
   - Typical final value: 0.5-2.0
   - If negative: Model prefers rejected (very bad!)

4. **Reward Statistics:**
   - Mean chosen reward: Should increase
   - Mean rejected reward: Should stay lower
   - Separation is what matters

### Logging

**TensorBoard:**
```bash
tensorboard --logdir outputs/reward_gpt2_hh_rlhf/logs
```

**Weights & Biases:**
```yaml
logging:
  use_wandb: true
  wandb_project: "llm-post-training"
  wandb_run_name: "reward_exp_1"
```

---

## Summary: Quick Reference

### Commands

```bash
# Quick test (synthetic)
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_synthetic \
  device=cpu

# Real training (Anthropic HH-RLHF)
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  device=cuda \
  training.fp16=true

# Override any parameter
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=5e-5 \
  data.num_train_examples=50000
```

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **learning_rate** | 1e-5 | 5e-6 to 5e-5 | Learning speed |
| **num_epochs** | 1 | 1-2 | Training iterations |
| **batch_size** | 2 | 1-4 | Memory vs speed |
| **gradient_accum** | 4 | 2-8 | Effective batch size |
| **use_lora** | varies | true/false | Parameter efficiency |

### Expected Results

| Dataset | Examples | Accuracy | Time (GPU) |
|---------|----------|----------|------------|
| Synthetic | 500 | 70-85% | 2-3 min |
| HH-RLHF | 10K | 60-70% | 1-2 hours |
| HH-RLHF | 161K (full) | 65-75% | 15-20 hours |

### Config Hierarchy

```
config_reward.yaml (BASE)
    ↓ loads
model/gpt2.yaml + technique/reward_model.yaml + data/preference.yaml
    ↓ overridden by
experiment/reward_gpt2_hh_rlhf.yaml (SPECIFIC)
    ↓ overridden by
Command-line arguments
```

---

## References

**Papers:**
- [Anthropic HH-RLHF Dataset](https://arxiv.org/abs/2204.05862)
- [Bradley-Terry Model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- [RLHF: Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)

**Code:**
- Reward model implementation: `src/models/reward.py`
- Training logic: `src/core/reward_modeling/`
- This config: `configs/config_reward.yaml`

**Related Docs:**
- `docs/REWARD_MODELING_THEORY.md` - Mathematical explanation
- `docs/CONFIGURATION_GUIDE.md` - General Hydra/OmegaConf guide
- `docs/TIED_WEIGHTS_GUIDE.md` - GPT-2 tied weights issue
- `notebooks/02_reward_modeling.ipynb` - Interactive tutorial
