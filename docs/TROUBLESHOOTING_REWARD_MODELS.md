# Troubleshooting Reward Model Training

This guide helps diagnose and fix poor reward model performance.

## Table of Contents
- [Understanding Reward Model Metrics](#understanding-reward-model-metrics)
- [Common Problems and Solutions](#common-problems-and-solutions)
- [Systematic Debugging Process](#systematic-debugging-process)
- [Hyperparameter Tuning Guide](#hyperparameter-tuning-guide)
- [Real-World Examples](#real-world-examples)

---

## Understanding Reward Model Metrics

### Ranking Accuracy

**What it means:** Percentage of preference pairs where the model correctly ranks chosen > rejected.

**Interpretation:**
- **50%**: Random guessing (model learned nothing) ❌
- **55-60%**: Very weak signal (barely better than random) ⚠️
- **60-65%**: Weak but learning (needs improvement) ⚠️
- **65-70%**: Acceptable performance ✓
- **70-75%**: Good performance (ready for RLHF) ✅
- **75-80%**: Very good performance ✅✅
- **80%+**: Excellent (unlikely on real data, check for overfitting) ⚠️

### Reward Margin

**What it means:** Average difference between rewards for chosen vs rejected responses.

**Interpretation:**
- **< 0.01**: Extremely weak discrimination (model is uncertain) ❌
- **0.01-0.05**: Weak discrimination (low confidence) ⚠️
- **0.05-0.20**: Moderate discrimination ✓
- **0.20-0.50**: Strong discrimination ✅
- **> 0.50**: Very strong (common in synthetic data) ✅

**Important:** A model can have high accuracy but low margin (confident but close calls) or low accuracy with high margin (confidently wrong).

### Loss

**What to look for:**
- Loss should **decrease steadily** during training
- If loss plateaus immediately → learning rate too low
- If loss oscillates wildly → learning rate too high
- If loss increases → training collapsed (restart with lower LR)

---

## Common Problems and Solutions

### Problem 1: Accuracy 50-60% (Not Learning)

**Symptoms:**
- Accuracy barely better than random
- Margin < 0.01
- Loss not decreasing or decreasing very slowly

**Root Causes:**
1. **Learning rate too high** (most common)
2. Data not being parsed correctly
3. Model not updating (frozen layers)
4. Batch size too small (noisy gradients)

**Solutions (in order of likelihood):**

#### Solution 1A: Lower Learning Rate (Try First!)
```bash
# Default uses 1e-5, try 2e-6 or 1e-6
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=2e-6 \
  training.fp16=true
```

**Why this works:** Reward models are sensitive to LR. Too high → overshoots optimal weights.

#### Solution 1B: Increase Effective Batch Size
```bash
# Larger batch = more stable gradients
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.per_device_train_batch_size=8 \
  training.gradient_accumulation_steps=4  # Effective batch = 32
  training.learning_rate=2e-6
```

#### Solution 1C: Check Data Loading
```bash
# Run with debug logging
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  logging.log_level=debug
```

Look for:
- "Processing preference pair" messages
- Correct parsing of chosen/rejected responses
- No errors or warnings about data format

#### Solution 1D: Verify Model is Updating
Check that LoRA is configured correctly:
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  model.use_lora=true \
  model.lora_config.r=32  # Larger rank = more capacity
```

Or try without LoRA (full fine-tuning):
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  model.use_lora=false \
  training.per_device_train_batch_size=4  # Lower batch for memory
```

---

### Problem 2: Accuracy Starts Good, Then Drops

**Symptoms:**
- Initial accuracy 60-65%
- Later accuracy 55-58%
- Loss starts decreasing, then increases
- Margin decreases over time

**Root Cause:** Overfitting or learning rate too high for later training

**Solutions:**

#### Solution 2A: Lower Learning Rate
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=1e-6  # Even lower than usual
```

#### Solution 2B: Use Learning Rate Scheduler
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.lr_scheduler_type=cosine \
  training.warmup_steps=500
```

#### Solution 2C: Early Stopping
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  checkpoint.load_best_model_at_end=true \
  checkpoint.metric_for_best_model=eval_accuracy
```

#### Solution 2D: Add Weight Decay
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.weight_decay=0.01
```

---

### Problem 3: Model Predicts Same Reward for Everything

**Symptoms:**
- Margin ≈ 0.000
- Accuracy exactly 50%
- Loss plateaus at same value

**Root Cause:** Training collapsed (reward head outputs constant)

**Solutions:**

#### Solution 3A: Restart with Much Lower LR
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=5e-7  # Very conservative
  training.warmup_steps=100
```

#### Solution 3B: Reinitialize Reward Head
```bash
# Start from scratch, don't resume from checkpoint
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  checkpoint.resume_from_checkpoint=null
```

#### Solution 3C: Check for Frozen Layers
Ensure the value head is trainable:
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  technique.freeze_base_model=false
```

---

### Problem 4: Training Too Slow / Out of Memory

**Symptoms:**
- OOM errors
- Very slow training (< 1 step/sec)
- GPU not fully utilized

**Solutions:**

#### Solution 4A: Enable FP16
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.fp16=true  # 2x faster, 2x less memory
```

#### Solution 4B: Reduce Batch Size, Increase Accumulation
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=8  # Same effective batch
```

#### Solution 4C: Use LoRA
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  model.use_lora=true \
  model.lora_config.r=16  # Smaller = less memory
```

#### Solution 4D: Use Gradient Checkpointing
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.gradient_checkpointing=true  # Slower but saves memory
```

---

### Problem 5: Good Accuracy on Synthetic, Poor on Real Data

**Symptoms:**
- Synthetic data: 75-85% accuracy ✅
- Anthropic HH-RLHF: 55-60% accuracy ❌
- Model works but can't generalize to real preferences

**Root Cause:** Real human preferences are ambiguous and noisy (this is normal!)

**Solutions:**

#### Solution 5A: Use More Training Data
```bash
# 10K examples may not be enough
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  data.num_train_examples=50000  # Use full dataset
  data.num_eval_examples=2000
```

#### Solution 5B: Train Longer
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.num_epochs=3  # Multiple passes over data
```

#### Solution 5C: Lower Learning Rate
```bash
# Real data needs gentler optimization
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=2e-6 \
  training.num_epochs=2
```

#### Solution 5D: Increase Model Capacity
```bash
# Larger LoRA rank or full fine-tuning
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  model.lora_config.r=64 \
  model.lora_config.lora_alpha=128
```

---

## Systematic Debugging Process

### Step 1: Verify Basic Setup Works

```bash
# Quick test on synthetic data (should get 75%+ accuracy)
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_synthetic \
  training.num_epochs=3 \
  training.fp16=true \
  data.num_train_examples=2000
```

**Expected:** 75-85% accuracy in 5 minutes

**If this fails:**
- Problem with basic setup (GPU, dependencies, code)
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall: `pip install -e ".[gpu,experiment]"`

**If this works but real data fails:**
- Problem is specific to real data training
- Continue to Step 2

### Step 2: Check Data Loading

```bash
# Run with debug logging
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  logging.log_level=debug \
  data.num_train_examples=100  # Small subset for quick test
```

**Look for:**
- "Loading Anthropic/hh-rlhf from HuggingFace" (not synthetic data)
- "Processing preference pair" messages
- Correct format: chosen/rejected responses
- No parsing errors

### Step 3: Test Different Learning Rates

```bash
# Try 3 different learning rates
for lr in 1e-6 2e-6 5e-6; do
  python scripts/train/train_reward_model.py \
    experiment=reward_gpt2_hh_rlhf \
    training.learning_rate=$lr \
    training.output_dir="./outputs/reward_lr_${lr}" \
    data.num_train_examples=5000
done
```

**Compare results:** Pick the LR with highest eval accuracy

### Step 4: Increase Training Duration

```bash
# Use best LR from Step 3, train longer
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=2e-6 \
  training.num_epochs=3 \
  data.num_train_examples=10000
```

### Step 5: Scale Up Data

```bash
# If still not good enough, use more data
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.learning_rate=2e-6 \
  training.num_epochs=2 \
  data.num_train_examples=50000
```

---

## Hyperparameter Tuning Guide

### Learning Rate

**Impact:** Most important hyperparameter for reward models

| Learning Rate | Effect | When to Use |
|---------------|--------|-------------|
| 1e-6 | Very conservative, slow learning | If training collapsed or accuracy drops |
| 2e-6 | Conservative, stable | **Recommended default for real data** |
| 5e-6 | Moderate | If 2e-6 is too slow |
| 1e-5 | Aggressive | Synthetic data or large batches |
| 2e-5 | Very aggressive | Usually too high, avoid |

**Rule of thumb:** Start with 2e-6, halve if unstable, double if too slow.

### Number of Epochs

| Epochs | Effect | When to Use |
|--------|--------|-------------|
| 1 | Quick training | Large dataset (50K+ examples) |
| 2 | Balanced | **Recommended for most cases** |
| 3 | Thorough | Small dataset (10K examples) |
| 5+ | Risk of overfitting | Only with early stopping |

**Rule of thumb:** Fewer epochs for larger datasets.

### Batch Size and Gradient Accumulation

**Effective batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps**

| Effective Batch | Effect | When to Use |
|-----------------|--------|-------------|
| 8 | Small, noisy gradients | Small GPUs, quick experiments |
| 16 | Moderate | **Recommended default** |
| 32 | Large, stable gradients | **Recommended for best results** |
| 64+ | Very stable | Large-scale training |

**For Colab T4 (16GB):**
```bash
per_device_train_batch_size=8
gradient_accumulation_steps=2  # Effective batch = 16
```

**For larger GPUs (A100, 40GB):**
```bash
per_device_train_batch_size=16
gradient_accumulation_steps=2  # Effective batch = 32
```

### LoRA Configuration

| Parameter | Effect | Recommended Value |
|-----------|--------|-------------------|
| `r` (rank) | Model capacity | 16 (fast), 32 (balanced), 64 (high capacity) |
| `lora_alpha` | Learning rate scaling | 2×r (e.g., r=32 → alpha=64) |
| `lora_dropout` | Regularization | 0.05 (default) |
| `target_modules` | What to adapt | `["c_attn", "c_proj"]` (GPT-2) |

**Rule of thumb:** If accuracy is poor, try doubling `r`.

### Training Data Size

| Examples | Accuracy (Typical) | Training Time (T4) | When to Use |
|----------|-------------------|-------------------|-------------|
| 1K | 60-65% | 2-3 min | Quick test |
| 5K | 63-67% | 8-10 min | Development |
| 10K | 65-70% | 15-20 min | **Good baseline** |
| 30K | 68-72% | 40-60 min | **Recommended** |
| 50K | 70-73% | 1-1.5 hrs | Production |
| 160K (full) | 72-75% | 3-4 hrs | Maximum quality |

**Diminishing returns after 30-50K examples for GPT-2 size models.**

---

## Real-World Examples

### Example 1: Typical Poor Performance (What You're Experiencing)

**Setup:**
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.fp16=true \
  training.per_device_train_batch_size=8 \
  data.num_train_examples=10000
```

**Results:**
- Accuracy: 56.10% ❌
- Margin: 0.0039 ❌
- Time: 11 minutes

**Diagnosis:** Learning rate too high (1e-5), not enough training

**Fix:**
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.fp16=true \
  training.learning_rate=2e-6 \
  training.num_epochs=2 \
  training.per_device_train_batch_size=8 \
  data.num_train_examples=30000
```

**Expected:** 68-72% accuracy, 40 minutes

---

### Example 2: Good Baseline Performance

**Setup:**
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.fp16=true \
  training.learning_rate=2e-6 \
  training.num_epochs=2 \
  training.per_device_train_batch_size=8 \
  training.gradient_accumulation_steps=2 \
  model.lora_config.r=32 \
  model.lora_config.lora_alpha=64 \
  data.num_train_examples=30000 \
  data.num_eval_examples=2000
```

**Expected Results:**
- Accuracy: 68-72% ✅
- Margin: 0.05-0.15 ✅
- Time: 40-50 minutes on T4

**Why this works:**
- Lower LR (2e-6) → stable learning
- 2 epochs → sufficient training
- 30K examples → enough data
- Larger LoRA rank (32) → more capacity
- Larger effective batch (16) → stable gradients

---

### Example 3: Production-Quality Model

**Setup:**
```bash
python scripts/train/train_reward_model.py \
  experiment=reward_gpt2_hh_rlhf \
  training.fp16=true \
  training.learning_rate=2e-6 \
  training.num_epochs=2 \
  training.per_device_train_batch_size=16 \
  training.gradient_accumulation_steps=2 \
  model.lora_config.r=64 \
  model.lora_config.lora_alpha=128 \
  data.num_train_examples=100000 \
  data.num_eval_examples=5000
```

**Expected Results:**
- Accuracy: 72-75% ✅✅
- Margin: 0.10-0.20 ✅✅
- Time: 2-3 hours on A100

**Use case:** Production RLHF training

---

## Quick Reference: Problem → Solution

| Problem | Quick Fix | Command |
|---------|-----------|---------|
| Accuracy 50-60% | Lower LR | `training.learning_rate=2e-6` |
| Loss not decreasing | Even lower LR | `training.learning_rate=1e-6` |
| Accuracy plateaus | More data + epochs | `data.num_train_examples=50000 training.num_epochs=2` |
| Accuracy drops | Lower LR + early stop | `training.learning_rate=1e-6 checkpoint.load_best_model_at_end=true` |
| Margin < 0.01 | More data + lower LR | `data.num_train_examples=30000 training.learning_rate=2e-6` |
| OOM error | Lower batch + FP16 | `training.per_device_train_batch_size=4 training.fp16=true` |
| Too slow | Enable FP16 + LoRA | `training.fp16=true model.use_lora=true` |

---

## Summary: Best Practices

1. **Always start with lower learning rates for real data (2e-6, not 1e-5)**
2. **Use effective batch size ≥ 16 for stable training**
3. **Train for 2-3 epochs on small datasets (10K), 1 epoch on large (50K+)**
4. **Use 30K-50K examples for good results (diminishing returns after)**
5. **Enable FP16 on GPU for 2x speedup**
6. **Test on synthetic data first (should get 75%+)**
7. **Real data accuracy 65-70% is acceptable, 70%+ is good**
8. **If accuracy < 60%, lower learning rate first**

---

## Still Having Issues?

1. **Run the diagnostic test** (synthetic data should work)
2. **Check training logs** for loss curves (should decrease)
3. **Verify data loading** (debug mode shows parsing)
4. **Try the optimal configuration** from Example 2 above
5. **Report issue** with full logs and config

For more help, see:
- `docs/REWARD_MODELING_CONFIGURATION.md` - Configuration guide
- `docs/CONFIGURATION_GUIDE.md` - Hydra/OmegaConf guide
- `scripts/debug_config.py` - Config debugging tool
