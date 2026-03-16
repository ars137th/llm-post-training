# Phase 5 (PPO/RLHF) Testing Guide

This document provides step-by-step testing commands for all Phase 5 components.
Run these tests to verify each component works correctly.

---

## Quick Test Summary

```bash
# Run all tests in sequence
cd /Users/akhil.shah/code/claude_sandbox/llm-post-training

# 1. Unit Tests
python -c "from src.core.ppo import *; print('✅ PPO imports work')"

# 2. Component Tests
python tests/test_ppo_loss.py
python tests/test_ppo_gae.py
python tests/test_ppo_buffer.py

# 3. Integration Test (synthetic data)
python scripts/train/train_ppo.py \
    model=gpt2 \
    data.use_synthetic=true \
    data.num_prompts=10 \
    training.num_rollouts=2 \
    device=cpu

# 4. Full Training (small scale)
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic \
    device=cpu
```

---

## 1. Import Tests

### Test 1.1: PPO Module Imports

**Purpose**: Verify all PPO components can be imported

**Command**:
```bash
python -c "
from src.core.ppo import (
    ppo_loss, value_loss, compute_gae, RolloutBuffer,
    PPOTrainer, PPOConfig
)
print('✅ All PPO imports successful')
"
```

**Expected Output**:
```
✅ All PPO imports successful
```

**Troubleshooting**:
- If `ModuleNotFoundError`: Check `PYTHONPATH` or run from project root
- If `ImportError`: Check dependencies in requirements

---

## 2. Unit Tests - PPO Loss Functions

### Test 2.1: PPO Clipped Loss

**Purpose**: Test PPO loss computation with clipping

**File**: Create `tests/test_ppo_loss.py`

**Code**:
```python
import torch
from src.core.ppo.loss import ppo_loss

# Test case: simple advantage
log_probs = torch.tensor([0.0, 0.0, 0.0])
old_log_probs = torch.tensor([0.0, 0.0, 0.0])
advantages = torch.tensor([1.0, -1.0, 0.5])

loss, details = ppo_loss(
    log_probs=log_probs,
    old_log_probs=old_log_probs,
    advantages=advantages,
    clip_range=0.2,
    return_details=True,
)

print(f"Loss: {loss.item():.4f}")
print(f"Clip fraction: {details['clip_fraction']:.2%}")
print(f"Approx KL: {details['approx_kl']:.6f}")

assert loss.item() < 0, "PPO loss should be negative (we maximize objective)"
print("✅ PPO loss test passed")
```

**Command**:
```bash
python tests/test_ppo_loss.py
```

**Expected Output**:
```
Loss: -0.5667
Clip fraction: 0.00%
Approx KL: 0.000000
✅ PPO loss test passed
```

### Test 2.2: Value Loss

**File**: Add to `tests/test_ppo_loss.py`

**Code**:
```python
from src.core.ppo.loss import value_loss

values = torch.tensor([1.5, 2.0, 1.0])
returns = torch.tensor([2.0, 1.5, 1.0])

loss, details = value_loss(
    values=values,
    returns=returns,
    return_details=True,
)

print(f"Value Loss: {loss.item():.4f}")
print(f"Explained Variance: {details['explained_variance']:.4f}")
assert loss.item() > 0, "Value loss should be positive (MSE)"
print("✅ Value loss test passed")
```

### Test 2.3: RLHF Reward with KL Penalty

**Code**:
```python
from src.core.ppo.loss import compute_rlhf_reward

reward_model_scores = torch.tensor([5.0, 3.0, 4.0])
log_probs = torch.tensor([0.0, 0.0, 0.0])
ref_log_probs = torch.tensor([0.0, 0.0, 0.0])

rewards, details = compute_rlhf_reward(
    reward_model_scores=reward_model_scores,
    log_probs=log_probs,
    ref_log_probs=ref_log_probs,
    kl_coef=0.05,
    return_details=True,
)

print(f"Total Reward: {details['reward_total_mean']:.4f}")
print(f"RM Reward: {details['reward_model_mean']:.4f}")
print(f"KL Penalty: {details['kl_penalty_mean']:.6f}")
assert rewards.mean().item() > 0, "Rewards should be positive"
print("✅ RLHF reward test passed")
```

---

## 3. Unit Tests - GAE

### Test 3.1: Simple GAE Computation

**File**: Create `tests/test_ppo_gae.py`

**Code**:
```python
import torch
from src.core.ppo.gae import compute_gae

# Single reward per sequence (simple case)
rewards = torch.tensor([1.0, 2.0, -0.5])
values = torch.tensor([0.5, 1.5, 0.0, 0.0])  # +1 for terminal

advantages, details = compute_gae(
    rewards=rewards,
    values=values,
    gamma=0.99,
    lam=0.95,
    return_details=True,
)

print(f"Advantages: {advantages}")
print(f"Advantage Mean: {details['advantage_mean']:.4f}")
print(f"Advantage Std: {details['advantage_std']:.4f}")

assert advantages.shape == rewards.shape, "Advantages shape should match rewards"
print("✅ GAE test passed")
```

**Command**:
```bash
python tests/test_ppo_gae.py
```

**Expected Output**:
```
Advantages: tensor([0.5000, 0.5000, -0.5000])
Advantage Mean: 0.1667
Advantage Std: 0.4714
✅ GAE test passed
```

### Test 3.2: Advantage Normalization

**Code**:
```python
from src.core.ppo.gae import normalize_advantages

advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
normalized = normalize_advantages(advantages)

print(f"Original: {advantages}")
print(f"Normalized: {normalized}")
print(f"Mean: {normalized.mean().item():.6f}")
print(f"Std: {normalized.std().item():.6f}")

assert abs(normalized.mean().item()) < 1e-6, "Mean should be ~0"
assert abs(normalized.std().item() - 1.0) < 1e-6, "Std should be ~1"
print("✅ Normalization test passed")
```

---

## 4. Unit Tests - Rollout Buffer

### Test 4.1: Buffer Add and Compute

**File**: Create `tests/test_ppo_buffer.py`

**Code**:
```python
import torch
from src.core.ppo.buffer import RolloutBuffer

buffer = RolloutBuffer(gamma=0.99, lam=0.95)

# Add dummy batch
batch_size = 4
seq_len = 10

buffer.add(
    prompt_input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
    prompt_attention_mask=torch.ones(batch_size, seq_len),
    response_input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
    response_attention_mask=torch.ones(batch_size, seq_len),
    input_ids=torch.randint(0, 1000, (batch_size, seq_len * 2)),
    attention_mask=torch.ones(batch_size, seq_len * 2),
    old_log_probs=torch.randn(batch_size),
    ref_log_probs=torch.randn(batch_size),
    rewards=torch.randn(batch_size),
    values=torch.randn(batch_size),
)

print(f"Buffer size: {len(buffer)}")
assert len(buffer) == batch_size, f"Expected {batch_size}, got {len(buffer)}"

# Compute advantages
buffer.compute_advantages()
stats = buffer.get_stats()

print(f"Reward mean: {stats['reward_mean']:.4f}")
print(f"Advantage mean: {stats['advantage_mean']:.4f}")

assert 'advantage_mean' in stats, "Advantages should be computed"
print("✅ Buffer test passed")
```

**Command**:
```bash
python tests/test_ppo_buffer.py
```

**Expected Output**:
```
Buffer size: 4
Reward mean: -0.1234
Advantage mean: 0.0000
✅ Buffer test passed
```

### Test 4.2: Mini-batch Sampling

**Code**:
```python
# Continue from previous test
mini_batch_size = 2
batches = list(buffer.sample_batches(
    batch_size=mini_batch_size,
    device=torch.device('cpu'),
    num_epochs=1,
    shuffle=False,
))

print(f"Number of mini-batches: {len(batches)}")
print(f"First mini-batch size: {len(batches[0])}")

assert len(batches) == batch_size // mini_batch_size, "Should have 2 mini-batches"
assert len(batches[0]) == mini_batch_size, "Mini-batch size should be 2"
print("✅ Mini-batch sampling test passed")
```

---

## 5. Integration Test - Synthetic Data

### Test 5.1: End-to-End PPO Training (Minimal)

**Purpose**: Test complete PPO pipeline with synthetic data

**Command**:
```bash
python scripts/train/train_ppo.py \
    model.name=gpt2 \
    data.use_synthetic=true \
    data.num_prompts=10 \
    training.num_rollouts=2 \
    training.batch_size=4 \
    training.ppo_epochs=2 \
    training.mini_batch_size=2 \
    device=cpu \
    logging.use_wandb=false
```

**Expected Output**:
```
Starting PPO training on 10 prompts
Device: cpu
Rollouts: 2
Batch size: 4
PPO epochs: 2

[Rollout 1/2]
  Time: 45.23s
  Reward: 0.1234 ± 0.4567
  Policy Loss: -0.3456
  Value Loss: 0.1234
  KL: 0.000123
  Clip Fraction: 15.00%
  Explained Variance: 0.5678
  KL Coef: 0.050000

[Rollout 2/2]
  Time: 43.12s
  Reward: 0.2345 ± 0.3456
  Policy Loss: -0.2345
  Value Loss: 0.0987
  KL: 0.000234
  Clip Fraction: 20.00%
  Explained Variance: 0.6789
  KL Coef: 0.050000

✅ PPO training complete!
Saved models to ./outputs/ppo_test/
```

**Success Criteria**:
- Training completes without errors
- Reward values are reasonable (not NaN or Inf)
- KL stays below 0.01
- Clip fraction between 10-30%
- Explained variance increases

**Time**: ~2-3 minutes on CPU

---

## 6. Component Tests with Real Models

### Test 6.1: Test Generation

**Purpose**: Verify response generation works

**Command**:
```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.language import LanguageModel

# Load tiny model
model = LanguageModel.from_pretrained('gpt2', use_lora=False)
tokenizer = model.tokenizer

# Generate
prompts = ['Hello, how are you?']
inputs = tokenizer(prompts, return_tensors='pt')
outputs = model.model.generate(
    inputs['input_ids'],
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Generated: {response}')
print('✅ Generation test passed')
"
```

**Expected Output**:
```
Generated: Hello, how are you? I'm doing great, thanks for asking!
✅ Generation test passed
```

### Test 6.2: Test Reward Model

**Purpose**: Verify reward model scoring works

**Command**:
```bash
python -c "
import torch
from src.models.language import LanguageModel
from src.models.reward import RewardModel

# Load base and create reward model
base = LanguageModel.from_pretrained('gpt2', use_lora=False)
rm = RewardModel(base_model=base, freeze_base=False)

# Score a sequence
text = 'This is a good response.'
inputs = rm.tokenizer(text, return_tensors='pt')
reward = rm(inputs['input_ids'], inputs['attention_mask'])

print(f'Reward: {reward.item():.4f}')
print('✅ Reward model test passed')
"
```

### Test 6.3: Test Four Models Together

**Purpose**: Verify all four models can coexist in memory

**File**: Create `tests/test_four_models.py`

**Code**:
```python
import torch
from src.models.language import LanguageModel
from src.models.reward import RewardModel

print("Loading models...")

# Actor
actor = LanguageModel.from_pretrained('gpt2', use_lora=True)
print(f"✅ Actor: {actor.num_trainable_parameters:,} trainable params")

# Critic (same as actor but with value head)
critic_base = LanguageModel.from_pretrained('gpt2', use_lora=True)
critic = RewardModel(base_model=critic_base, freeze_base=False)
print(f"✅ Critic: {critic.num_trainable_parameters:,} trainable params")

# Reference (frozen)
reference = LanguageModel.from_pretrained('gpt2', use_lora=False)
for p in reference.model.parameters():
    p.requires_grad = False
print(f"✅ Reference: frozen")

# Reward Model (frozen)
rm_base = LanguageModel.from_pretrained('gpt2', use_lora=False)
reward_model = RewardModel(base_model=rm_base, freeze_base=True)
print(f"✅ Reward Model: frozen")

# Memory check
import psutil
import os
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"\nMemory usage: {memory_mb:.0f} MB")

print("\n✅ Four models test passed")
```

**Command**:
```bash
python tests/test_four_models.py
```

**Expected Output**:
```
Loading models...
✅ Actor: 294,912 trainable params
✅ Critic: 295,169 trainable params
✅ Reference: frozen
✅ Reward Model: frozen

Memory usage: 2048 MB

✅ Four models test passed
```

---

## 7. Full Training Tests

### Test 7.1: Quick Training Run (10 rollouts)

**Purpose**: Verify training runs without crashes

**Command**:
```bash
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic \
    training.num_rollouts=10 \
    data.num_prompts=50 \
    device=cpu
```

**Time**: ~10-15 minutes on CPU

**Check**:
- [ ] Training completes
- [ ] Models saved to `./outputs/ppo_gpt2_synthetic/`
- [ ] Logs show reasonable metrics
- [ ] Reward increases over time
- [ ] KL stays stable (< 0.01)

### Test 7.2: Training with Evaluation

**Command**:
```bash
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic \
    training.num_rollouts=20 \
    evaluation.do_eval=true \
    evaluation.eval_every=5 \
    device=cpu
```

**Check**:
- [ ] Evaluation runs every 5 rollouts
- [ ] Evaluation metrics printed
- [ ] Best model saved based on reward

---

## 8. Visualization Tests

### Test 8.1: Plot Training Curves

**File**: Create `scripts/utils/plot_ppo_metrics.py`

**Command**:
```bash
python scripts/utils/plot_ppo_metrics.py \
    --log_dir ./outputs/ppo_gpt2_synthetic/logs \
    --output ./outputs/ppo_plots.png
```

**Check**:
- [ ] Plot saved to specified path
- [ ] Shows reward, loss, KL, clip fraction
- [ ] Trends make sense (reward ↑, loss ↓)

---

## 9. Comparison Tests

### Test 9.1: PPO vs DPO on Same Data

**Commands**:
```bash
# Train DPO
python scripts/train/train_dpo.py \
    experiment=dpo_gpt2_synthetic \
    data.num_train_examples=100 \
    device=cpu

# Train PPO
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic \
    data.num_prompts=100 \
    training.num_rollouts=20 \
    device=cpu

# Compare
python scripts/evaluate/compare_ppo_dpo.py \
    --dpo_model ./outputs/dpo_gpt2_synthetic/final_model \
    --ppo_model ./outputs/ppo_gpt2_synthetic/actor \
    --prompts ./datasets/test_prompts.txt
```

**Check**:
- [ ] Both models train successfully
- [ ] Comparison report generated
- [ ] Quality metrics computed
- [ ] Generation examples shown

---

## 10. Google Colab Test

### Test 10.1: Run in Colab with GPU

**Notebook**: Create and upload to Colab

**Commands**:
```python
# Cell 1: Setup
!git clone https://github.com/ars137th/llm-post-training.git
%cd llm-post-training
!pip install -e .

# Cell 2: Quick test
!python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic \
    training.num_rollouts=5 \
    device=cuda \
    training.fp16=true

# Cell 3: Check outputs
!ls -lh outputs/ppo_gpt2_synthetic/
```

**Expected Time**: ~5-10 minutes on T4 GPU

---

## 11. Troubleshooting Guide

### Issue: Out of Memory

**Symptoms**: CUDA OOM or system freeze

**Solutions**:
```bash
# Use smaller batch sizes
python scripts/train/train_ppo.py \
    training.batch_size=4 \
    training.mini_batch_size=2

# Use gradient accumulation
python scripts/train/train_ppo.py \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4

# Offload models
python scripts/train/train_ppo.py \
    training.offload_ref_model=true \
    training.offload_reward_model=true
```

### Issue: KL Explodes (> 10)

**Symptoms**: KL divergence shoots up, model generates gibberish

**Solutions**:
```bash
# Lower learning rate
python scripts/train/train_ppo.py \
    training.learning_rate=5e-7

# Increase KL coefficient
python scripts/train/train_ppo.py \
    training.kl_coef=0.1

# Use adaptive KL
python scripts/train/train_ppo.py \
    training.use_adaptive_kl=true \
    training.target_kl=0.01
```

### Issue: Reward Hacking

**Symptoms**: High reward but poor quality responses

**Solutions**:
```bash
# Clip rewards
python scripts/train/train_ppo.py \
    training.reward_clip=10.0

# Increase KL penalty
python scripts/train/train_ppo.py \
    training.kl_coef=0.1

# Use better reward model (train longer in Phase 3)
```

### Issue: Training Too Slow

**Symptoms**: Hours per rollout

**Solutions**:
```bash
# Enable mixed precision
python scripts/train/train_ppo.py \
    training.fp16=true  # or bf16=true

# Use smaller max_length
python scripts/train/train_ppo.py \
    data.max_response_length=128

# Profile generation
python -m cProfile -s cumtime scripts/train/train_ppo.py ...
```

---

## 12. Acceptance Criteria

Before considering Phase 5 complete, verify:

- [ ] All unit tests pass
- [ ] Integration test completes successfully
- [ ] Four models fit in memory (< 16GB on GPU, < 32GB on CPU)
- [ ] Training completes without crashes
- [ ] Reward increases over rollouts
- [ ] KL stays below threshold (< 0.01)
- [ ] Clip fraction reasonable (10-30%)
- [ ] Generated text quality improves
- [ ] Models can be saved and loaded
- [ ] Works on both CPU and GPU
- [ ] Works in Google Colab
- [ ] Documentation covers all features
- [ ] Tutorial notebook runs end-to-end

---

## Quick Reference - Common Commands

```bash
# Minimal test (fastest)
python scripts/train/train_ppo.py \
    model=gpt2 data.use_synthetic=true data.num_prompts=5 \
    training.num_rollouts=1 device=cpu

# Standard test (recommended)
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic device=cpu

# Production test (full scale)
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_hh_rlhf \
    device=cuda training.fp16=true

# Debug mode (verbose logging)
python scripts/train/train_ppo.py \
    experiment=ppo_gpt2_synthetic \
    logging.log_level=DEBUG \
    training.num_rollouts=2
```

---

## Next Steps After Testing

Once all tests pass:
1. Run full-scale training experiment
2. Evaluate on held-out prompts
3. Compare with DPO baseline
4. Document findings in tutorial notebook
5. Move to Phase 6 (Multimodal)

Good luck! 🚀
