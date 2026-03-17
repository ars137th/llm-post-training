# Tied Weights in Language Models: A Deep Dive

This document explains tied weights in language models, why GPT-2 uses them, the safetensors compatibility issue, and how this compares to modern LLMs.

## Table of Contents
- [What Are Tied Weights?](#what-are-tied-weights)
- [Why Does GPT-2 Use Tied Weights?](#why-does-gpt-2-use-tied-weights)
- [The Safetensors Issue](#the-safetensors-issue)
- [How We Fixed It](#how-we-fixed-it)
- [Modern LLMs: The Evolution](#modern-llms-the-evolution)
- [Practical Implications](#practical-implications)
- [Debugging Tied Weights](#debugging-tied-weights)

---

## What Are Tied Weights?

**Tied weights** (also called **weight tying** or **weight sharing**) is when two or more layers in a neural network share the same weight tensor in memory.

### In GPT-2 Specifically

GPT-2 ties two weight matrices:

1. **Input Embedding Layer** (`wte.weight`): Maps token IDs to embeddings
   - Shape: `[vocab_size, hidden_size]`
   - Example: `[50257, 768]` for GPT-2-small

2. **Output Language Model Head** (`lm_head.weight`): Maps hidden states to vocabulary logits
   - Shape: `[vocab_size, hidden_size]`
   - Same shape as `wte.weight`

**Key point:** These are not two separate weight matrices—they are **the same tensor** in memory.

### Visual Representation

```
Input Token IDs
       ↓
┌─────────────────┐
│  wte.weight     │  ← Token embeddings (vocab_size × hidden_size)
│  [50257, 768]   │
└─────────────────┘
       ↓
┌─────────────────┐
│  Transformer    │
│  Blocks (12)    │
└─────────────────┘
       ↓
┌─────────────────┐
│  lm_head.weight │  ← Same tensor as wte.weight!
│  [50257, 768]   │     (not a copy, same memory)
└─────────────────┘
       ↓
Output Logits
```

### Code Verification

```python
import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Check if weights are tied
wte = model.transformer.wte.weight
lm_head = model.lm_head.weight

print(f"wte shape: {wte.shape}")
print(f"lm_head shape: {lm_head.shape}")
print(f"Same tensor? {wte.data_ptr() == lm_head.data_ptr()}")
# Output: True - they point to the same memory!

# Modifying one modifies the other
original_value = wte[0, 0].item()
wte[0, 0] = 999.0
print(f"lm_head[0, 0] = {lm_head[0, 0].item()}")
# Output: 999.0 - they share memory!
```

---

## Why Does GPT-2 Use Tied Weights?

Weight tying was introduced in the paper:
> **"Using the Output Embedding to Improve Language Models"**
> Press & Wolf (2017)
> https://arxiv.org/abs/1608.05859

### Benefits

#### 1. **Reduced Parameter Count**

Without tying:
- Input embeddings: `vocab_size × hidden_size` = 50,257 × 768 = 38.6M params
- Output head: `vocab_size × hidden_size` = 50,257 × 768 = 38.6M params
- **Total:** 77.2M parameters

With tying:
- Shared embeddings: 50,257 × 768 = 38.6M params
- **Total:** 38.6M parameters
- **Savings:** 38.6M parameters (~33% of GPT-2-small's 124M total)

#### 2. **Theoretical Motivation**

The intuition is that input and output embeddings serve similar semantic purposes:

- **Input embedding:** "What does this word mean in context?"
- **Output projection:** "Which word best fits this context?"

Both map between vocabulary space and semantic space, just in opposite directions.

**Mathematical perspective:**
- Input: `token_id → embedding` (lookup)
- Output: `hidden_state → logits` (dot product with all embeddings)
- Using the same embeddings enforces consistency

#### 3. **Improved Generalization**

Weight tying acts as a regularizer:
- Forces the model to learn embeddings that work well in both directions
- Prevents overfitting by reducing model capacity
- Empirically shown to improve perplexity in language modeling

#### 4. **Memory Efficiency**

- Saves 38.6M parameters worth of GPU memory
- Crucial for training on resource-constrained hardware
- More important for larger vocabularies

---

## The Safetensors Issue

### What is Safetensors?

**Safetensors** is a modern format for storing neural network weights, developed by HuggingFace.

**Benefits over PyTorch `.bin` format:**
- ✅ Faster loading (no Python unpickling)
- ✅ Memory-mapped (doesn't load entire file at once)
- ✅ Safer (prevents arbitrary code execution)
- ✅ Cross-framework (Rust, Python, JavaScript)
- ✅ Smaller file size (no pickle overhead)

**GitHub:** https://github.com/huggingface/safetensors

### The Problem with Tied Weights

Safetensors has a design decision: **each tensor must have a unique storage**.

**Why?** To prevent ambiguity when loading:
- What if two tensors share memory but have different metadata?
- Which one should be loaded first?
- How do you ensure consistency?

**GPT-2 violates this assumption:**

```python
# When saving to safetensors
state_dict = model.state_dict()
# state_dict contains:
# {
#     'transformer.wte.weight': tensor(...),  # Points to memory location A
#     'lm_head.weight': tensor(...),          # Also points to memory location A!
# }

# Safetensors detects this and raises an error:
# RuntimeError: Some tensors share memory, this will lead to duplicate
# memory on disk and potential differences when loading them again
```

### Why This Matters

If safetensors saved both tensors separately:
1. **Duplicate memory:** The same 38.6M parameters would be stored twice
2. **Loading inconsistency:** Which copy should be used? What if they diverge?
3. **Wastes disk space:** File would be unnecessarily large

### The Error Message

```
RuntimeError:
    Some tensors share memory, this will lead to duplicate memory on
    disk and potential differences when loading them again:
    [{'model.lm_head.weight', 'model.transformer.wte.weight'}].

    A potential way to correctly save your model is to use `save_model`.
    More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
```

---

## How We Fixed It

### Solution 1: Use PyTorch Format (Our Approach)

We override the `_save()` method to use PyTorch's `.bin` format:

```python
# src/core/reward_modeling/trainer.py

def _save(self, output_dir: Optional[str] = None, state_dict=None):
    """Override save to handle tied weights in GPT-2 models."""
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = self.model

    if state_dict is None:
        state_dict = model_to_save.state_dict()

    # Save with torch.save instead of safetensors
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # Save config and tokenizer
    model_to_save.config.save_pretrained(output_dir)
    # ...
```

**Why this works:**
- PyTorch's pickle-based format handles tied weights correctly
- It saves metadata indicating which tensors share storage
- When loading, PyTorch reconstructs the sharing relationship

**Trade-offs:**
- ❌ Slower loading than safetensors
- ❌ Pickle security concerns (minor for our use case)
- ✅ Works perfectly with tied weights
- ✅ Standard PyTorch format

### Solution 2: Untie Weights Before Saving

```python
# Alternative approach (not used in our codebase)
def untie_weights_for_saving(model):
    """Create independent copies of tied weights."""
    # Make lm_head.weight independent
    model.lm_head.weight = nn.Parameter(
        model.transformer.wte.weight.clone()
    )
    return model

# Save with safetensors
model = untie_weights_for_saving(model)
model.save_pretrained(output_dir)  # Uses safetensors
```

**Why we don't use this:**
- ❌ Doubles memory usage during saving
- ❌ Increases saved model size (2× larger)
- ❌ Loses the semantic benefit of tied weights
- ❌ Requires re-tying when loading

### Solution 3: Save Only One Copy (Safetensors Workaround)

```python
# Another alternative (complex)
state_dict = model.state_dict()

# Remove one of the tied tensors
del state_dict['lm_head.weight']

# Save with safetensors
save_file(state_dict, "model.safetensors")

# Add metadata about tying
metadata = {"tied_weights": "lm_head.weight <- transformer.wte.weight"}
```

**Why we don't use this:**
- ❌ Requires custom loading logic
- ❌ Breaks compatibility with standard loaders
- ❌ Complex to maintain

---

## Modern LLMs: The Evolution

### Weight Tying Across Different Models

| Model Family | Input/Output Tied? | Vocab Size | Embedding Dim | Params Saved |
|--------------|-------------------|------------|---------------|--------------|
| **GPT-2** (2019) | ✅ Yes | 50,257 | 768-1600 | 38.6M-80M |
| **GPT-3** (2020) | ✅ Yes | 50,257 | 12,288 | 617M |
| **GPT-Neo/GPT-J** (2021) | ✅ Yes | 50,257 | 4,096 | 206M |
| **LLaMA** (2023) | ❌ No | 32,000 | 4,096 | 131M |
| **LLaMA 2** (2023) | ❌ No | 32,000 | 4,096-8,192 | 131M-262M |
| **Mistral** (2023) | ❌ No | 32,000 | 4,096 | 131M |
| **Gemma** (2024) | ❌ No | 256,000 | 2,048-3,072 | 524M-786M |
| **Qwen** (2023) | ❌ No | 151,936 | 4,096 | 622M |

### Why Modern Models Don't Tie Weights

#### 1. **Different Vocabularies**

Modern tokenizers are more efficient:

**GPT-2 (BPE):**
- Vocab size: 50,257
- Trained on English-heavy data
- Inefficient for non-English text

**LLaMA/Mistral (SentencePiece):**
- Vocab size: 32,000 (more efficient)
- Multilingual from the start
- Better compression per token

With smaller vocabularies, the parameter savings from tying are less significant:
- GPT-2: Saves 38.6M params (31% of 124M total)
- LLaMA-7B: Would save 131M params (1.9% of 7B total)

#### 2. **Scale Has Changed**

**GPT-2 era (2019):**
- Models: 124M-1.5B parameters
- Embedding savings: 30-50% of total
- Memory was a major constraint

**Modern era (2023+):**
- Models: 7B-70B+ parameters
- Embedding savings: <2% of total
- Memory is less of a constraint with better hardware

**The math:**
- GPT-2-small: 38.6M embeddings / 124M total = 31%
- LLaMA-7B: 131M embeddings / 7B total = 1.9%

At scale, the savings are negligible.

#### 3. **Decoupled Learning**

Modern models benefit from **independent input/output representations**:

**Untied advantages:**
- Input embeddings can specialize for context understanding
- Output head can specialize for generation
- More model capacity (better performance at scale)
- No constraint that one representation works well bidirectionally

**Research findings:**
- At small scale (<1B params): Tied weights help (regularization)
- At large scale (>7B params): Untied weights perform better
- Reference: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)

#### 4. **Hardware Has Improved**

**2019 (GPT-2):**
- GPUs: 16GB VRAM (V100)
- Every parameter counted
- Weight tying was essential

**2024 (Modern models):**
- GPUs: 80GB VRAM (A100), 192GB (H100)
- Multi-GPU training is standard
- Parameter efficiency less critical for embeddings

#### 5. **Architectural Innovations**

Modern models use other techniques instead:

**Grouped Query Attention (GQA):**
- Used in: LLaMA 2, Mistral
- Reduces KV cache memory
- More impactful than tying embeddings

**Flash Attention:**
- Memory-efficient attention
- Allows longer context windows
- Bigger win than tied weights

**4-bit/8-bit Quantization:**
- Reduces all parameters, not just embeddings
- More effective overall memory savings

### Notable Exceptions

Some modern models **still** use tied weights:

#### **PaLM** (Google, 2022)
- 540B parameters
- Still ties input/output embeddings
- Reason: Extreme scale makes even 1% savings meaningful

#### **Falcon** (2023)
- 7B-180B parameters
- Ties embeddings
- Reason: Optimized for inference efficiency

### Summary: The Trade-off Evolution

```
                    GPT-2 Era              Modern Era
                    (2019-2020)            (2023-2024)

Model Size          100M-1.5B              7B-70B
Memory/Param        Critical               Less critical
Vocab Efficiency    Lower (50K tokens)     Higher (32K tokens)
Embedding %         30-50% of params       <2% of params
Best Practice       Tie weights ✅         Untie weights ✅
Primary Reason      Parameter savings      Better performance
```

---

## Practical Implications

### For Our Codebase

#### When Using GPT-2 Models

✅ **Do:**
- Use PyTorch format for saving (handles tied weights)
- Be aware of the tying when debugging
- Expect smaller model files (no duplication)

❌ **Don't:**
- Try to use safetensors directly (will error)
- Modify wte.weight without expecting lm_head to change
- Assume input/output embeddings can be independently tuned

#### When Using Modern Models (LLaMA, Mistral, etc.)

✅ **Do:**
- Use safetensors format (faster, safer)
- Treat input/output embeddings as independent
- Can use untied-weight-specific optimizations

❌ **Don't:**
- Assume tied weights (they're not!)
- Try to share memory manually

### Code Compatibility

```python
def save_model_safely(model, output_dir):
    """Save model with automatic tied weight handling."""

    # Check if model has tied weights
    has_tied_weights = False
    if hasattr(model, 'tie_weights'):
        # Models with tie_weights() method usually have tied weights
        has_tied_weights = True

    # Alternative check: compare data pointers
    if hasattr(model, 'transformer') and hasattr(model, 'lm_head'):
        wte = model.transformer.wte.weight
        lm_head = model.lm_head.weight
        has_tied_weights = (wte.data_ptr() == lm_head.data_ptr())

    if has_tied_weights:
        # Use PyTorch format
        print("Tied weights detected, using PyTorch format")
        torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    else:
        # Use safetensors (faster, modern)
        print("No tied weights, using safetensors")
        model.save_pretrained(output_dir, safe_serialization=True)
```

### Loading Models

```python
def load_model_safely(model_class, path):
    """Load model regardless of format."""

    # Try safetensors first (modern format)
    if os.path.exists(f"{path}/model.safetensors"):
        return model_class.from_pretrained(path)

    # Fall back to PyTorch format (tied weights)
    elif os.path.exists(f"{path}/pytorch_model.bin"):
        return model_class.from_pretrained(path)

    else:
        raise FileNotFoundError(f"No model found at {path}")
```

---

## Debugging Tied Weights

### How to Detect Tied Weights

```python
import torch
from transformers import AutoModel

def check_tied_weights(model):
    """
    Check which weights are tied in a model.

    Returns:
        List of tied weight groups
    """
    param_map = {}  # data_ptr -> list of parameter names

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr not in param_map:
            param_map[ptr] = []
        param_map[ptr].append(name)

    # Find groups with more than one name (tied weights)
    tied_groups = [names for names in param_map.values() if len(names) > 1]

    return tied_groups

# Example usage
model = AutoModel.from_pretrained("gpt2")
tied = check_tied_weights(model)

for group in tied:
    print(f"Tied weights: {group}")
# Output: Tied weights: ['transformer.wte.weight', 'lm_head.weight']
```

### Visualizing Memory Layout

```python
def visualize_model_memory(model):
    """Show memory usage and sharing."""
    import pandas as pd

    params = []
    memory_ptrs = {}

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        size_mb = param.numel() * param.element_size() / (1024 * 1024)

        # Check if this memory is shared
        if ptr in memory_ptrs:
            shared_with = memory_ptrs[ptr]
            status = f"Shared with {shared_with}"
        else:
            memory_ptrs[ptr] = name
            status = "Unique"

        params.append({
            'name': name,
            'shape': tuple(param.shape),
            'params': param.numel(),
            'size_mb': size_mb,
            'status': status,
        })

    df = pd.DataFrame(params)

    print(f"\nTotal unique memory: {df[df['status'] == 'Unique']['size_mb'].sum():.2f} MB")
    print(f"Shared memory: {df[df['status'] != 'Unique']['size_mb'].sum():.2f} MB")

    return df

# Example
model = AutoModel.from_pretrained("gpt2")
df = visualize_model_memory(model)
print(df[df['status'] != 'Unique'])
```

### Common Issues and Solutions

#### Issue 1: Model Fails to Save

**Error:**
```
RuntimeError: Some tensors share memory
```

**Solution:**
```python
# Use PyTorch format instead of safetensors
torch.save(model.state_dict(), "pytorch_model.bin")
```

#### Issue 2: Unexpected Weight Changes

**Problem:**
```python
model.transformer.wte.weight[0, 0] = 999.0
# Now lm_head.weight[0, 0] is also 999.0!
```

**Solution:** Be aware that modifications propagate to tied weights.

#### Issue 3: Loading Tied Weights from Safetensors

**Problem:** Loaded model doesn't have tied weights (both tensors are independent)

**Solution:** Manually tie them after loading:
```python
model = AutoModel.from_pretrained(path)
# Re-tie the weights
model.lm_head.weight = model.transformer.wte.weight
```

---

## References

### Papers

1. **Weight Tying:**
   - "Using the Output Embedding to Improve Language Models"
   - Press & Wolf (2017)
   - https://arxiv.org/abs/1608.05859

2. **GPT-2:**
   - "Language Models are Unsupervised Multitask Learners"
   - Radford et al. (2019)
   - https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

3. **Scaling Laws:**
   - "Scaling Laws for Neural Language Models"
   - Kaplan et al. (2020)
   - https://arxiv.org/abs/2001.08361

4. **LLaMA:**
   - "LLaMA: Open and Efficient Foundation Language Models"
   - Touvron et al. (2023)
   - https://arxiv.org/abs/2302.13971

### Documentation

- **Safetensors Shared Tensors:** https://huggingface.co/docs/safetensors/torch_shared_tensors
- **Safetensors GitHub Issue:** https://github.com/huggingface/safetensors/issues/195
- **HuggingFace Tied Weights:** https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#transformers.GPT2LMHeadModel

### Code

- **Our Fix:** `src/core/reward_modeling/trainer.py` (line 315)
- **Tied Weight Check:** `src/utils/model_utils.py` (if we add it)

---

## Summary

### Quick Reference

**GPT-2:**
- ✅ Has tied weights (wte ↔ lm_head)
- ✅ Use PyTorch format for saving
- ✅ 31% parameter savings in small models
- ❌ Can't use safetensors directly

**Modern LLMs (LLaMA, Mistral, etc.):**
- ❌ No tied weights
- ✅ Can use safetensors
- ✅ Better performance at scale
- ✅ More model capacity

**The Evolution:**
```
2019 (GPT-2)          →  2024 (Modern LLMs)
├─ Small scale            ├─ Large scale
├─ Tied weights ✓         ├─ Untied weights ✓
├─ 50K vocab              ├─ 32K vocab
├─ Memory critical        ├─ Performance critical
└─ PyTorch format         └─ Safetensors format
```

**Bottom Line:**
- Weight tying was a good idea for GPT-2's scale
- Modern models have outgrown this optimization
- We handle both cases in our codebase
