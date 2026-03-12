# Direct Preference Optimization (DPO) - Theory and Intuition

This document explains DPO, a simpler and more stable alternative to traditional RLHF (PPO).

## Table of Contents
- [What is DPO?](#what-is-dpo)
- [The Problem with Traditional RLHF](#the-problem-with-traditional-rlhf)
- [DPO's Key Insight](#dpos-key-insight)
- [Mathematical Formulation](#mathematical-formulation)
- [DPO vs PPO: Comparison](#dpo-vs-ppo-comparison)
- [Training Process](#training-process)
- [Advantages and Limitations](#advantages-and-limitations)
- [When to Use DPO](#when-to-use-dpo)

---

## What is DPO?

**DPO (Direct Preference Optimization)** is a method for training language models to align with human preferences **without** needing a separate reward model or complex reinforcement learning.

**Paper:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
- arXiv: https://arxiv.org/abs/2305.18290

### The Big Idea

Instead of:
```
Step 1: Train reward model from preferences
Step 2: Use PPO to optimize policy with reward model
```

DPO does:
```
Single Step: Directly optimize policy from preferences
```

**Key Insight:** You can reformulate RLHF as a supervised learning problem!

---

## The Problem with Traditional RLHF

### Traditional RLHF Pipeline (PPO)

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: SFT                             │
│  Train base model on demonstrations                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Phase 2: Reward Modeling                      │
│  Train reward model on preference pairs                     │
│  Input: (prompt, chosen, rejected)                          │
│  Output: Scalar reward R(x, y)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Phase 3: PPO Training                         │
│  4 Models Required:                                         │
│  - Policy (being trained)                                   │
│  - Value function (critic)                                  │
│  - Reference policy (KL constraint)                         │
│  - Reward model (frozen)                                    │
│                                                             │
│  Complex RL Training Loop:                                  │
│  1. Generate samples with policy                            │
│  2. Score with reward model                                 │
│  3. Compute advantages (GAE)                                │
│  4. Update policy with clipped objective                    │
│  5. Update value function                                   │
└─────────────────────────────────────────────────────────────┘
```

### Problems with PPO:

1. **Complexity:** 4 models, complex training loop
2. **Instability:** RL can diverge, reward hacking
3. **Hyperparameter Sensitivity:** Learning rates, clip ratios, KL penalty weights
4. **Memory:** Need to load 4 models simultaneously
5. **Slow:** Many forward passes per update
6. **Reward Model Errors:** Inaccurate reward model → bad policy

**Question:** Can we avoid all this complexity?

---

## DPO's Key Insight

### The Breakthrough

DPO discovered that you can **directly optimize the policy** to satisfy the RLHF objective, without ever training a separate reward model or using RL!

### Mathematical Magic

**Traditional RLHF objective:**
```
Maximize: E[R(x, y)] - β * KL(π || π_ref)
```
Where:
- R(x, y) = reward model output
- π = policy being trained
- π_ref = reference policy (SFT model)
- β = KL penalty coefficient

**DPO insight:** This objective has a **closed-form optimal solution**!

```
π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(R(x,y) / β)
```

Where Z(x) is a normalizing constant.

**Rearranging:**
```
R(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)
```

**Key observation:** The optimal policy π* **implicitly defines** the reward function!

### Preference Data → Direct Optimization

**Bradley-Terry model for preferences:**
```
P(y_chosen > y_rejected | x) = σ(R(x, y_chosen) - R(x, y_rejected))
```

**Substitute the DPO reparameterization:**
```
P(y_chosen > y_rejected | x) = σ(
    β * log(π(y_chosen|x) / π_ref(y_chosen|x))
  - β * log(π(y_rejected|x) / π_ref(y_rejected|x))
)
```

**The Z(x) terms cancel!** We can now optimize directly!

---

## Mathematical Formulation

### DPO Loss Function

Given preference data: (x, y_chosen, y_rejected)

**DPO Loss:**
```
L_DPO = -log σ(β * log(π_θ(y_chosen|x) / π_ref(y_chosen|x))
              - β * log(π_θ(y_rejected|x) / π_ref(y_rejected|x)))
```

Simplified:
```
L_DPO = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
```

Where:
- log_ratio_chosen = log(π_θ(y_chosen|x)) - log(π_ref(y_chosen|x))
- log_ratio_rejected = log(π_θ(y_rejected|x)) - log(π_ref(y_rejected|x))

**Intuition:**
- Increase probability of chosen responses
- Decrease probability of rejected responses
- Stay close to reference model (implicit via log ratios)

### Gradient Interpretation

The gradient pushes the model to:
1. **Increase log-prob of chosen:** π_θ(y_chosen|x) ↑
2. **Decrease log-prob of rejected:** π_θ(y_rejected|x) ↓
3. **KL constraint (implicit):** Don't move too far from π_ref

The β parameter controls the trade-off:
- **Large β:** Stay close to reference (conservative)
- **Small β:** Aggressive optimization (risk overfitting)

---

## DPO vs PPO: Comparison

### Complexity

| Aspect | PPO (RLHF) | DPO |
|--------|------------|-----|
| **Phases** | 3 (SFT, Reward, RL) | 2 (SFT, DPO) |
| **Models** | 4 (Policy, Critic, Ref, Reward) | 2 (Policy, Ref) |
| **Training** | RL (complex) | Supervised (simple) |
| **Stability** | Can diverge | More stable |
| **Hyperparameters** | Many (clip, KL, GAE, etc.) | Few (β, learning rate) |
| **Memory** | High (4 models) | Medium (2 models) |
| **Speed** | Slow (rollouts) | Faster (direct) |

### Performance

**Empirical results (from DPO paper):**
- **Similar or better performance** than PPO
- **More stable training** (no divergence)
- **Simpler to tune** (fewer hyperparameters)
- **Faster convergence** (no reward model needed)

### Visual Comparison

**PPO Pipeline:**
```
Preference Data
    ↓
Train Reward Model (R)
    ↓
Generate with Policy (π)
    ↓
Score with R
    ↓
Compute Advantages
    ↓
Update π with RL (complex)
```

**DPO Pipeline:**
```
Preference Data
    ↓
Load Reference (π_ref)
    ↓
Directly optimize π with supervised loss
```

**DPO eliminates:** Reward modeling, RL complexity, rollout generation!

---

## Training Process

### Step-by-Step DPO Training

**Input:**
- SFT model (becomes reference π_ref)
- Preference dataset: (prompt, chosen, rejected) pairs

**Setup:**
1. Initialize policy π_θ from SFT model
2. Freeze reference model π_ref (copy of SFT)
3. Set β hyperparameter (typically 0.1 - 0.5)

**Training Loop:**

```python
for batch in preference_dataset:
    # 1. Forward pass through policy (trainable)
    logprobs_chosen_policy = π_θ(batch.chosen | batch.prompt)
    logprobs_rejected_policy = π_θ(batch.rejected | batch.prompt)

    # 2. Forward pass through reference (frozen)
    logprobs_chosen_ref = π_ref(batch.chosen | batch.prompt)
    logprobs_rejected_ref = π_ref(batch.rejected | batch.prompt)

    # 3. Compute log ratios
    log_ratio_chosen = logprobs_chosen_policy - logprobs_chosen_ref
    log_ratio_rejected = logprobs_rejected_policy - logprobs_rejected_ref

    # 4. DPO loss
    logits = β * (log_ratio_chosen - log_ratio_rejected)
    loss = -log_sigmoid(logits).mean()

    # 5. Backprop and update policy
    loss.backward()
    optimizer.step()
```

**That's it!** No reward model, no RL, just supervised learning.

### Key Implementation Details

**1. Reference Model:**
- Frozen copy of SFT model
- Provides KL constraint implicitly
- Can share parameters with policy (memory optimization)

**2. Log Probability Computation:**
```python
# Get token-level log probabilities
logprobs = log_softmax(logits, dim=-1)

# Sum over sequence length
sequence_logprob = logprobs.gather(dim=-1, index=labels).sum()
```

**3. Numerical Stability:**
- Use `log_sigmoid(x)` not `log(sigmoid(x))`
- Clip very large/small values
- Use mixed precision (fp16/bf16)

---

## Advantages and Limitations

### Advantages ✅

1. **Simplicity:**
   - No reward model training
   - No RL complexity
   - Standard supervised learning

2. **Stability:**
   - No divergence issues
   - No reward hacking
   - Predictable training

3. **Efficiency:**
   - Fewer models (2 vs 4)
   - Less memory
   - Faster training

4. **Performance:**
   - Matches or exceeds PPO
   - Better sample efficiency
   - More consistent results

5. **Easier to Debug:**
   - Standard ML training
   - No complex RL bugs
   - Clear loss signal

6. **Mobile-Friendly:**
   - Lower memory footprint
   - Simpler deployment
   - Works on-device

### Limitations ⚠️

1. **Static Preferences:**
   - Uses fixed preference dataset
   - Can't adapt to new feedback easily
   - PPO can update reward model on-the-fly

2. **Reference Model Dependency:**
   - Quality depends on SFT model
   - Can't fix bad SFT with DPO alone

3. **Preference Data Quality:**
   - Sensitive to noisy preferences
   - Needs high-quality pairs
   - More data = better results

4. **Limited Exploration:**
   - Only learns from provided pairs
   - Doesn't explore alternative strategies
   - PPO can discover new behaviors

5. **Extrapolation:**
   - May not generalize beyond preference data distribution
   - PPO's reward model can guide on new inputs

### When Limitations Matter

**Use PPO if:**
- You have a trained reward model
- Need online learning (adapt during training)
- Want to explore beyond training distribution
- Have infrastructure for RL

**Use DPO if:**
- Starting from scratch
- Want simpler training
- Have good preference data
- Limited compute/memory
- Prioritize stability

---

## When to Use DPO

### DPO is Great For:

✅ **Most practical applications:**
- Chatbots and assistants
- Content generation
- Code generation
- Summarization

✅ **Resource-constrained settings:**
- Limited GPU memory
- Mobile/edge deployment
- Academic research

✅ **First-time RLHF:**
- Learning alignment techniques
- Proof of concept
- Quick experiments

✅ **Stable production systems:**
- Predictable behavior
- Easy to maintain
- Fewer failure modes

### Use PPO (RLHF) Instead If:

⚠️ **Need online learning:**
- Reward model adapts during training
- Interactive learning scenarios

⚠️ **Have specific reward function:**
- Safety constraints
- Task-specific metrics
- Complex reward shaping

⚠️ **Exploration important:**
- Discovering novel strategies
- Open-ended tasks
- Creative applications

---

## DPO Variants

### Recent Improvements

**1. IPO (Identity Preference Optimization):**
- Uses squared loss instead of log-sigmoid
- More robust to noise
- Less sensitive to outliers

**2. KTO (Kahneman-Tversky Optimization):**
- Uses only positive or negative feedback (no pairs)
- More flexible data collection

**3. ORPO (Odds Ratio Preference Optimization):**
- Combines SFT and preference learning
- Single-stage training

**4. SimPO (Simple Preference Optimization):**
- Removes reference model requirement
- Even simpler!

### Evolution

```
PPO (2017) → DPO (2023) → IPO/KTO/ORPO (2024)

Trend: Simpler, more stable, more practical
```

---

## Example: Training a Helpful Assistant

### Preference Data

```json
{
  "prompt": "How do I make coffee?",
  "chosen": "To make coffee: 1) Add water to machine, 2) Add coffee grounds, 3) Press brew. Enjoy!",
  "rejected": "You make coffee with a coffee maker."
}
```

### Traditional RLHF (PPO):

1. **Train Reward Model:**
   ```
   R("helpful response") = 2.5
   R("unhelpful response") = -1.2
   ```

2. **Use PPO:**
   - Generate: "Coffee is made by..."
   - Score: R = 0.8
   - Compute advantage: A = 0.8 - baseline
   - Update policy to increase P(high-reward responses)

3. **Repeat for thousands of iterations**

### DPO:

1. **Directly optimize:**
   ```python
   loss = -log σ(
       log(π("helpful"|prompt) / π_ref("helpful"|prompt))
     - log(π("unhelpful"|prompt) / π_ref("unhelpful"|prompt))
   )
   ```

2. **Update model to:**
   - Increase P("helpful response" | prompt)
   - Decrease P("unhelpful response" | prompt)
   - Stay close to reference

3. **Done in one pass through data!**

---

## Hyperparameters

### Key Hyperparameters for DPO

**1. β (Beta) - Most Important**
- **Range:** 0.1 - 0.5
- **Effect:** Controls KL divergence from reference
- **Small β (0.1):** Aggressive optimization, risk overfitting
- **Large β (0.5):** Conservative, stays close to reference
- **Default:** 0.1 - 0.2 works well

**2. Learning Rate**
- **Range:** 1e-6 to 5e-6
- **Lower than SFT** (fine-tuning SFT model)
- **Default:** 5e-7 to 1e-6

**3. Epochs**
- **Range:** 1-3
- **Less than SFT** (can overfit to preferences)
- **Default:** 1-2 epochs

**4. Batch Size**
- **Larger is better** (more stable gradients)
- **Default:** 16-32 (or higher if memory allows)

### Typical Config

```yaml
dpo:
  beta: 0.1
  learning_rate: 5e-7
  num_epochs: 1
  batch_size: 32
  warmup_ratio: 0.1
  max_length: 512
```

---

## Summary: The DPO Revolution

### Before DPO (2023)

**RLHF was:**
- Complex (3 phases, 4 models)
- Unstable (RL divergence)
- Slow (rollouts + RL)
- Hard to tune (many hyperparameters)

### After DPO (2023+)

**Preference learning is:**
- ✅ Simple (1 phase, 2 models)
- ✅ Stable (supervised learning)
- ✅ Fast (direct optimization)
- ✅ Easy to tune (few hyperparameters)

### Impact

**Industry shift:**
- Many companies moved from PPO → DPO
- Easier to implement and maintain
- Better results with less effort
- Democratized RLHF for smaller teams

**Research impact:**
- Inspired many variants (IPO, KTO, ORPO)
- New theoretical understanding
- Simpler baselines for comparison

### The Future

DPO shows that **complex RL may not be necessary** for aligning LLMs. The trend is toward:
- Simpler methods
- More stable training
- Better theoretical foundations
- Practical deployment

**DPO is now the recommended starting point for preference learning!**

---

## References

**Original Paper:**
- Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- https://arxiv.org/abs/2305.18290

**Related Work:**
- IPO: https://arxiv.org/abs/2310.12036
- KTO: https://arxiv.org/abs/2402.01306
- ORPO: https://arxiv.org/abs/2403.07691

**Implementations:**
- HuggingFace TRL: https://github.com/huggingface/trl
- Our implementation: `src/core/dpo/`

**Further Reading:**
- RLHF survey: https://arxiv.org/abs/2203.02155
- Alignment research: https://www.anthropic.com/research

---

## Implementation Approach: Why HuggingFace Trainer?

### Design Decision: Pragmatic vs Pure Educational

Our DPO implementation makes a **deliberate trade-off** between pragmatism and educational transparency.

#### What We Use: HuggingFace Trainer

```python
class DPOTrainer(Trainer):
    """Extends HuggingFace Trainer for DPO."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom DPO loss computation - this is where DPO happens!"""
        # Forward through policy and reference models
        # Compute log probabilities
        # Calculate DPO loss
        return loss
```

**What Trainer handles for us:**
- Training loop (epochs, batches, gradient accumulation)
- Device management (CPU/GPU/multi-GPU)
- Mixed precision (fp16/bf16)
- Checkpointing and resuming
- Logging (tensorboard, wandb)
- Evaluation and metrics
- Learning rate scheduling
- Gradient clipping

**What we implement (the interesting part):**
- DPO loss computation (`compute_loss`)
- Log probability calculation
- Implicit reward tracking
- KL divergence monitoring

### Why This Approach?

#### Pros ✅
1. **Focus on What Matters**: DPO is essentially supervised learning with a custom loss
2. **Less Boilerplate**: ~400 lines vs ~800 lines for custom loop
3. **Battle-Tested**: Trainer handles edge cases we'd otherwise have to debug
4. **Practical**: Users can actually train models without debugging infrastructure
5. **Still Educational**: The **core algorithm** (DPO loss) is fully implemented and visible

#### Cons ❌
1. **Hidden Training Loop**: Don't see the actual `for batch in dataloader` loop
2. **Abstraction Leaks**: Sometimes need to work around Trainer assumptions
3. **Less Transparent**: Harder to see exactly when forward/backward happens

### What Does the Hidden Loop Look Like?

If we wrote it manually, the training loop would be:

```python
def train_dpo_manual(
    policy_model,
    reference_model,
    train_dataset,
    num_epochs=1,
    learning_rate=5e-7,
):
    """Custom DPO training loop (educational)."""
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_dataset, batch_size=2)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 1. Forward pass through policy
            policy_chosen_logps = compute_log_probs(
                policy_model, batch['chosen_input_ids']
            )
            policy_rejected_logps = compute_log_probs(
                policy_model, batch['rejected_input_ids']
            )
            
            # 2. Forward pass through reference (no grad)
            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(
                    reference_model, batch['chosen_input_ids']
                )
                ref_rejected_logps = compute_log_probs(
                    reference_model, batch['rejected_input_ids']
                )
            
            # 3. Compute DPO loss
            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=0.1,
            )
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # 6. Logging (if needed)
            if step % logging_steps == 0:
                print(f"Loss: {loss.item():.4f}")
```

**That's it!** DPO training is just supervised learning. The magic is in the loss function, not the training loop.

### Why Not Custom Loop for DPO?

**DPO is fundamentally supervised learning:**
- Standard training loop (forward → loss → backward → step)
- No rollouts, no advantages, no policy/value updates
- The innovation is the **loss function**, not the training procedure

**Compare to PPO (Phase 5)**, which **requires** a custom loop:
- Two-phase algorithm: rollout then update
- Generate samples with current policy
- Compute advantages from value function
- Multiple update epochs on same batch
- Clipped policy objective
- Separate value function updates

**Our strategy:**
- **DPO (Phase 4)**: Use Trainer, focus on loss function (pragmatic)
- **PPO (Phase 5)**: Full custom loop, expose all RL details (educational)
- **Result**: Users see both approaches, understand the difference

### Where to Find the Interesting Code

Even with Trainer, the **algorithmic core** is fully visible:

1. **Loss Functions** (`src/core/dpo/loss.py`):
   - `dpo_loss()`: The core DPO algorithm (lines 68-158)
   - `compute_sequence_log_probs()`: Log probability computation (lines 17-65)
   - `ipo_loss()`: IPO variant (lines 254-313)

2. **Trainer** (`src/core/dpo/trainer.py`):
   - `compute_loss()`: Where DPO happens (lines 118-243)
   - Forward passes through policy and reference
   - Log probability calculations
   - Loss computation and metric tracking

3. **Training Script** (`scripts/train/train_dpo.py`):
   - Data loading and preprocessing
   - Model initialization (policy + reference)
   - Trainer setup and configuration
   - Evaluation and result reporting

### Educational Value Preserved

**You still learn:**
- ✅ How DPO loss is computed mathematically
- ✅ Why we need policy and reference models
- ✅ How log probabilities are calculated
- ✅ What implicit rewards mean
- ✅ How KL divergence is tracked
- ✅ Difference between DPO and IPO loss

**You don't see:**
- ❌ The literal `for batch in dataloader` loop
- ❌ Exact timing of forward/backward passes
- ❌ How gradient accumulation works internally

**Trade-off accepted** because:
- Training loop is standard PyTorch (not DPO-specific)
- Users can learn that from any PyTorch tutorial
- The **DPO algorithm** is what matters, and that's fully implemented

### Alternative: Custom Loop Available

If you want to see a manual training loop, check out:
- **Phase 5 (PPO)**: Full custom RL training loop (coming soon)
- **Notebooks**: Interactive examples with explicit loops
- **Simple example**: `examples/minimal_dpo.py` (if added)

### For Researchers and Advanced Users

If you need **full control** and want to write your own loop:

```python
from src.core.dpo.loss import dpo_loss, compute_sequence_log_probs

# Your custom training loop here
# Use our loss functions but write your own loop
# Useful for: research, custom sampling, non-standard training
```

Our loss functions are standalone and don't depend on Trainer!

---

## Summary: Implementation Philosophy

**Core Principle**: Focus educational effort where it matters most.

**For DPO:**
- ✅ Custom loss function (the innovation)
- ✅ Clear documentation and theory
- ⚖️ Standard training loop (via Trainer)

**For PPO (Phase 5):**
- ✅ Custom loss function
- ✅ Custom training loop (essential to understand RL)
- ✅ Clear documentation and theory

**Result**: 
- Users learn the **algorithms** (DPO, PPO)
- Users get **working code** they can actually use
- Users see different implementation approaches
- Balance between education and practicality

This is an **educational repository that produces usable models**, not a pure teaching tool or a pure production library. The Trainer decision reflects that balance.

