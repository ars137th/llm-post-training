# Proximal Policy Optimization (PPO) for RLHF - Theory and Implementation

## Table of Contents
1. [Overview](#overview)
2. [RLHF Pipeline](#rlhf-pipeline)
3. [The Four Models](#the-four-models)
4. [PPO Algorithm](#ppo-algorithm)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Generalized Advantage Estimation (GAE)](#generalized-advantage-estimation-gae)
7. [Training Loop](#training-loop)
8. [PPO vs DPO](#ppo-vs-dpo)
9. [Implementation Approach](#implementation-approach)
10. [Hyperparameters](#hyperparameters)
11. [Common Issues](#common-issues)

---

## Overview

**PPO (Proximal Policy Optimization)** is a policy gradient reinforcement learning algorithm used in RLHF (Reinforcement Learning from Human Feedback) to fine-tune language models based on human preferences.

### What is RLHF?

RLHF is a three-stage process:
1. **Stage 1 - SFT**: Supervised fine-tuning on instructions
2. **Stage 2 - Reward Modeling**: Train a model to predict human preferences
3. **Stage 3 - RL Fine-tuning**: Use PPO to optimize policy based on reward model

### Why PPO?

- **Safe updates**: Prevents catastrophic policy changes via clipping
- **Sample efficient**: Reuses data for multiple gradient updates
- **Stable**: More stable than vanilla policy gradient (REINFORCE)
- **Effective**: State-of-the-art for RLHF (used by InstructGPT, ChatGPT)

### Key Insight

PPO optimizes a language model to maximize rewards from a reward model while staying close to a reference policy (KL constraint). This balance ensures:
- Model learns to generate preferred responses (high reward)
- Model doesn't drift too far from original behavior (prevents mode collapse)

---

## RLHF Pipeline

```
┌─────────────┐
│   SFT Model │  (Stage 1: Supervised fine-tuning)
│   (Phase 2) │
└──────┬──────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌─────────────┐  ┌──────────────┐
│Reward Model │  │Reference Model│
│  (Stage 2)  │  │   (Frozen)    │
│  (Phase 3)  │  └───────────────┘
└─────────────┘         │
       │                │
       └────────┬───────┘
                ▼
        ┌───────────────┐
        │  PPO Training │  (Stage 3: RL fine-tuning)
        │   (Phase 5)   │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ Aligned Model │
        └───────────────┘
```

**Workflow**:
1. Train SFT model on instructions → becomes reference policy
2. Train reward model on preferences using SFT as base
3. Copy SFT model → becomes trainable actor (policy)
4. Train actor with PPO using reward model feedback
5. Reference policy provides KL constraint during training

---

## The Four Models

PPO requires managing four models simultaneously:

### 1. Actor (Policy Model) - Trainable
- **Role**: The model being optimized
- **Input**: Prompts
- **Output**: Generated responses
- **Training**: Updated via PPO to maximize reward
- **Initialization**: Copy of SFT model

### 2. Critic (Value Function) - Trainable
- **Role**: Estimates expected future rewards
- **Input**: Prompts (or prompt + partial response)
- **Output**: Scalar value estimate
- **Training**: Updated via value loss (MSE with actual returns)
- **Architecture**: Usually same as actor + value head

### 3. Reference Model - Frozen
- **Role**: Provides KL penalty to prevent drift
- **Input**: Prompts + generated responses
- **Output**: Log probabilities
- **Training**: Never updated (frozen)
- **Initialization**: Copy of SFT model (same as initial actor)

### 4. Reward Model - Frozen
- **Role**: Scores quality of generated responses
- **Input**: Prompts + generated responses
- **Output**: Scalar reward
- **Training**: Never updated (pre-trained in Stage 2)
- **Source**: From Phase 3 (reward modeling)

### Model Interactions

```
Prompt → Actor → Response
              │
              ├──→ Reward Model → Reward score
              │
              ├──→ Reference Model → KL penalty
              │
              └──→ Critic → Value estimate

Combined: Total reward = Reward score - β * KL penalty
```

---

## PPO Algorithm

### High-Level Overview

PPO alternates between two phases:

**Phase 1 - Rollout (Data Collection)**:
1. Generate responses using current policy
2. Score responses with reward model
3. Compute KL penalty vs reference
4. Estimate values with critic
5. Calculate advantages (GAE)

**Phase 2 - Update (Policy Optimization)**:
1. Compute PPO loss (clipped objective)
2. Update actor (policy)
3. Compute value loss
4. Update critic (value function)
5. Repeat for multiple epochs on same batch

### Why Clipping?

Without clipping, large policy updates can cause:
- **Mode collapse**: Model learns to exploit reward model
- **Instability**: Loss of diversity in generations
- **Catastrophic forgetting**: Model forgets SFT behavior

PPO clips the policy ratio to prevent updates that change the policy too much.

---

## Mathematical Foundations

### 1. Policy Gradient Objective

Standard policy gradient maximizes expected reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

Where:
- $\theta$ = policy parameters
- $\tau$ = trajectory (prompt + response)
- $R(\tau)$ = total reward

### 2. Importance Sampling Ratio

PPO uses old trajectories with new policy:

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

Where:
- $r_t(\theta)$ = probability ratio (new policy / old policy)
- $a_t$ = action (token) at step t
- $s_t$ = state (prompt + tokens so far)

### 3. Clipped Surrogate Objective (Core of PPO)

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]
$$

Where:
- $A_t$ = advantage at step t (how much better than expected)
- $\epsilon$ = clip range (typically 0.2)
- $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ = clamp ratio to [0.8, 1.2]

**Intuition**:
- If $A_t > 0$ (good action): Allow ratio up to $1 + \epsilon$ (encourage)
- If $A_t < 0$ (bad action): Allow ratio down to $1 - \epsilon$ (discourage)
- Beyond clip range: gradient is zero (no change)

### 4. Value Function Loss

Critic is trained to predict returns:

$$
L^{VF}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - R_t)^2 \right]
$$

Where:
- $\phi$ = critic parameters
- $V_\phi(s_t)$ = predicted value
- $R_t$ = actual return (sum of future rewards)

### 5. Entropy Bonus

Encourages exploration:

$$
L^{ENT}(\theta) = -\mathbb{E}_t \left[ H(\pi_\theta(\cdot | s_t)) \right]
$$

Where $H$ = entropy of policy distribution

### 6. Total Loss

$$
L(\theta, \phi) = L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 L^{ENT}(\theta)
$$

Where:
- $c_1$ = value loss coefficient (typically 0.5)
- $c_2$ = entropy coefficient (typically 0.01)

### 7. RLHF-Specific Reward

For language models:

$$
R(x, y) = R_{RM}(x, y) - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})
$$

Where:
- $R_{RM}(x, y)$ = reward model score for prompt $x$ and response $y$
- $\beta$ = KL penalty coefficient (typically 0.01-0.1)
- $D_{KL}$ = KL divergence between current and reference policy

**KL Divergence** (per-token):

$$
D_{KL} = \sum_t \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{ref}(y_t | x, y_{<t})}
$$

---

## Generalized Advantage Estimation (GAE)

### What is Advantage?

Advantage measures "how much better is action $a$ compared to average":

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

Where:
- $Q(s_t, a_t)$ = expected return from taking action $a_t$
- $V(s_t)$ = expected return from state $s_t$ (any action)

**Intuition**:
- $A > 0$: This action is better than average → increase probability
- $A < 0$: This action is worse than average → decrease probability

### Temporal Difference (TD) Error

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Where:
- $r_t$ = reward at step t
- $\gamma$ = discount factor (typically 0.99)
- TD error estimates advantage but has high variance

### GAE Formula

GAE trades off bias vs variance:

$$
A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Where:
- $\lambda$ = GAE lambda (typically 0.95)
- Higher $\lambda$ = lower bias, higher variance
- Lower $\lambda$ = higher bias, lower variance

**Practical computation** (recursive):

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

With $A_T = 0$ (terminal state).

### Why GAE?

- **Variance reduction**: Exponential weighting reduces noise
- **Bias-variance trade-off**: $\lambda$ parameter tunes the balance
- **Sample efficiency**: Better advantage estimates → fewer samples needed

---

## Training Loop

### Outer Loop (PPO Iterations)

```python
for iteration in range(num_iterations):
    # Phase 1: Rollout
    trajectories = rollout(
        actor=actor,
        prompts=sample_prompts(),
        num_samples=ppo_batch_size,
    )

    # Score with reward model and compute KL
    rewards = compute_rewards(
        trajectories=trajectories,
        reward_model=reward_model,
        reference_model=reference_model,
        actor=actor,
        kl_coef=kl_coef,
    )

    # Estimate values
    values = critic(trajectories.states)

    # Compute advantages
    advantages = compute_gae(
        rewards=rewards,
        values=values,
        gamma=gamma,
        lam=lam,
    )

    # Phase 2: Update (multiple epochs on same data)
    for epoch in range(ppo_epochs):
        for batch in trajectories.shuffle().batch(mini_batch_size):
            # Compute PPO loss
            policy_loss = compute_ppo_loss(
                actor=actor,
                batch=batch,
                advantages=advantages,
                clip_range=clip_range,
            )

            # Compute value loss
            value_loss = compute_value_loss(
                critic=critic,
                batch=batch,
                returns=advantages + values,  # advantage + baseline
            )

            # Update both models
            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            total_loss.backward()
            optimizer.step()
```

### Hyperparameters in Loop

**Rollout**:
- `ppo_batch_size`: Number of trajectories per iteration (e.g., 128)
- `max_length`: Max tokens per response (e.g., 256)

**Reward**:
- `kl_coef` (β): KL penalty coefficient (e.g., 0.05)

**GAE**:
- `gamma` (γ): Discount factor (e.g., 0.99)
- `lam` (λ): GAE lambda (e.g., 0.95)

**Update**:
- `ppo_epochs`: Update epochs per rollout (e.g., 4)
- `mini_batch_size`: Batch size for updates (e.g., 32)
- `clip_range` (ε): PPO clip parameter (e.g., 0.2)
- `vf_coef` (c₁): Value loss coefficient (e.g., 0.5)
- `ent_coef` (c₂): Entropy coefficient (e.g., 0.01)

---

## PPO vs DPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| **Complexity** | High (4 models, RL loop) | Low (2 models, supervised) |
| **Training** | Two-phase (rollout + update) | Single-phase (supervised) |
| **Stability** | Can be unstable | Very stable |
| **Sample Efficiency** | Lower (needs on-policy data) | Higher (uses fixed dataset) |
| **Reward Signal** | Explicit reward model | Implicit from preferences |
| **KL Constraint** | Explicit in reward | Implicit in loss |
| **Flexibility** | Can use any reward function | Limited to preference data |
| **Online Learning** | Yes (generates new data) | No (fixed dataset) |
| **Mode Collapse Risk** | Higher | Lower |
| **Hyperparameters** | Many (10+) | Few (2-3) |
| **Training Time** | 10-100x longer | Faster |
| **Final Performance** | Often slightly better | Competitive |
| **Use Case** | When you need RL (multi-turn, long-horizon) | When you have preferences (single-turn) |

### When to Use PPO vs DPO?

**Use PPO when**:
- You have a trained reward model
- You need online learning (model improves during training)
- Task has long-horizon dependencies
- Multi-turn conversations
- You can afford compute (10-100x more than DPO)
- You want state-of-the-art results (willing to tune extensively)

**Use DPO when**:
- You have preference data (no need for reward model)
- You want simple, stable training
- Single-turn tasks
- Limited compute budget
- You want to iterate quickly
- Good-enough performance is acceptable

**Key Insight**: DPO is often preferred in practice due to simplicity and stability. Use PPO only when you specifically need RL dynamics or have a high-quality reward model.

---

## Implementation Approach

### Why Custom Training Loop?

Unlike DPO (which used HuggingFace Trainer), PPO requires a **custom training loop**:

**Reasons**:
1. **Two-phase algorithm**: Rollout and update are fundamentally different
2. **On-policy data**: Must generate new data each iteration
3. **Four models**: Trainer doesn't support this architecture
4. **Mini-batch updates**: Multiple epochs on same rollout data
5. **Educational value**: Understanding RL requires seeing the loop

### Architecture Overview

```
src/core/ppo/
├── loss.py          # PPO loss, value loss, entropy
├── buffer.py        # Rollout buffer (stores trajectories)
├── gae.py           # Generalized Advantage Estimation
├── generation.py    # Response generation utilities
└── trainer.py       # PPOTrainer (custom loop, not HF Trainer)
```

### Key Components

**1. Rollout Buffer** (`buffer.py`):
```python
class RolloutBuffer:
    """Stores trajectories from rollout phase."""
    def __init__(self):
        self.prompts = []
        self.responses = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []

    def compute_returns_and_advantages(self, gamma, lam):
        """Compute GAE advantages."""
        # ... GAE implementation
```

**2. PPO Loss** (`loss.py`):
```python
def ppo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_range: float = 0.2,
) -> Tensor:
    """
    Compute clipped PPO loss.

    L^CLIP = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
    """
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages

    loss = -torch.min(loss_unclipped, loss_clipped).mean()
    return loss
```

**3. PPO Trainer** (`trainer.py`):
```python
class PPOTrainer:
    """Custom trainer for PPO with rollout-update loop."""

    def __init__(
        self,
        actor: PreTrainedModel,
        critic: PreTrainedModel,
        reference: PreTrainedModel,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: PPOConfig,
    ):
        self.actor = actor
        self.critic = critic
        self.reference = reference
        self.reward_model = reward_model
        # ... setup optimizers, buffers

    def train(self, prompts: List[str]) -> Dict:
        """Main training loop."""
        for iteration in range(self.config.num_iterations):
            # Rollout phase
            rollout_data = self.rollout(prompts)

            # Update phase
            for epoch in range(self.config.ppo_epochs):
                metrics = self.update(rollout_data)

            self.log(metrics)
```

---

## Hyperparameters

### Critical Hyperparameters

| Parameter | Symbol | Typical Value | Effect |
|-----------|--------|---------------|--------|
| **KL Coefficient** | β | 0.01 - 0.1 | Higher = stay closer to reference |
| **Clip Range** | ε | 0.1 - 0.3 | Higher = larger policy updates allowed |
| **GAE Lambda** | λ | 0.9 - 0.98 | Higher = lower bias, higher variance |
| **Discount Factor** | γ | 0.99 - 1.0 | Higher = care more about future |
| **Learning Rate** | α | 1e-6 - 1e-5 | Lower than SFT (5-10x) |
| **PPO Epochs** | - | 2 - 8 | More = better data use, slower |
| **Batch Size** | - | 32 - 256 | Larger = more stable, more memory |
| **Value Loss Coef** | c₁ | 0.5 - 2.0 | Weight of value loss |
| **Entropy Coef** | c₂ | 0.0 - 0.01 | Higher = more exploration |

### Tuning Guide

**If model diverges** (KL explodes):
- ↓ Decrease learning rate (try 5e-7)
- ↑ Increase KL coefficient (try 0.1)
- ↓ Decrease clip range (try 0.1)
- ↓ Decrease PPO epochs (try 2)

**If model doesn't improve** (flat reward):
- ↑ Increase learning rate (try 5e-6)
- ↓ Decrease KL coefficient (try 0.01)
- ↑ Increase clip range (try 0.3)
- ↑ Increase PPO epochs (try 6-8)

**If training is unstable** (high variance):
- ↑ Increase batch size
- ↓ Decrease GAE lambda (try 0.90)
- ↑ Increase gradient clip norm
- Use learning rate warmup

---

## Common Issues

### 1. KL Divergence Explodes

**Symptoms**: KL > 10, model generates gibberish

**Causes**:
- Learning rate too high
- KL coefficient too low
- Reward model gives extreme scores

**Fixes**:
- Lower learning rate (try 1e-6)
- Increase KL coefficient (try 0.1)
- Add adaptive KL (adjust β based on KL)
- Clip rewards to reasonable range

### 2. Reward Hacking

**Symptoms**: High reward, but responses are low-quality or repetitive

**Causes**:
- Reward model has exploitable patterns
- KL penalty too weak
- Model finds adversarial examples

**Fixes**:
- Increase KL coefficient
- Add length penalty (prevent long gibberish)
- Add repetition penalty
- Improve reward model training

### 3. Mode Collapse

**Symptoms**: Model always generates same response

**Causes**:
- Entropy too low
- Overoptimization (too many updates)
- KL penalty too strong

**Fixes**:
- Increase entropy coefficient (try 0.01)
- Reduce PPO epochs (try 2-4)
- Decrease KL coefficient
- Add temperature sampling

### 4. Training is Too Slow

**Symptoms**: Takes days to see improvement

**Causes**:
- Batch size too small
- Generation is slow
- Four models don't fit in memory

**Fixes**:
- Use gradient accumulation
- Enable KV caching for generation
- Use mixed precision (fp16/bf16)
- Offload reference and reward models to CPU
- Use smaller models for critic

### 5. Critic Value Estimates Are Poor

**Symptoms**: Advantages have high variance, unstable training

**Causes**:
- Value loss coefficient too low
- Critic underfitting
- Not enough value updates

**Fixes**:
- Increase value loss coefficient (try 1.0-2.0)
- Train critic for more steps before PPO
- Use larger critic learning rate
- Add value clipping

---

## Practical Tips

### Memory Optimization

**Four models in memory is expensive!**

Strategies:
1. **Quantize reference and reward models** (8-bit or 4-bit)
2. **Offload to CPU** when not in use
3. **Share base model** between actor and critic (add value head)
4. **Use LoRA** for actor and critic (freeze base layers)
5. **Gradient checkpointing** for actor

### Training Speed

**PPO is 10-100x slower than DPO!**

Strategies:
1. **Start with small prompts** (test on 100-1000 before full dataset)
2. **Use smaller max_length** during development (64-128 tokens)
3. **Profile generation** (this is usually the bottleneck)
4. **Use bf16** (bfloat16) if available
5. **Parallelize generation** across multiple GPUs

### Debugging

**How to know if it's working?**

Monitor these metrics:
- **Reward**: Should increase over time
- **KL**: Should stay < 5 (ideally < 1)
- **Value loss**: Should decrease
- **Approx KL**: Should be small (< 0.01 per token)
- **Clip fraction**: Should be 10-30% (not 0%, not 100%)
- **Entropy**: Should decrease slowly (not collapse)

**Red flags**:
- Reward increases but quality decreases → reward hacking
- KL > 10 → divergence
- Clip fraction = 100% → updates too large
- Entropy → 0 → mode collapse

---

## Summary

### Key Concepts

1. **Four Models**: Actor (train), Critic (train), Reference (frozen), Reward (frozen)
2. **Two Phases**: Rollout (generate data) + Update (optimize policy)
3. **Clipped Objective**: Prevents large policy changes
4. **GAE**: Estimates advantages with bias-variance trade-off
5. **KL Penalty**: Keeps policy close to reference

### PPO in One Sentence

**PPO optimizes a language model to maximize reward from a reward model while staying close to a reference policy, using a clipped objective to ensure stable updates.**

### Implementation Checklist

- [ ] Rollout buffer for storing trajectories
- [ ] GAE implementation for advantage estimation
- [ ] PPO loss with clipping
- [ ] Value loss for critic
- [ ] Custom training loop (rollout + update)
- [ ] Four model management (memory, devices)
- [ ] Extensive logging (reward, KL, value, clip fraction)
- [ ] Adaptive KL (optional but recommended)

### When You're Done

After implementing PPO, you'll understand:
- **How RL works**: On-policy learning, policy gradients
- **Why it's hard**: Stability, sample efficiency, hyperparameter sensitivity
- **Why DPO exists**: Simpler alternative for preference learning
- **Trade-offs**: Complexity vs performance vs stability

PPO is the most complex algorithm in this repository, but it's also the most educational. Understanding PPO gives you deep insight into RL and RLHF!

---

## Next Steps

1. **Implement core components**:
   - `src/core/ppo/loss.py` - PPO loss, value loss
   - `src/core/ppo/buffer.py` - Rollout buffer
   - `src/core/ppo/gae.py` - Advantage estimation

2. **Implement trainer**:
   - `src/core/ppo/trainer.py` - Custom training loop

3. **Create training script**:
   - `scripts/train/train_ppo.py` - Full pipeline

4. **Test thoroughly**:
   - Start with tiny model + synthetic data
   - Monitor all metrics closely
   - Tune hyperparameters iteratively

5. **Compare with DPO**:
   - Train same model with both methods
   - Compare final quality and training cost
   - Document findings

Good luck! PPO is challenging but incredibly rewarding to implement from scratch. 🚀
