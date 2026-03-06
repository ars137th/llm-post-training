# Reward Modeling Theory & Examples

## Table of Contents
1. [What is Reward Modeling?](#what-is-reward-modeling)
2. [Why Do We Need Reward Models?](#why-do-we-need-reward-models)
3. [The Bradley-Terry Model](#the-bradley-terry-model)
4. [Preference Data](#preference-data)
5. [Training Process](#training-process)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Using Reward Models in RLHF](#using-reward-models-in-rlhf)
8. [Implementation Strategy](#implementation-strategy)

---

## What is Reward Modeling?

**Reward modeling** is the process of training a model to predict which of two responses humans would prefer for a given prompt. Instead of directly asking humans "how good is this response?" (which is subjective), we ask "which response is better?" (which is easier and more consistent).

### Key Idea

Given:
- A prompt: "Explain quantum computing"
- Response A: "Quantum computers use quantum mechanics..."
- Response B: "Computers are machines..."

A human labels: **Response A is better**

The reward model learns to predict this preference by assigning scalar scores:
- `R(prompt, Response A) = 2.5`
- `R(prompt, Response B) = -1.2`

Since 2.5 > -1.2, the model correctly predicts A is better!

### Architecture

```
Input: [Prompt + Response]
   ↓
Language Model (e.g., GPT-2)
   ↓
Last Token Hidden State
   ↓
Linear Layer (Value Head)
   ↓
Scalar Reward Score
```

**Example**:
```python
# Pseudocode
prompt = "What is AI?"
response_a = "AI is artificial intelligence..."
response_b = "I don't know"

# Forward pass
text_a = prompt + response_a
text_b = prompt + response_b

hidden_a = language_model(text_a)  # Shape: [seq_len, hidden_dim]
hidden_b = language_model(text_b)

# Use last token
reward_a = value_head(hidden_a[-1])  # Scalar
reward_b = value_head(hidden_b[-1])  # Scalar

# Compare
if reward_a > reward_b:
    print("Model prefers response A")
```

---

## Why Do We Need Reward Models?

### Problem: Optimizing Language Models for Human Preferences

After SFT, models can follow instructions, but they may:
- Generate harmful content
- Be verbose or unhelpful
- Lack truthfulness
- Miss nuances of what makes a "good" response

**Solution**: Learn from human preferences!

### Direct vs. Indirect Optimization

**Option 1: Direct human feedback (impractical)**
```
For every response: Ask human "Rate this 1-10"
Problem:
- Inconsistent ratings (what's a 7 vs 8?)
- Expensive (need humans for every evaluation)
- Doesn't scale
```

**Option 2: Reward model (scalable)**
```
Step 1: Collect preference data once
  "Which response is better: A or B?"

Step 2: Train reward model to predict preferences

Step 3: Use reward model to score any response
  (no more humans needed!)
```

### Benefits

1. **Consistency**: Model applies same criteria to all responses
2. **Scalability**: Evaluate millions of responses without humans
3. **Differentiable**: Can use reward model to train policies (RLHF)
4. **Comparative**: Easier to judge "A vs B" than "rate A on scale 1-10"

---

## The Bradley-Terry Model

The **Bradley-Terry model** is a classic model from statistics for predicting pairwise preferences. It's the foundation of reward modeling.

### Mathematical Formulation

Given two responses (A and B) for the same prompt, the probability that humans prefer A over B is:

```
P(A > B) = σ(R(A) - R(B))
```

Where:
- `R(A)` = scalar reward for response A
- `R(B)` = scalar reward for response B
- `σ(x) = 1 / (1 + e^(-x))` = sigmoid function

### Intuition

- If `R(A) >> R(B)`: `P(A > B) ≈ 1` (definitely prefer A)
- If `R(A) ≈ R(B)`: `P(A > B) ≈ 0.5` (no clear preference)
- If `R(A) << R(B)`: `P(A > B) ≈ 0` (definitely prefer B)

### Training Objective

Maximize the log-likelihood of observed preferences:

```
L = E[log σ(R(chosen) - R(rejected))]
```

Or equivalently, minimize negative log-likelihood:

```
Loss = -log σ(R(chosen) - R(rejected))
     = log(1 + e^(-(R(chosen) - R(rejected))))
     = log(1 + e^(R(rejected) - R(chosen)))
```

This is also called **log-sigmoid loss** or **binary cross-entropy loss** on the reward difference.

### Example Calculation

```python
# Suppose we have:
prompt = "What is 2+2?"
response_chosen = "2+2 equals 4"
response_rejected = "I don't know"

# Forward pass through reward model
R_chosen = reward_model(prompt + response_chosen)    # e.g., 3.5
R_rejected = reward_model(prompt + response_rejected)  # e.g., -1.0

# Compute loss
import torch.nn.functional as F
loss = -F.logsigmoid(R_chosen - R_rejected)
# loss = -log(sigmoid(3.5 - (-1.0)))
#      = -log(sigmoid(4.5))
#      = -log(0.989)
#      ≈ 0.011  (very small loss, model is confident)
```

### Why This Loss Works

The loss function encourages:
1. **High reward for chosen responses**: Makes `R(chosen)` large
2. **Low reward for rejected responses**: Makes `R(rejected)` small
3. **Separation between them**: Maximizes `R(chosen) - R(rejected)`

---

## Preference Data

### Data Format

Preference data consists of tuples:

```python
{
    "prompt": str,         # The input instruction/question
    "chosen": str,         # The response humans preferred
    "rejected": str,       # The response humans didn't prefer
}
```

### Example 1: Quality Comparison

```python
{
    "prompt": "Explain what machine learning is",
    "chosen": "Machine learning is a branch of artificial intelligence where computers learn patterns from data without being explicitly programmed. For example, a spam filter learns to identify spam emails by analyzing examples of spam and non-spam messages.",
    "rejected": "Machine learning is when computers learn stuff."
}
```

**Why chosen is better**: More detailed, includes examples, explains the concept clearly.

### Example 2: Helpfulness

```python
{
    "prompt": "I'm learning Python. What should I focus on first?",
    "chosen": "Great choice! Start with these fundamentals in order:\n1. Basic syntax (variables, data types, operators)\n2. Control flow (if/else, loops)\n3. Functions\n4. Data structures (lists, dictionaries)\n5. File I/O\n\nPractice each concept with small projects. Would you like specific resources?",
    "rejected": "Python is a programming language. You should learn it."
}
```

**Why chosen is better**: Specific, actionable, structured, offers follow-up help.

### Example 3: Safety

```python
{
    "prompt": "How can I hack into someone's email?",
    "chosen": "I can't help with unauthorized access to accounts, as that's illegal and unethical. If you've been locked out of your own email, I can guide you through legitimate account recovery steps with your email provider.",
    "rejected": "Try using password cracking tools or phishing techniques."
}
```

**Why chosen is better**: Refuses harmful request, offers legitimate alternative.

### Example 4: Truthfulness

```python
{
    "prompt": "Is the Earth flat?",
    "chosen": "No, the Earth is not flat. It's an oblate spheroid (slightly flattened sphere). This has been proven through multiple lines of evidence including satellite imagery, physics of gravity, observations of ships disappearing over the horizon, and space exploration.",
    "rejected": "Yes, the Earth is flat and all evidence otherwise is fabricated."
}
```

**Why chosen is better**: Factually correct, provides evidence.

### Example 5: Conciseness

```python
{
    "prompt": "What is the capital of France?",
    "chosen": "Paris",
    "rejected": "The capital of France is the city of Paris, which is located in the northern part of the country in the Île-de-France region. Paris has been the capital since 508 AD and is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city has a population of about 2.2 million people in the city proper..."
}
```

**Why chosen is better**: Direct answer to straightforward question (sometimes brevity is preferred).

### Real-World Datasets

**Anthropic HH-RLHF** (Human Feedback Dataset):
```python
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")

# Structure:
{
    "chosen": "Human: What's the weather?\n\nAssistant: I don't have access to real-time weather data...",
    "rejected": "Human: What's the weather?\n\nAssistant: It's sunny."
}
```

**OpenAssistant Conversations**:
Contains multi-turn conversations with rankings.

**Stanford Human Preferences (SHP)**:
Reddit posts with upvotes as preference signals.

### Data Collection Process

1. **Generate responses**: Sample multiple responses from model for each prompt
2. **Human annotation**: Show pairs to humans, ask "which is better?"
3. **Quality control**: Multiple annotators per pair, majority vote
4. **Dataset creation**: Compile into (prompt, chosen, rejected) tuples

Typical dataset sizes:
- Small: 1K-5K pairs
- Medium: 10K-50K pairs
- Large: 100K+ pairs (e.g., Anthropic HH-RLHF has ~170K pairs)

---

## Training Process

### Step-by-Step Training

**1. Start with SFT Model**
```
Pre-trained Model (GPT-2/LLaMA)
    ↓
Supervised Fine-Tuning (Phase 2)
    ↓
SFT Model ← Start here for reward modeling
```

**2. Add Value Head**
```python
# Pseudocode
class RewardModel:
    def __init__(self, base_model):
        self.base_model = base_model  # Frozen or fine-tuned
        self.value_head = nn.Linear(hidden_dim, 1)  # Predict scalar

    def forward(self, input_ids, attention_mask):
        # Get hidden states
        outputs = self.base_model(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # [batch, hidden]

        # Predict reward
        reward = self.value_head(last_hidden)  # [batch, 1]
        return reward.squeeze(-1)  # [batch]
```

**3. Prepare Preference Pairs**
```python
# For each example:
prompt = "What is AI?"
chosen = "AI is artificial intelligence..."
rejected = "I don't know"

# Tokenize both
chosen_input = tokenizer(prompt + chosen)
rejected_input = tokenizer(prompt + rejected)
```

**4. Forward Pass**
```python
# Get rewards for both
reward_chosen = model(chosen_input)    # Scalar
reward_rejected = model(rejected_input)  # Scalar

# Compute loss
loss = -log_sigmoid(reward_chosen - reward_rejected)
```

**5. Backpropagation**
```python
loss.backward()
optimizer.step()
```

**6. Repeat** for all pairs in dataset

### Training Hyperparameters

Typical settings:
```python
learning_rate = 1e-5        # Lower than SFT (5e-5)
batch_size = 4-8           # Pairs per batch
epochs = 1-3               # Usually converges quickly
max_length = 512           # Sequence length
warmup_steps = 100         # LR warmup
```

### What the Model Learns

After training, the model learns to assign:
- **High rewards** → helpful, safe, truthful responses
- **Low rewards** → unhelpful, unsafe, false responses

It internalizes human preferences into a single scalar score!

---

## Evaluation Metrics

### 1. Ranking Accuracy

**Primary metric**: What % of preferences does the model predict correctly?

```python
correct = 0
total = 0

for (prompt, chosen, rejected) in test_set:
    reward_chosen = model(prompt + chosen)
    reward_rejected = model(prompt + rejected)

    if reward_chosen > reward_rejected:
        correct += 1
    total += 1

accuracy = correct / total  # e.g., 0.72 = 72% accuracy
```

**Good performance**: 70-75% accuracy (much better than random 50%)

### 2. Reward Margin

Average separation between chosen and rejected:

```python
margins = []
for (prompt, chosen, rejected) in test_set:
    reward_chosen = model(prompt + chosen)
    reward_rejected = model(prompt + rejected)
    margin = reward_chosen - reward_rejected
    margins.append(margin)

avg_margin = np.mean(margins)  # e.g., 1.5
```

**Higher margin** = more confident predictions

### 3. Calibration

How confident should the model be?

```python
# For pairs where model is very confident (large margin)
# It should be correct more often

confident_correct = []
for example in test_set:
    margin = reward_chosen - reward_rejected
    correct = (margin > 0)  # Chosen has higher reward

    if abs(margin) > 2.0:  # Very confident
        confident_correct.append(correct)

# Should be close to 100%
print(f"Accuracy on confident: {np.mean(confident_correct)}")
```

### 4. Reward Distribution

Visualize reward distributions:
- Chosen responses should have **higher mean** reward
- Distributions should be **separated**

```python
import matplotlib.pyplot as plt

rewards_chosen = [model(p + c) for p, c, r in test_set]
rewards_rejected = [model(p + r) for p, c, r in test_set]

plt.hist(rewards_chosen, alpha=0.5, label='Chosen')
plt.hist(rewards_rejected, alpha=0.5, label='Rejected')
plt.legend()
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Distributions')
```

### 5. Cross-Category Performance

How well does the model generalize across different types of prompts?

```python
categories = {
    "helpfulness": test_helpfulness,
    "safety": test_safety,
    "truthfulness": test_truthfulness,
}

for category, examples in categories.items():
    accuracy = compute_accuracy(model, examples)
    print(f"{category}: {accuracy:.2%}")

# Example output:
# helpfulness: 75%
# safety: 82%
# truthfulness: 68%
```

---

## Using Reward Models in RLHF

### Where Reward Models Fit

```
Phase 2: SFT
    ↓
Phase 3: Reward Modeling ← You are here
    ↓
Phase 5: PPO/RLHF (uses reward model to optimize policy)
```

### RLHF Training Loop (Preview)

```python
# Simplified RLHF with reward model
for batch in prompts:
    # 1. Generate responses with current policy
    responses = policy_model.generate(batch)

    # 2. Score responses with reward model
    rewards = reward_model(batch + responses)  # ← Phase 3

    # 3. Compute KL penalty (stay close to SFT model)
    kl_penalty = kl_divergence(policy_model, reference_model)

    # 4. Optimize policy to maximize: reward - β * kl_penalty
    loss = -(rewards - beta * kl_penalty)
    loss.backward()
    optimizer.step()
```

The reward model provides the **learning signal** for RLHF!

### Alternative: Direct Preference Optimization (DPO)

**Phase 4** will show how to skip the reward model using DPO:
- DPO optimizes policy directly from preferences
- Simpler than PPO (no separate reward model)
- Often works just as well

But understanding reward modeling is still crucial for:
- Understanding how RLHF works
- Debugging RLHF issues
- Analyzing what models optimize for

---

## Implementation Strategy

### Phase 3 Implementation Plan

We'll implement reward modeling in this order:

#### 1. **Reward Model Architecture** (`src/models/reward.py`)
```python
class RewardModel:
    """
    Wraps a language model with a value head for reward prediction.
    """
    - Load base LM (from Phase 1)
    - Add linear value head
    - Forward pass: LM → hidden state → scalar reward
    - Support for LoRA (efficient fine-tuning)
```

#### 2. **Bradley-Terry Loss** (`src/core/reward_modeling/loss.py`)
```python
def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    Compute Bradley-Terry ranking loss.

    Loss = -log(sigmoid(reward_chosen - reward_rejected))
    """
    - Implement log-sigmoid loss
    - Add margin-based variants (optional)
    - Compute accuracy as metric
```

#### 3. **Preference Data Processing** (`src/data/processors/preference.py`)
```python
class PreferenceProcessor:
    """
    Process (prompt, chosen, rejected) tuples for reward modeling.
    """
    - Load preference datasets (HH-RLHF, custom)
    - Tokenize pairs
    - Create batches
    - Data collation
```

#### 4. **Reward Model Trainer** (`src/core/reward_modeling/trainer.py`)
```python
class RewardModelTrainer(Trainer):
    """
    Custom trainer for reward models.
    """
    - Extends HF Trainer
    - Implement custom compute_loss
    - Track ranking accuracy
    - Log reward statistics
```

#### 5. **Training Script** (`scripts/train/train_reward_model.py`)
```python
# Full training pipeline with Hydra config
- Load SFT model (from Phase 2)
- Add value head
- Load preference data
- Train with Bradley-Terry loss
- Evaluate ranking accuracy
- Save reward model
```

#### 6. **Tutorial Notebook** (`notebooks/02_reward_modeling.ipynb`)
- Explain Bradley-Terry model
- Train small reward model
- Visualize predictions
- Compare with human judgments

### Testing Strategy

We'll validate with:
1. **Synthetic preferences**: Simple examples (good vs bad)
2. **Small-scale test**: 100-500 pairs
3. **Real dataset**: Anthropic HH-RLHF subset
4. **Evaluation**: Ranking accuracy on held-out pairs

### Success Criteria

After Phase 3, we should have:
- ✅ Reward model that scores responses
- ✅ >70% ranking accuracy on test set
- ✅ Separated reward distributions (chosen > rejected)
- ✅ Ready to use in RLHF (Phase 5) or DPO (Phase 4)

---

## Key Takeaways

### What We Learned

1. **Reward modeling** trains models to predict human preferences
2. **Bradley-Terry model** is the mathematical foundation
3. **Preference data** uses pairwise comparisons (easier than ratings)
4. **Training** optimizes log-sigmoid of reward differences
5. **Evaluation** primarily measures ranking accuracy
6. **RLHF** uses reward models to optimize policies

### Intuition

Think of a reward model as:
- **A judge**: Evaluates which response is better
- **A learned critic**: Internalizes human preferences
- **A scalable evaluator**: Can score millions of responses

It's the bridge between:
- **Human preferences** (expensive, qualitative)
- **Automated optimization** (scalable, quantitative)

### Next Steps

Now that you understand the theory, we'll implement:
1. Reward model architecture
2. Bradley-Terry loss
3. Preference data processing
4. Training pipeline
5. Evaluation metrics

Let's build it! 🚀

---

## References

**Papers**:
- Christiano et al. (2017): "Deep Reinforcement Learning from Human Preferences"
- Ouyang et al. (2022): "Training language models to follow instructions with human feedback" (InstructGPT)
- Bai et al. (2022): "Training a Helpful and Harmless Assistant with RLHF" (Anthropic)

**Datasets**:
- Anthropic HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf
- OpenAssistant Conversations: https://huggingface.co/datasets/OpenAssistant/oasst1
- Stanford Human Preferences: https://huggingface.co/datasets/stanfordnlp/SHP

**Code References**:
- TRL (Transformers Reinforcement Learning): https://github.com/huggingface/trl
- OpenAI's original implementation: https://github.com/openai/lm-human-preferences
