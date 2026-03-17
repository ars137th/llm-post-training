# Configuration System Guide: Hydra & OmegaConf

This document explains how we use Hydra and OmegaConf for configuration management throughout the codebase.

## Table of Contents
- [What is Hydra?](#what-is-hydra)
- [What is OmegaConf?](#what-is-omegaconf)
- [Repository Configuration Structure](#repository-configuration-structure)
- [⚠️ CRITICAL: Understanding `_self_` and Config Merge Order](#️-critical-understanding-_self_-and-config-merge-order)
- [The `cfg` Pattern](#the-cfg-pattern)
- [Command-Line Overrides](#command-line-overrides)
- [Creating Experiment Configs](#creating-experiment-configs)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

---

## What is Hydra?

**Hydra** is a configuration management framework that allows you to:
- Compose configurations from multiple files
- Override parameters from the command line
- Manage complex experiment configurations
- Organize hyperparameters hierarchically

**Official docs:** https://hydra.cc/

**Why we use it:**
- ✅ No hardcoded parameters in scripts
- ✅ Easy to run experiments with different settings
- ✅ Reproducible: configs are saved with outputs
- ✅ Composable: mix and match model/data/technique configs
- ✅ Type-safe with OmegaConf

---

## What is OmegaConf?

**OmegaConf** is the underlying configuration library used by Hydra.

**Key features:**
- **Dict-like access:** `cfg.model.name` or `cfg["model"]["name"]`
- **Dot notation:** `cfg.training.learning_rate`
- **Type checking:** Validates config types
- **Variable interpolation:** `output_dir: "./outputs/${model.name}"`
- **YAML-based:** Human-readable configuration files

**Official docs:** https://omegaconf.readthedocs.io/

---

## Repository Configuration Structure

Our configs are organized hierarchically in the `configs/` directory:

```
configs/
├── config.yaml                    # Main SFT config (entry point)
├── config_dpo.yaml               # Main DPO config (entry point)
├── config_ppo.yaml               # Main PPO config (entry point)
├── config_reward.yaml            # Main Reward Model config (entry point)
│
├── model/                        # Model configurations
│   ├── gpt2.yaml                # GPT-2 settings
│   └── opt-350m.yaml            # OPT-350m settings
│
├── technique/                    # Technique-specific hyperparameters
│   ├── sft.yaml                 # Supervised Fine-Tuning
│   ├── reward_model.yaml        # Reward Modeling
│   ├── dpo.yaml                 # Direct Preference Optimization
│   └── ppo.yaml                 # Proximal Policy Optimization
│
├── data/                         # Data configurations
│   ├── conversation.yaml        # Conversation data (SFT)
│   ├── preference.yaml          # Preference pairs (DPO, Reward)
│   └── prompts.yaml             # Prompts for PPO
│
└── experiment/                   # Full experiment configs
    ├── dpo_gpt2_synthetic.yaml
    ├── dpo_gpt2_hh_rlhf.yaml
    ├── ppo_gpt2_synthetic.yaml
    └── ...
```

### Configuration Composition

Hydra uses a **defaults list** to compose configurations from multiple files:

```yaml
# configs/config_dpo.yaml
defaults:
  - model: gpt2              # Load configs/model/gpt2.yaml
  - technique: dpo           # Load configs/technique/dpo.yaml
  - data: preference         # Load configs/data/preference.yaml
  - optional experiment: null  # Optionally load experiment config
  - _self_                   # Apply settings from this file last

# These defaults are loaded in order and merged
```

**Key concept:** Files listed in `defaults` are loaded and merged together. Later files override earlier ones. `_self_` determines when the current file's settings are applied.

---

## ⚠️ CRITICAL: Understanding `_self_` and Config Merge Order

**This is NOT a bug, but an expected Hydra pattern that is easy to get wrong.**

### The Problem: Experiment Configs Not Working

The position of `_self_` in the `defaults` list controls **when** that config file's values are applied in the merge order. If you place `_self_` in the wrong position, your experiment configs won't override base configs.

**Real bug we encountered:**
```yaml
# ❌ WRONG - Base config was like this initially:
defaults:
  - model: gpt2
  - technique: dpo
  - data: preference
  - optional experiment: null  # Experiment loads HERE
  - _self_                     # Base config overrides it! ❌

data:
  use_synthetic: true  # Base default
```

**What happens:**
1. Load `model/gpt2.yaml` ✅
2. Load `technique/dpo.yaml` ✅
3. Load `data/preference.yaml` ✅
4. Load `experiment/dpo_gpt2_hh_rlhf.yaml` (sets `use_synthetic: false`) ✅
5. **Apply base config's `_self_`** (sets `use_synthetic: true`) ❌ **BASE WINS!**

**Result:** Even though experiment config says `use_synthetic: false`, the final value is `true` because base config applied last.

### The Fix: Position `_self_` Correctly

```yaml
# ✅ CORRECT - All our base configs now look like this:
defaults:
  - model: gpt2
  - technique: dpo
  - data: preference
  - _self_                     # Apply base config values first ✅
  - optional experiment: null  # Then experiment overrides them ✅

data:
  # NOTE: These are BASE DEFAULTS. Experiment configs override these values.
  use_synthetic: true
  num_train_examples: 500
```

**What happens:**
1. Load `model/gpt2.yaml` ✅
2. Load `technique/dpo.yaml` ✅
3. Load `data/preference.yaml` ✅
4. **Apply base config's `_self_`** (sets `use_synthetic: true`) ✅
5. Load `experiment/dpo_gpt2_hh_rlhf.yaml` (sets `use_synthetic: false`) ✅ **EXPERIMENT WINS!**

**Result:** Final value is `use_synthetic: false` as expected. 🎉

### Visual Example: Merge Order Matters

**Example 1: Base config without proper `_self_` positioning**

```yaml
# configs/config_reward.yaml
defaults:
  - model: gpt2
  - optional experiment: null
  - _self_  # ❌ Applied LAST, overrides experiment

data:
  use_synthetic: true  # This wins even if experiment says false!
  num_train_examples: 500
```

**Merge order:** model → experiment → **base (wins!)**

```yaml
# configs/experiment/reward_gpt2_hh_rlhf.yaml
data:
  use_synthetic: false       # ❌ Gets overridden by base!
  dataset_name: "Anthropic/hh-rlhf"
  num_train_examples: 10000  # ❌ Gets overridden by base!
```

**Final config:**
```yaml
data:
  use_synthetic: true        # ❌ Base value (wrong!)
  dataset_name: "Anthropic/hh-rlhf"  # ✅ Experiment value
  num_train_examples: 500    # ❌ Base value (wrong!)
```

**Example 2: Base config with correct `_self_` positioning**

```yaml
# configs/config_reward.yaml
defaults:
  - model: gpt2
  - _self_  # ✅ Applied BEFORE experiment
  - optional experiment: null

data:
  use_synthetic: true  # Base default, but experiment can override
  num_train_examples: 500
```

**Merge order:** model → **base** → experiment (wins!)

```yaml
# configs/experiment/reward_gpt2_hh_rlhf.yaml
data:
  use_synthetic: false       # ✅ Overrides base!
  dataset_name: "Anthropic/hh-rlhf"
  num_train_examples: 10000  # ✅ Overrides base!
```

**Final config:**
```yaml
data:
  use_synthetic: false       # ✅ Experiment value (correct!)
  dataset_name: "Anthropic/hh-rlhf"  # ✅ Experiment value
  num_train_examples: 10000  # ✅ Experiment value (correct!)
```

### Rules for `_self_` Positioning

**1. Base configs (config.yaml, config_reward.yaml, config_dpo.yaml, config_ppo.yaml):**
```yaml
defaults:
  - model: gpt2
  - technique: dpo
  - data: preference
  - _self_                     # ✅ BEFORE optional experiment
  - optional experiment: null  # Lets experiments override us
```

**2. Experiment configs (experiment/xxx.yaml):**
```yaml
# @package _global_

defaults:
  - override /model: gpt2
  - override /technique: dpo
  - override /data: preference
  - _self_  # ✅ Apply our overrides last

data:
  use_synthetic: false  # Override base config
```

**Why experiment configs need `_self_` too:**
- Ensures experiment's own values take priority
- Prevents nested config groups from overriding experiment settings

### When Order Matters Most

**Scenario 1: Experiment configs (our bug)**
- **Problem:** Experiment says use real data, but synthetic data is used
- **Cause:** Base config's `_self_` came after `optional experiment`
- **Fix:** Move `_self_` before `optional experiment` in base config

**Scenario 2: Nested config groups**
```yaml
# model/gpt2.yaml
name: "gpt2"
hidden_size: 768

# Your config
defaults:
  - model: gpt2
  - _self_

model:
  hidden_size: 1024  # Override nested config

# Result: hidden_size = 1024 (your value wins)
```

Without `_self_`, `hidden_size` would be 768 from `model/gpt2.yaml`.

**Scenario 3: Multiple inheritance**
```yaml
defaults:
  - base_config
  - _self_
  - advanced_config  # Overrides both base_config and _self_

# Merge order: base_config → this file → advanced_config
```

### Debugging Config Merge Issues

We created `scripts/debug_config.py` to diagnose these issues:

```bash
# Debug what config values are actually loaded
python scripts/debug_config.py experiment=reward_gpt2_hh_rlhf

# Output shows:
# [DATA CONFIGURATION]
#   use_synthetic: False  ✅ (should be False for this experiment)
#   dataset_name: Anthropic/hh-rlhf
#   format: anthropic
```

**If values are wrong:**
1. Check `_self_` position in base config
2. Verify experiment config has `_self_` directive
3. Clear Hydra cache: `rm -rf .hydra/` and `rm -rf outputs/.hydra/`
4. Check for typos in config file names

### Common Pitfalls

**Pitfall 1: Forgetting `_self_` in experiment configs**
```yaml
# ❌ Missing _self_
defaults:
  - override /model: gpt2
  - override /technique: dpo
  # Missing _self_!

data:
  use_synthetic: false  # Might get overridden by nested configs
```

**Pitfall 2: `_self_` in wrong position in base configs**
```yaml
# ❌ _self_ after optional experiment
defaults:
  - model: gpt2
  - optional experiment: null  # Experiment loads here
  - _self_  # Base config overrides experiment!
```

**Pitfall 3: Not using `override /` prefix in experiment configs**
```yaml
# ❌ Without override prefix
defaults:
  - model: gpt2  # Doesn't override, just loads

# ✅ With override prefix
defaults:
  - override /model: gpt2  # Explicitly overrides
```

### Summary: The Golden Rules

1. **Base configs:** Put `_self_` **BEFORE** `optional experiment: null`
   ```yaml
   defaults:
     - model: gpt2
     - _self_                     # ← HERE (before experiment)
     - optional experiment: null
   ```

2. **Experiment configs:** Put `_self_` **LAST** in defaults
   ```yaml
   defaults:
     - override /model: gpt2
     - override /technique: dpo
     - _self_  # ← HERE (last)
   ```

3. **When in doubt:** Use `scripts/debug_config.py` to see actual values

4. **Remember:** This is NOT a bug—it's a powerful feature that gives you precise control over config composition!

### References

- **Hydra Defaults List:** https://hydra.cc/docs/advanced/defaults_list/
- **Hydra Composition Order:** https://hydra.cc/docs/advanced/composition_order/
- **Our docs:**
  - `docs/REWARD_MODELING_CONFIGURATION.md` - Detailed example of this pattern
  - `scripts/debug_config.py` - Debug tool for config issues

---

## The `cfg` Pattern

### In Training Scripts

All training scripts follow this pattern:

```python
# scripts/train/train_dpo.py

from omegaconf import DictConfig, OmegaConf
import hydra

def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Configuration object from Hydra
    """
    # Access nested config values with dot notation
    model_name = cfg.model.name
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.per_device_train_batch_size

    # Check if optional values exist
    if cfg.training.get('fp16', False):
        print("FP16 training enabled")

    # Pretty print entire config
    print(OmegaConf.to_yaml(cfg))

    # Convert OmegaConf dict to regular Python dict (if needed)
    training_args = OmegaConf.to_container(cfg.training, resolve=True)

if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Setup Hydra
    GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=CONFIGS_PATH, job_name="train_dpo"):
        cfg = compose(config_name="config_dpo", overrides=sys.argv[1:])
        main(cfg)
```

### Accessing Config Values

```python
# Dot notation (preferred)
model_name = cfg.model.name
use_lora = cfg.model.use_lora

# Dictionary notation (also works)
model_name = cfg["model"]["name"]

# Safe access with default
fp16 = cfg.training.get("fp16", False)

# Check if key exists
if "fp16" in cfg.training:
    # ...

# Nested access
lora_rank = cfg.model.lora_config.r
```

### Type Annotations

```python
from omegaconf import DictConfig

def train_model(cfg: DictConfig):
    # cfg is type-hinted for better IDE support
    # Access config values directly
    epochs: int = cfg.training.num_epochs
    lr: float = cfg.training.learning_rate
```

---

## Command-Line Overrides

Hydra allows overriding any config value from the command line:

### Basic Overrides

```bash
# Override a single value
python scripts/train/train_dpo.py training.learning_rate=1e-6

# Override nested values
python scripts/train/train_dpo.py model.lora_config.r=16

# Override multiple values
python scripts/train/train_dpo.py \
    training.learning_rate=1e-6 \
    training.num_epochs=3 \
    model.use_lora=true
```

### Override Syntax

```bash
# String values
python script.py model.name=gpt2

# Numeric values
python script.py training.learning_rate=1e-6
python script.py training.batch_size=32

# Boolean values
python script.py model.use_lora=true
python script.py training.fp16=false

# Null values
python script.py data.max_length=null

# Lists
python script.py model.lora_config.target_modules=[c_attn,c_proj]
```

### Selecting Different Configs

```bash
# Use different model config
python scripts/train/train_dpo.py model=opt-350m

# Use different technique config
python scripts/train/train_dpo.py technique=dpo

# Use different data config
python scripts/train/train_dpo.py data=preference

# Use experiment config (overrides all base configs)
python scripts/train/train_dpo.py experiment=dpo_gpt2_hh_rlhf
```

### Special Override Prefixes

```bash
# + (Add): Add a new key-value pair
python script.py +new_param=value

# ++ (Force add): Force add even if key exists
python script.py ++existing_param=new_value

# ~ (Delete): Remove a key
python script.py ~model.use_lora
```

---

## Creating Experiment Configs

Experiment configs bundle all settings for a specific experiment.

### Example: DPO Experiment

**File:** `configs/experiment/dpo_gpt2_hh_rlhf.yaml`

```yaml
# @package _global_
# This directive tells Hydra to merge at the global level

# Specify which base configs to use
defaults:
  - override /model: gpt2
  - override /technique: dpo
  - override /data: preference
  - _self_  # Apply this file's settings last

# Override model settings
model:
  name: "gpt2"
  use_lora: true
  lora_config:
    r: 8
    lora_alpha: 16
    target_modules: ["c_attn"]

# Override data settings
data:
  use_synthetic: false
  dataset_name: "Anthropic/hh-rlhf"
  num_train_examples: 10000
  num_eval_examples: 1000

# Override training settings
training:
  output_dir: "./outputs/dpo_gpt2_hh_rlhf"
  num_epochs: 1
  learning_rate: 5e-7
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  fp16: true

# Override technique settings
technique:
  beta: 0.1
  loss_type: "dpo"

# System settings
device: "cuda"
seed: 42
```

**Usage:**

```bash
# Run with this experiment config
python scripts/train/train_dpo.py experiment=dpo_gpt2_hh_rlhf

# Further override experiment settings
python scripts/train/train_dpo.py \
    experiment=dpo_gpt2_hh_rlhf \
    training.learning_rate=1e-6 \
    device=cpu
```

### When to Create an Experiment Config

Create an experiment config when:
- ✅ You have a specific set of hyperparameters you use repeatedly
- ✅ You want to save a configuration for reproducibility
- ✅ You want to share a working configuration with others
- ✅ You're comparing multiple approaches systematically

**Naming convention:**
- `{technique}_{model}_{dataset}.yaml`
- Examples: `dpo_gpt2_synthetic.yaml`, `ppo_gpt2_full.yaml`

---

## Advanced Usage

### Variable Interpolation

Reference other config values within the config:

```yaml
model:
  name: "gpt2"

training:
  output_dir: "./outputs/${model.name}"  # Interpolates to "./outputs/gpt2"

logging:
  wandb_run_name: "${model.name}_${technique.beta}"
```

### Conditional Logic

```yaml
# Use different settings based on device
training:
  fp16: ${oc.select:device,cuda}  # True if device is cuda, false otherwise
```

### Environment Variables

```yaml
# Reference environment variables
training:
  output_dir: "${oc.env:OUTPUT_DIR,./outputs}"  # Falls back to ./outputs if not set
```

### Config Groups

Create alternative configurations in subfolders:

```
configs/model/
├── gpt2/
│   ├── base.yaml
│   ├── medium.yaml
│   └── large.yaml
└── opt/
    ├── 350m.yaml
    └── 1.3b.yaml
```

```bash
# Select specific variant
python script.py model=gpt2/large
python script.py model=opt/1.3b
```

### Structured Configs (Type Safety)

For advanced type checking, use dataclasses:

```python
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class ModelConfig:
    name: str = MISSING  # Must be provided
    use_lora: bool = False
    hidden_size: int = 768

@dataclass
class Config:
    model: ModelConfig = ModelConfig()

# Hydra validates types at runtime
```

---

## Best Practices

### 1. Use Structured Configs for Reusable Components

**Good:**
```yaml
# configs/model/gpt2.yaml
name: "gpt2"
use_lora: true
lora_config:
  r: 8
  lora_alpha: 16
```

**Bad:**
```yaml
# Hardcoding everything in one file
model_name: "gpt2"
model_use_lora: true
model_lora_r: 8
```

### 2. Keep Defaults Sensible

Base configs should have sensible defaults that work out-of-the-box:

```yaml
# configs/technique/dpo.yaml
beta: 0.1  # Good default, works for most cases
loss_type: "dpo"  # Standard choice
```

### 3. Document Config Values

Add comments explaining non-obvious parameters:

```yaml
technique:
  beta: 0.1  # KL penalty coefficient (0.05=aggressive, 0.3=conservative)
  loss_type: "dpo"  # Options: "dpo" (standard) or "ipo" (robust to outliers)
```

### 4. Use `_self_` Deliberately

Control merge order with `_self_`:

```yaml
defaults:
  - model: gpt2
  - _self_  # This file's settings override model/gpt2.yaml

model:
  name: "gpt2"  # Overrides the name from model/gpt2.yaml
```

**⚠️ CRITICAL:** The position of `_self_` is crucial! See the [Understanding `_self_` and Config Merge Order](#️-critical-understanding-_self_-and-config-merge-order) section for detailed examples and common pitfalls.

### 5. Validate Configs Early

Check required fields at the start of your training function:

```python
def main(cfg: DictConfig):
    # Validate required fields
    assert cfg.model.name is not None, "model.name must be specified"
    assert cfg.training.learning_rate > 0, "learning_rate must be positive"

    # Continue with training...
```

### 6. Pretty Print Configs

Always print the full config at the start of training:

```python
from omegaconf import OmegaConf

def main(cfg: DictConfig):
    # Print config in readable YAML format
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Or save to file
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
```

### 7. Use `.get()` for Optional Values

```python
# Good: Provides default if key doesn't exist
fp16 = cfg.training.get("fp16", False)

# Bad: Will error if key doesn't exist
fp16 = cfg.training.fp16
```

### 8. Keep Experiment Configs Minimal

Don't repeat everything from base configs:

**Good:**
```yaml
# experiment/dpo_custom.yaml
defaults:
  - override /model: gpt2
  - override /technique: dpo
  - _self_

# Only override what's different
training:
  learning_rate: 1e-6  # Changed from default
```

**Bad:**
```yaml
# Repeating everything unnecessarily
defaults:
  - override /model: gpt2

model:
  name: "gpt2"
  use_lora: true
  lora_config:
    r: 8
    lora_alpha: 16
    # ... everything from base config
```

---

## Common Patterns in Our Codebase

### Pattern 1: Training Script Setup

```python
# Standard setup at top of training scripts
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
CONFIGS_PATH = str(project_root / "configs")

def main(cfg: DictConfig):
    # Use config
    pass

if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_PATH, job_name="train_xxx"):
        cfg = compose(config_name="config_xxx", overrides=sys.argv[1:])
        main(cfg)
```

### Pattern 2: Converting OmegaConf to Python Dict

```python
# When you need a plain Python dict (e.g., for HuggingFace APIs)
training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)

# Use in TrainingArguments
training_args = TrainingArguments(**training_args_dict)
```

### Pattern 3: Conditional Config Values

```python
# Check if optional config exists
if cfg.model.get('use_lora', False):
    # Apply LoRA
    model = apply_lora(model, cfg.model.lora_config)
```

### Pattern 4: Dynamic Output Directories

```yaml
# In config file
training:
  output_dir: "./outputs/${model.name}_${technique.beta}"
  # Results in: ./outputs/gpt2_0.1
```

---

## Examples: Common Use Cases

### Use Case 1: Quick Hyperparameter Sweep

```bash
# Test different learning rates
for lr in 1e-7 5e-7 1e-6; do
    python scripts/train/train_dpo.py \
        experiment=dpo_gpt2_synthetic \
        training.learning_rate=$lr \
        training.output_dir="./outputs/dpo_lr_${lr}"
done
```

### Use Case 2: Debugging with Small Data

```bash
# Run on tiny dataset for debugging
python scripts/train/train_dpo.py \
    data.num_train_examples=10 \
    data.num_eval_examples=5 \
    training.num_epochs=1
```

### Use Case 3: Resume from Checkpoint

```bash
python scripts/train/train_dpo.py \
    experiment=dpo_gpt2_hh_rlhf \
    training.resume_from_checkpoint="./outputs/dpo_gpt2/checkpoint-1000"
```

### Use Case 4: CPU vs GPU

```bash
# CPU training
python scripts/train/train_dpo.py device=cpu

# GPU with FP16
python scripts/train/train_dpo.py device=cuda training.fp16=true

# Multi-GPU (handled by accelerate)
accelerate launch scripts/train/train_dpo.py device=cuda
```

---

## Troubleshooting

### Error: "Could not override 'X'"

**Problem:** Trying to override a config group that doesn't exist in defaults.

**Solution:**
```bash
# Bad
python script.py experiment=my_exp  # experiment not in defaults

# Good - add to defaults in config file
defaults:
  - optional experiment: null  # Now can override

# Or use +
python script.py +experiment=my_exp
```

### Error: "Missing mandatory value"

**Problem:** Required config value not set.

**Solution:** Set it in config file or via command line:
```bash
python script.py model.name=gpt2
```

### Config Values Not Updating

**Problem:** `_self_` in wrong place in defaults list, causing experiment configs not to override base configs.

**Solution:** For base configs, put `_self_` **BEFORE** `optional experiment: null`:
```yaml
# Base config (config_reward.yaml, config_dpo.yaml, etc.)
defaults:
  - model: gpt2
  - technique: dpo
  - _self_                     # Apply base values first
  - optional experiment: null  # Let experiments override
```

For experiment configs, put `_self_` **LAST**:
```yaml
# Experiment config
defaults:
  - override /model: gpt2
  - override /technique: dpo
  - _self_  # Apply experiment values last
```

**⚠️ See detailed explanation:** [Understanding `_self_` and Config Merge Order](#️-critical-understanding-_self_-and-config-merge-order)

### Type Errors

**Problem:** Passing wrong type from command line.

**Solution:** Use correct syntax:
```bash
# Strings: no quotes needed
python script.py model.name=gpt2

# Booleans: lowercase
python script.py model.use_lora=true  # or false

# Numbers: no quotes
python script.py training.learning_rate=1e-6

# Lists: brackets
python script.py model.layers=[1,2,3]
```

---

## Resources

**Official Documentation:**
- Hydra: https://hydra.cc/
- OmegaConf: https://omegaconf.readthedocs.io/

**Tutorials:**
- Hydra Tutorial: https://hydra.cc/docs/tutorials/intro/
- OmegaConf Primer: https://omegaconf.readthedocs.io/en/latest/usage.html

**Related Docs in This Repo:**
- `docs/REWARD_MODELING_CONFIGURATION.md` - Reward modeling config guide with `_self_` examples
- `docs/DPO_CONFIGURATION.md` - DPO-specific config guide (includes HH-RLHF dataset)
- `scripts/debug_config.py` - Debug tool for diagnosing config merge issues
- `configs/technique/ppo.yaml` - PPO hyperparameters with inline docs
- `configs/technique/dpo.yaml` - DPO hyperparameters with inline docs

---

## Summary: Quick Reference

### Directory Structure
```
configs/
├── config_{technique}.yaml    # Entry point configs
├── model/                     # Model configs
├── technique/                 # Technique hyperparameters
├── data/                      # Data configs
└── experiment/                # Full experiment configs
```

### Command-Line Usage
```bash
# Basic
python scripts/train/train_dpo.py

# Override values
python scripts/train/train_dpo.py training.learning_rate=1e-6

# Use experiment config
python scripts/train/train_dpo.py experiment=dpo_gpt2_hh_rlhf

# Multiple overrides
python scripts/train/train_dpo.py \
    model=gpt2 \
    technique.beta=0.2 \
    training.fp16=true \
    device=cuda
```

### In Python Code
```python
from omegaconf import DictConfig, OmegaConf

def main(cfg: DictConfig):
    # Access values
    lr = cfg.training.learning_rate

    # Safe access with default
    fp16 = cfg.training.get("fp16", False)

    # Convert to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Pretty print
    print(OmegaConf.to_yaml(cfg))
```

### Creating Configs
```yaml
# experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /model: gpt2
  - override /technique: dpo
  - _self_

training:
  learning_rate: 1e-6
  output_dir: "./outputs/my_experiment"
```

### ⚠️ Critical Reminder: `_self_` Positioning

**For base configs:**
```yaml
defaults:
  - model: gpt2
  - _self_                     # BEFORE optional experiment
  - optional experiment: null
```

**For experiment configs:**
```yaml
defaults:
  - override /model: gpt2
  - _self_  # LAST in the list
```

**Why this matters:** Wrong `_self_` position = experiment configs won't override base configs!

See full details: [Understanding `_self_` and Config Merge Order](#️-critical-understanding-_self_-and-config-merge-order)

---

**That's it!** You now understand how Hydra and OmegaConf power our configuration system. This makes experimenting with different hyperparameters, models, and datasets as simple as changing a command-line argument.

**Remember:** The `_self_` directive is a powerful tool for precise config control—use it correctly!
