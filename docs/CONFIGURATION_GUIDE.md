# Configuration System Guide: Hydra & OmegaConf

This document explains how we use Hydra and OmegaConf for configuration management throughout the codebase.

## Table of Contents
- [What is Hydra?](#what-is-hydra)
- [What is OmegaConf?](#what-is-omegaconf)
- [Repository Configuration Structure](#repository-configuration-structure)
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

**Problem:** `_self_` in wrong place in defaults list.

**Solution:** Put `_self_` after other defaults to ensure overrides apply:
```yaml
defaults:
  - model: gpt2
  - _self_  # This file overrides model/gpt2.yaml
```

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
- `docs/DPO_CONFIGURATION.md` - DPO-specific config guide (includes HH-RLHF dataset)
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

---

**That's it!** You now understand how Hydra and OmegaConf power our configuration system. This makes experimenting with different hyperparameters, models, and datasets as simple as changing a command-line argument.
