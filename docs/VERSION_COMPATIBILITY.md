# Version Compatibility Guide

## Overview

This repository supports multiple versions of `transformers` and `PyTorch` to accommodate different platforms:

- **macOS with nomkl**: PyTorch 2.0.x, transformers 4.35.x
- **GPU platforms (Linux/Colab/Cloud)**: PyTorch 2.4+, transformers 4.36+

To handle API differences between these versions, we've implemented a **compatibility layer** that automatically detects installed versions and adapts the code accordingly.

---

## API Differences Between transformers 4.35.x and 4.36+

### 1. TrainingArguments

**transformers <4.36**:
```python
TrainingArguments(
    evaluation_strategy="steps",  # Old parameter name
    logging_dir="./logs",         # Accepted parameter
    ...
)
```

**transformers 4.36+**:
```python
TrainingArguments(
    eval_strategy="steps",  # Renamed parameter
    # logging_dir removed, use env var instead
    ...
)
os.environ['TENSORBOARD_LOGGING_DIR'] = "./logs"
```

### 2. Trainer.__init__()

**transformers <4.36**:
```python
Trainer(
    model=model,
    tokenizer=tokenizer,  # Accepted parameter
    ...
)
```

**transformers 4.36+**:
```python
# tokenizer parameter removed
Trainer(
    model=model,
    # Must store tokenizer separately
    ...
)
trainer.tokenizer = tokenizer
```

### 3. Trainer.training_step()

**transformers <4.36**:
```python
def training_step(self, model, inputs):
    # Two parameters
    ...
```

**transformers 4.36+**:
```python
def training_step(self, model, inputs, num_items_in_batch):
    # Three parameters (num_items_in_batch added)
    ...
```

---

## Compatibility Layer Implementation

### Location
All compatibility utilities are in: `src/utils/compat.py`

### Key Functions

#### 1. `get_training_args_kwargs()`

Generates version-appropriate TrainingArguments kwargs:

```python
from src.utils.compat import get_training_args_kwargs

kwargs = get_training_args_kwargs(
    output_dir="./outputs",
    eval_enabled=True,
    logging_dir="./logs",
    learning_rate=5e-5,
    # ... other args
)

training_args = TrainingArguments(**kwargs)
```

**What it does**:
- Detects transformers version
- Uses `eval_strategy` or `evaluation_strategy` as appropriate
- Includes or excludes `logging_dir` as appropriate
- Sets `TENSORBOARD_LOGGING_DIR` env var for 4.36+ when needed

#### 2. `get_trainer_init_kwargs()`

Generates version-appropriate Trainer kwargs:

```python
from src.utils.compat import get_trainer_init_kwargs

trainer_kwargs, tokenizer_to_store = get_trainer_init_kwargs(
    model=model,
    args=args,
    tokenizer=tokenizer,
    # ... other args
)

trainer = Trainer(**trainer_kwargs)

# Store tokenizer separately if needed (4.36+)
if tokenizer_to_store:
    trainer.tokenizer = tokenizer_to_store
```

**What it does**:
- For 4.35.x: Includes tokenizer in kwargs, returns None for tokenizer_to_store
- For 4.36+: Excludes tokenizer from kwargs, returns tokenizer to store separately

#### 3. `training_step_accepts_num_items()`

Checks if training_step should accept num_items_in_batch:

```python
from src.utils.compat import training_step_accepts_num_items

if training_step_accepts_num_items(trainer_class):
    # Use new signature with num_items_in_batch
else:
    # Use old signature without num_items_in_batch
```

**Solution**: Use default parameter for backwards compatibility:
```python
def training_step(self, model, inputs, num_items_in_batch=None):
    # Works with both versions
    ...
```

#### 4. Version Detection

```python
from src.utils.compat import (
    TRANSFORMERS_VERSION,
    TORCH_VERSION,
    TRANSFORMERS_4_36,
)

if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
    # Use new API
else:
    # Use old API
```

---

## Integration in Training Code

### scripts/train/train_sft.py

**Before (Hardcoded for 4.36+)**:
```python
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",  # Only works in 4.36+
    # ... other args
)
```

**After (Version-aware)**:
```python
from src.utils.compat import get_training_args_kwargs

kwargs = get_training_args_kwargs(
    output_dir=output_dir,
    eval_enabled=cfg.evaluation.do_eval,
    logging_dir=logging_dir,
    # ... other args
)

training_args = TrainingArguments(**kwargs)
```

### src/core/sft/trainer.py

**Before (Hardcoded for 4.36+)**:
```python
class SFTTrainer(Trainer):
    def __init__(self, tokenizer, ...):
        self.tokenizer = tokenizer  # Store separately
        super().__init__(model=model, args=args, ...)  # Don't pass tokenizer
```

**After (Version-aware)**:
```python
from ...utils.compat import get_trainer_init_kwargs

class SFTTrainer(Trainer):
    def __init__(self, tokenizer, ...):
        trainer_kwargs, tokenizer_to_store = get_trainer_init_kwargs(
            model=model,
            args=args,
            tokenizer=tokenizer,
            ...
        )

        if tokenizer_to_store:
            self.tokenizer = tokenizer_to_store

        super().__init__(**trainer_kwargs)
```

---

## Testing Compatibility

Run the compatibility test script:

```bash
python examples/test_version_compat.py
```

**Expected output on GPU platforms (transformers 4.36+)**:
```
=======================================================================
Library Versions
=======================================================================
transformers   : 4.36.0
torch          : 2.4.0
...
=======================================================================

API Compatibility:
  - Using transformers 4.36+ API
    * eval_strategy (not evaluation_strategy)
    * No tokenizer in Trainer.__init__()
    * training_step() with num_items_in_batch
    * TENSORBOARD_LOGGING_DIR env var (not logging_dir)

ALL COMPATIBILITY TESTS PASSED! ✅
```

**Expected output on macOS (transformers 4.35.x)**:
```
=======================================================================
Library Versions
=======================================================================
transformers   : 4.35.2
torch          : 2.0.1
...
=======================================================================

API Compatibility:
  - Using transformers <4.36 API
    * evaluation_strategy (not eval_strategy)
    * tokenizer in Trainer.__init__()
    * training_step() without num_items_in_batch
    * logging_dir parameter

ALL COMPATIBILITY TESTS PASSED! ✅
```

---

## Adding New Compatibility Checks

When you encounter new version-specific APIs:

1. **Add version check to `compat.py`**:
   ```python
   # New API change threshold
   TRANSFORMERS_4_40 = version.parse("4.40.0")

   def some_new_compat_helper():
       if TRANSFORMERS_VERSION >= TRANSFORMERS_4_40:
           # New API
       else:
           # Old API
   ```

2. **Use the helper in your code**:
   ```python
   from src.utils.compat import some_new_compat_helper

   result = some_new_compat_helper()
   ```

3. **Add test case to `test_version_compat.py`**:
   ```python
   # Test new compatibility check
   result = some_new_compat_helper()
   assert result == expected_value
   ```

4. **Document in this file** under "API Differences"

---

## Debugging Version Issues

### Print Version Info

```python
from src.utils.compat import print_version_info

print_version_info()
```

Output:
```
=======================================================================
Library Versions
=======================================================================
transformers   : 4.36.0
torch          : 2.4.0
python         : 3.10.12
platform       : Linux
=======================================================================

API Compatibility:
  - Using transformers 4.36+ API
    * eval_strategy (not evaluation_strategy)
    * No tokenizer in Trainer.__init__()
    * training_step() with num_items_in_batch
    * TENSORBOARD_LOGGING_DIR env var (not logging_dir)
=======================================================================
```

### Get Version Dict

```python
from src.utils.compat import get_version_info

info = get_version_info()
print(info)
# {'transformers': '4.36.0', 'torch': '2.4.0', ...}
```

---

## Summary

The compatibility layer ensures:
- ✅ **Single codebase** works across all platforms
- ✅ **Automatic detection** of installed versions
- ✅ **No manual version checks** in training code
- ✅ **Easy testing** with test script
- ✅ **Clear documentation** of API differences
- ✅ **Extensible** for future version changes

This approach allows the repository to support both:
- Educational use on macOS (transformers 4.35.x)
- Production training on GPU platforms (transformers 4.36+)

Without requiring users to modify code or maintain separate branches.
