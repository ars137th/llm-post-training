# Python Practices and Patterns in This Repository

A comprehensive guide to the Python techniques, patterns, and best practices used throughout this codebase. Perfect for becoming reacquainted with modern Python development.

## Table of Contents
- [Type Hints and the typing Module](#type-hints-and-the-typing-module)
- [The @property Decorator](#the-property-decorator)
- [Class Design and OOP](#class-design-and-oop)
- [Dataclasses and Configuration](#dataclasses-and-configuration)
- [Factory Functions and **kwargs](#factory-functions-and-kwargs)
- [Context Managers](#context-managers)
- [Decorators](#decorators)
- [Modern Python Features](#modern-python-features)
- [Documentation Patterns](#documentation-patterns)

---

## Type Hints and the typing Module

Type hints make code self-documenting and enable static analysis tools like mypy to catch bugs before runtime.

### Basic Type Hints

**Location**: Throughout codebase, especially `src/models/language.py`

```python
from typing import Dict, List, Optional, Union, Tuple

def prepare_data(
    dataset: List[Dict[str, str]],
    max_length: int = 512,
    format_fn: Optional[callable] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Args:
        dataset: List of dictionaries with text data
        max_length: Maximum sequence length (default: 512)
        format_fn: Optional formatting function (can be None)
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    pass
```

**Key types used:**
- `List[T]`: List containing items of type T
- `Dict[K, V]`: Dictionary with keys of type K and values of type V  
- `Optional[T]`: Either T or None (shorthand for `Union[T, None]`)
- `Union[A, B]`: Either type A or type B
- `Tuple[A, B, C]`: Tuple with specific types at each position

### Example: LanguageModel Type Hints

**File**: `src/models/language.py:40-80`

```python
class LanguageModel:
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],  # Can be either type
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,  # Optional: defaults to None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```

**Why this matters:**
- IDEs provide autocomplete and catch errors
- Readers immediately know what types to pass
- `Union[PreTrainedModel, PeftModel]` documents that we support both base and PEFT models

### Complex Type Hints

**File**: `src/core/dpo/loss.py:68-75`

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Returns:
        If return_details=False: Loss tensor (scalar)
        If return_details=True: (loss, details_dict)
    """
```

**Union return type** documents that the return value changes based on parameters.

### Generic Types

**File**: `src/data/processors/preference.py`

```python
from typing import Dict, List, Optional, Callable

def create_preference_dataset(
    examples: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> List[Dict[str, torch.Tensor]]:
    """
    Transform list of string dicts into list of tensor dicts.
    
    Input:  List[Dict[str, str]]       # Strings
    Output: List[Dict[str, Tensor]]    # Tensors
    """
    pass
```

### Type Aliases for Clarity

```python
# Create readable aliases for complex types
PreferenceExample = Dict[str, str]
TokenizedExample = Dict[str, torch.Tensor]
Metrics = Dict[str, float]

def process_examples(
    examples: List[PreferenceExample]
) -> List[TokenizedExample]:
    """Much more readable than Dict[str, str]!"""
    pass
```

---

## The @property Decorator

`@property` makes methods look like attributes, providing clean access to computed values.

### Basic @property

**File**: `src/models/language.py:394-402`

```python
class LanguageModel:
    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
```

**Usage:**
```python
model = LanguageModel.from_pretrained("gpt2")

# Call like an attribute, not a method
print(f"Parameters: {model.num_parameters:,}")  # Not model.num_parameters()
print(f"Trainable: {model.num_trainable_parameters:,}")
```

**Why @property?**
- ✅ Cleaner syntax: `model.num_parameters` vs `model.get_num_parameters()`
- ✅ Computed on demand (not stored, always fresh)
- ✅ Looks like an attribute but can have complex logic
- ✅ Can add validation or caching later without changing interface

### Property with Computation

**File**: `src/models/reward.py` (conceptual example)

```python
class RewardModel:
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        # Computed by checking where parameters are
        return next(self.model.parameters()).device
    
    @property
    def percent_trainable(self) -> float:
        """Percentage of parameters that are trainable."""
        total = self.num_parameters
        trainable = self.num_trainable_parameters
        return (trainable / total * 100) if total > 0 else 0.0
```

### Property with Caching (Advanced)

```python
from functools import cached_property

class LanguageModel:
    @cached_property
    def config_dict(self) -> Dict:
        """
        Expensive computation, cached after first access.
        Recomputed if object changes.
        """
        return self.model.config.to_dict()
```

**When to use @property:**
- ✅ Computed values (parameters, device, metrics)
- ✅ Values that might change (model might move devices)
- ✅ Read-only attributes (no setter)
- ❌ Expensive operations (use `@cached_property` or regular method)
- ❌ When you need arguments (use regular method)

---

## Class Design and OOP

### Wrapper Pattern

**File**: `src/models/language.py:40-80`

The `LanguageModel` class **wraps** HuggingFace models to provide a unified interface.

```python
class LanguageModel:
    """
    Wrapper around PreTrainedModel that adds:
    - Unified interface for all model types
    - LoRA/PEFT integration
    - Convenience methods for RLHF
    - Clean property access
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model          # Wrapped object
        self.tokenizer = tokenizer
        self.is_peft_model = isinstance(model, PeftModel)
    
    def forward(self, *args, **kwargs):
        """Delegate to wrapped model."""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Allow wrapper to be called like the original model."""
        return self.forward(*args, **kwargs)
```

**Pattern**: Composition over inheritance
- We **have** a model (composition) rather than **are** a model (inheritance)
- Provides flexibility to swap implementations
- Clean separation of concerns

### Class Method for Construction

**File**: `src/models/language.py:81-180`

```python
class LanguageModel:
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        use_lora: bool = False,
        **kwargs,
    ) -> "LanguageModel":
        """
        Factory method for creating instances.
        
        Note: Returns 'LanguageModel' (string annotation for forward reference)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        
        # Apply LoRA if requested
        if use_lora:
            model = get_peft_model(model, lora_config)
        
        # Return instance of our wrapper class
        return cls(model=model, tokenizer=tokenizer)
```

**Why @classmethod?**
- ✅ Alternative constructors (like `from_pretrained`, `from_config`)
- ✅ Factory pattern - complex creation logic
- ✅ Returns instance of the class: `cls(...)`
- ✅ Works with inheritance (subclasses get correct type)

**Usage:**
```python
# Clean construction without __init__ complexity
model = LanguageModel.from_pretrained("gpt2", use_lora=True)
```

### Inheritance: Extending HuggingFace Trainer

**File**: `src/core/dpo/trainer.py:23-117`

```python
from transformers import Trainer

class DPOTrainer(Trainer):
    """
    Extend Trainer to add DPO-specific behavior.
    
    Inherits: All standard training infrastructure
    Overrides: Loss computation and evaluation
    Adds: DPO-specific logging and metrics
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,  # DPO-specific parameter
        beta: float = 0.1,            # DPO-specific parameter
        **kwargs,                     # Pass rest to parent
    ):
        # Call parent constructor
        super().__init__(model=model, **kwargs)
        
        # Add DPO-specific attributes
        self.ref_model = ref_model
        self.beta = beta
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override parent method to compute DPO loss.
        
        Parent would compute standard language modeling loss.
        We compute preference-based DPO loss instead.
        """
        # Custom DPO logic here
        pass
```

**When to use inheritance:**
- ✅ Extending existing classes (like Trainer)
- ✅ "Is-a" relationship (DPOTrainer **is a** Trainer)
- ✅ Want to override specific methods while keeping others
- ❌ Complex multiple inheritance (prefer composition)
- ❌ Just for code reuse (use composition)

### Data Classes (Configuration Objects)

**Pattern used implicitly via Hydra, explicit example:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # Required fields
    beta: float
    loss_type: str
    
    # Optional fields with defaults
    learning_rate: float = 5e-7
    num_epochs: int = 1
    log_rewards: bool = True
    num_rewards_to_log: int = 5
    
    def __post_init__(self):
        """Validation after initialization."""
        if self.beta <= 0:
            raise ValueError("Beta must be positive")
        if self.loss_type not in ["dpo", "ipo"]:
            raise ValueError("loss_type must be 'dpo' or 'ipo'")
```

**Usage:**
```python
config = DPOConfig(beta=0.1, loss_type="dpo")
print(config.learning_rate)  # 5e-7 (default)
```

**Why dataclasses?**
- ✅ Automatic `__init__`, `__repr__`, `__eq__`
- ✅ Type hints built-in
- ✅ Less boilerplate than manual classes
- ✅ Immutable option with `@dataclass(frozen=True)`

---

## Factory Functions and **kwargs

Factory functions are a common pattern for creating objects when you have multiple similar but slightly different variations. The `**kwargs` pattern allows flexible argument passing.

### What is a Factory Function?

**A factory function creates and returns objects**, often choosing which specific class or configuration to use based on parameters.

**File**: `src/data/collators/multimodal.py:416-477`

```python
def create_multimodal_collator(
    model_type: str,
    tokenizer: PreTrainedTokenizer,
    image_processor: Any,
    **kwargs,
):
    """
    Factory function to create appropriate collator for model type.

    Returns:
        CLIPDataCollator, LLaVADataCollator, or MultimodalDataCollator
    """
    if model_type.lower() == "clip":
        # CLIP only accepts: max_length, padding
        clip_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'padding']
        }
        return CLIPDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **clip_kwargs,
        )
    elif model_type.lower() == "llava":
        # LLaVA accepts: max_length, instruction_template, ignore_index
        llava_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'instruction_template', 'ignore_index']
        }
        return LLaVADataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **llava_kwargs,
        )
    else:
        return MultimodalDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **kwargs,
        )
```

**Usage:**
```python
# User doesn't need to know which collator class to use
collator = create_multimodal_collator(
    model_type="clip",  # Factory chooses CLIPDataCollator
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
    max_length=77,
)

# Different model type → different collator
llava_collator = create_multimodal_collator(
    model_type="llava",  # Factory chooses LLaVADataCollator
    tokenizer=llava_processor.tokenizer,
    image_processor=llava_processor.image_processor,
    instruction_template="Describe this image:",
    max_length=512,
)
```

### Why Use Factory Functions?

**1. Hide Implementation Details**

Users don't need to know:
- Which specific class to instantiate
- Import paths for each class
- Differences between class constructors

```python
# Without factory: Users must know implementation details
from src.data.collators.multimodal import CLIPDataCollator, LLaVADataCollator

if model_type == "clip":
    collator = CLIPDataCollator(tokenizer, image_processor, max_length=77)
else:
    collator = LLaVADataCollator(
        tokenizer,
        image_processor,
        max_length=512,
        instruction_template="Describe:"
    )

# With factory: Simple and clean
collator = create_multimodal_collator(
    model_type=model_type,
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_length=max_length,
)
```

**2. Single Point of Change**

When you add a new model type, users' code doesn't change:

```python
# Factory updated to support new "blip" model
def create_multimodal_collator(model_type, ...):
    if model_type == "clip":
        return CLIPDataCollator(...)
    elif model_type == "llava":
        return LLaVADataCollator(...)
    elif model_type == "blip":  # New!
        return BLIPDataCollator(...)

# User code unchanged:
collator = create_multimodal_collator(model_type="blip", ...)
```

**3. Validation and Configuration**

Factory can enforce rules and apply defaults:

```python
def create_vision_language_model(
    model_type: str,
    model_name: str,
    use_lora: bool = False,
    device: str = "auto",
):
    # Validation
    if model_type not in ["clip", "llava"]:
        raise ValueError(f"Unsupported model: {model_type}")

    # Device logic
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create appropriate wrapper
    if model_type == "clip":
        return CLIPWrapper(model_name, use_lora, device)
    else:
        return LLaVAWrapper(model_name, use_lora, device)
```

### Understanding **kwargs

`**kwargs` collects **keyword arguments** into a dictionary.

**Basic Example:**

```python
def print_info(name, age, **kwargs):
    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"Other info: {kwargs}")

# Call with extra arguments
print_info(
    name="Alice",
    age=30,
    city="NYC",       # Goes into kwargs
    occupation="Engineer"  # Goes into kwargs
)

# Output:
# Name: Alice
# Age: 30
# Other info: {'city': 'NYC', 'occupation': 'Engineer'}
```

**The `**` operator:**
- **Packing** (in function definition): `**kwargs` collects extra keyword args into a dict
- **Unpacking** (in function call): `**some_dict` expands dict into keyword arguments

### **kwargs Packing

**Collect extra arguments:**

```python
def train_model(model, data, **training_args):
    """
    training_args catches: learning_rate, num_epochs, batch_size, etc.
    """
    print(f"Model: {model}")
    print(f"Training args: {training_args}")

train_model(
    model="gpt2",
    data="wikitext",
    learning_rate=1e-5,  # Goes into training_args
    num_epochs=3,        # Goes into training_args
    batch_size=16,       # Goes into training_args
)

# training_args = {
#     'learning_rate': 1e-5,
#     'num_epochs': 3,
#     'batch_size': 16
# }
```

### **kwargs Unpacking

**Pass dict contents as keyword arguments:**

```python
def train(learning_rate, num_epochs, batch_size):
    print(f"LR: {learning_rate}, Epochs: {num_epochs}, Batch: {batch_size}")

# Have a config dict
config = {
    'learning_rate': 1e-5,
    'num_epochs': 3,
    'batch_size': 16,
}

# Unpack dict into keyword arguments
train(**config)
# Equivalent to: train(learning_rate=1e-5, num_epochs=3, batch_size=16)
```

### Real-World Example: Filtering kwargs

**Problem**: Factory function receives kwargs that not all classes accept.

**File**: `src/data/collators/multimodal.py:445-455`

```python
def create_multimodal_collator(
    model_type: str,
    tokenizer: PreTrainedTokenizer,
    image_processor: Any,
    **kwargs,  # Receives: max_length, padding, instruction_template, etc.
):
    if model_type == "clip":
        # CLIPDataCollator only accepts: max_length, padding
        # Filter out instruction_template, ignore_index, etc.
        clip_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'padding']
        }
        return CLIPDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **clip_kwargs,  # Unpack filtered kwargs
        )
    elif model_type == "llava":
        # LLaVADataCollator accepts different parameters
        llava_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'instruction_template', 'ignore_index']
        }
        return LLaVADataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **llava_kwargs,
        )
```

**Why filter?**

Without filtering:
```python
# User calls factory with all possible options
collator = create_multimodal_collator(
    model_type="clip",
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_length=77,
    instruction_template="Describe:",  # Not valid for CLIP!
)

# Without filtering, this would error:
# TypeError: CLIPDataCollator.__init__() got unexpected keyword argument 'instruction_template'

# With filtering, instruction_template is silently ignored for CLIP
```

### Dictionary Comprehension (kwargs filtering)

**Syntax**: `{key: value for key, value in dict.items() if condition}`

```python
# Original kwargs
kwargs = {
    'max_length': 77,
    'padding': 'max_length',
    'instruction_template': 'Describe:',  # Not wanted
    'ignore_index': -100,  # Not wanted
}

# Filter to only include max_length and padding
clip_kwargs = {
    k: v for k, v in kwargs.items()
    if k in ['max_length', 'padding']
}

# Result:
# clip_kwargs = {'max_length': 77, 'padding': 'max_length'}
```

**Step by step:**
1. `kwargs.items()` → [('max_length', 77), ('padding', 'max_length'), ...]
2. For each `(k, v)` pair
3. If `k in ['max_length', 'padding']` → include in new dict
4. Otherwise → skip

### Combining *args and **kwargs

**Pattern**: Accept any positional and keyword arguments

```python
def flexible_function(*args, **kwargs):
    """
    *args: Tuple of positional arguments
    **kwargs: Dict of keyword arguments
    """
    print(f"Positional: {args}")
    print(f"Keyword: {kwargs}")

flexible_function(1, 2, 3, name="Alice", age=30)
# Positional: (1, 2, 3)
# Keyword: {'name': 'Alice', 'age': 30}
```

**Real example** - Wrapper function:

```python
def logged_train(*args, **kwargs):
    """Wrapper that adds logging to any train function."""
    print("Starting training...")
    result = train(*args, **kwargs)  # Pass everything through
    print("Training complete!")
    return result

# Works with any train function signature:
logged_train(model, data, learning_rate=1e-5)
logged_train(model, data, epochs=3, batch_size=16)
```

### When to Use Factories and **kwargs

**Use factory functions when:**
- ✅ Multiple related classes with similar purpose but different implementations
- ✅ Users shouldn't need to know which class to use
- ✅ Configuration/selection logic is complex
- ✅ You want a single import point

**Examples in this repo:**
- `create_multimodal_collator()` - Choose collator based on model type
- `create_vision_language_model()` - Choose CLIP vs LLaVA wrapper
- `create_optimizer()` - Choose Adam, AdamW, SGD, etc.

**Use **kwargs when:**
- ✅ Function has many optional parameters
- ✅ Forwarding arguments to another function
- ✅ Building flexible APIs
- ✅ Config-based function calls

**Examples in this repo:**
- Factory functions passing options to constructors
- Trainer wrappers forwarding to HuggingFace Trainer
- Config-driven training scripts

**Avoid when:**
- ❌ Function has only a few well-defined parameters (explicit is better)
- ❌ You want IDE autocomplete (kwargs lose type hints)
- ❌ Debugging is important (kwargs hide argument flow)

### Common Pitfall: Mutable Default Arguments

**DON'T DO THIS:**

```python
def create_collator(tokenizer, options={}):  # ❌ BAD!
    options['tokenizer'] = tokenizer
    return options

# Problem: Same dict reused across calls!
c1 = create_collator(tok1)
c2 = create_collator(tok2)
# c1 and c2 share the SAME dict!
```

**DO THIS:**

```python
def create_collator(tokenizer, options=None):  # ✅ GOOD
    if options is None:
        options = {}
    options['tokenizer'] = tokenizer
    return options

# Each call gets a fresh dict
```

### Summary: Factory Functions

**Factory Pattern:**
```python
# Factory decides which class to instantiate
def create_thing(type, **kwargs):
    if type == "A":
        return ThingA(**kwargs)
    elif type == "B":
        return ThingB(**kwargs)
    else:
        raise ValueError(f"Unknown type: {type}")

# Users don't need to know about ThingA vs ThingB
thing = create_thing("A", option1=True, option2=42)
```

**Benefits:**
- 🎯 Single point of entry
- 🔒 Encapsulation (hide implementation)
- 🔧 Easy to extend (add new types)
- 📝 Self-documenting (type string explains choice)

**kwargs Patterns:**
```python
# 1. Collect extras
def func(required, **extras):
    pass

# 2. Filter and forward
def factory(**kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return SomeClass(**filtered)

# 3. Pass through wrapper
def wrapper(*args, **kwargs):
    return underlying_func(*args, **kwargs)
```

---

## Context Managers

### Using Context Managers

**File**: `scripts/train/train_dpo.py:366-368`

```python
from hydra import initialize_config_dir

# Context manager ensures cleanup
with initialize_config_dir(config_dir=CONFIGS_PATH):
    cfg = compose(config_name="config_dpo")
    main(cfg)
# Hydra automatically cleaned up here
```

### No-Gradient Context

**File**: `src/core/dpo/trainer.py:152-164`

```python
# Forward pass for reference model (no gradients needed)
with torch.no_grad():
    ref_outputs_chosen = self.ref_model(
        input_ids=inputs['chosen_input_ids'],
        attention_mask=inputs['chosen_attention_mask'],
    )
```

**Why `with torch.no_grad()`:**
- ✅ Disables gradient tracking (saves memory)
- ✅ Faster inference
- ✅ Automatically re-enables gradients after block

### File Operations

```python
# Automatically closes file, even if exception occurs
with open("output.txt", "w") as f:
    f.write("data")
# File guaranteed closed here
```

### Creating Custom Context Managers

```python
from contextlib import contextmanager

@contextmanager
def temporary_seed(seed: int):
    """Temporarily set random seed."""
    state = torch.get_rng_state()  # Save state
    torch.manual_seed(seed)
    try:
        yield  # Code block runs here
    finally:
        torch.set_rng_state(state)  # Restore state

# Usage
with temporary_seed(42):
    random_data = torch.randn(10)
# Original seed restored here
```

---

## Decorators

### Function Decorators

**File**: `src/models/language.py` (conceptual)

```python
from functools import wraps
import time

def timing_decorator(func):
    """Measure and print function execution time."""
    @wraps(func)  # Preserves original function metadata
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@timing_decorator
def train_model(model, data):
    """Training function with automatic timing."""
    pass
```

### Method Decorators

```python
class Model:
    @staticmethod
    def normalize(tensor):
        """
        Static method - doesn't need self or cls.
        Pure utility function within class namespace.
        """
        return (tensor - tensor.mean()) / tensor.std()
    
    @classmethod  
    def from_config(cls, config):
        """
        Class method - receives cls, not self.
        Alternative constructor.
        """
        return cls(**config)
    
    @property
    def device(self):
        """Property - accessed like attribute."""
        return self._device
```

**Decorator types:**
- `@staticmethod`: No self/cls, just namespacing
- `@classmethod`: Receives class, for alternate constructors
- `@property`: Makes method look like attribute
- Custom decorators: Add behavior (timing, logging, caching)

---

## Modern Python Features

### F-Strings (Formatted String Literals)

**File**: `scripts/train/train_dpo.py` (throughout)

```python
# Old way
print("Accuracy: " + str(accuracy) + "%")

# Modern way
print(f"Accuracy: {accuracy:.2%}")
print(f"Loss: {loss:.4f}")
print(f"Parameters: {num_params:,}")  # Thousands separator

# Multi-line f-strings
message = (
    f"Training complete!\n"
    f"  Accuracy: {accuracy:.2%}\n"
    f"  Loss: {loss:.4f}"
)
```

### Type Union with |  (Python 3.10+)

```python
# Old syntax
from typing import Union
def func(x: Union[int, float]) -> Union[str, None]:
    pass

# New syntax (Python 3.10+)
def func(x: int | float) -> str | None:
    pass
```

**Note**: We use older syntax for compatibility with Python 3.9+

### Dictionary Unpacking

**File**: `src/models/language.py:140-150`

```python
def load_model(model_name: str, **model_kwargs):
    """Accept arbitrary keyword arguments."""
    load_kwargs = {
        "trust_remote_code": True,
        **model_kwargs,  # Unpack and merge
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,  # Unpack as keyword arguments
    )
```

### Walrus Operator := (Assignment in Expression)

```python
# Old way
value = compute_expensive_value()
if value > threshold:
    process(value)

# New way (Python 3.8+)
if (value := compute_expensive_value()) > threshold:
    process(value)
```

### Pattern Matching (Python 3.10+)

```python
match loss_type:
    case "dpo":
        return dpo_loss(...)
    case "ipo":
        return ipo_loss(...)
    case _:
        raise ValueError(f"Unknown loss type: {loss_type}")
```

---

## Documentation Patterns

### Docstring Format (Google Style)

**File**: `src/core/dpo/loss.py:68-108`

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Compute DPO loss.
    
    DPO Loss = -log(σ(β * (log(π_θ/π_ref)[y_w] - log(π_θ/π_ref)[y_l])))
    
    Where:
    - π_θ = policy model (being trained)
    - π_ref = reference model (frozen)
    - y_w = chosen/winner response
    - y_l = rejected/loser response
    - β = temperature parameter controlling strength of KL constraint
    - σ = sigmoid function
    
    Args:
        policy_chosen_logps: Log probs from policy for chosen responses [batch_size]
        policy_rejected_logps: Log probs from policy for rejected responses [batch_size]
        reference_chosen_logps: Log probs from reference for chosen responses [batch_size]
        reference_rejected_logps: Log probs from reference for rejected responses [batch_size]
        beta: Temperature parameter (default: 0.1)
        return_details: Whether to return detailed metrics
    
    Returns:
        If return_details=False: Loss tensor (scalar)
        If return_details=True: (loss, details_dict)
    
    Example:
        >>> # Forward pass through policy and reference
        >>> policy_chosen_lp = compute_sequence_log_probs(policy_logits_chosen, labels_chosen)
        >>> reference_chosen_lp = compute_sequence_log_probs(ref_logits_chosen, labels_chosen)
        >>> # ... same for rejected
        >>> loss = dpo_loss(policy_chosen_lp, policy_rejected_lp,
        ...                 reference_chosen_lp, reference_rejected_lp)
    """
    pass
```

**Docstring sections:**
1. **Summary**: One-line description
2. **Extended Description**: Algorithm explanation, formulas
3. **Args**: Parameter descriptions with types and defaults
4. **Returns**: Return value description
5. **Example**: Usage example (optional but helpful)
6. **Raises**: Exceptions that might be raised (optional)

### Type Hints in Docstrings

```python
def process_batch(
    data: List[Dict[str, Any]],  # Type hint here
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Process data in batches.
    
    Args:
        data: List of dictionaries  # No need to repeat type
        batch_size: Batch size for processing
    
    Returns:
        Processed tensor
    """
```

**Best practice**: Put types in signature, descriptions in docstring.

### Module Docstrings

**File**: `src/core/dpo/loss.py:1-9`

```python
"""
DPO (Direct Preference Optimization) Loss Functions

Implements the DPO loss for training language models directly from preference data
without needing a separate reward model or complex RL.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
           Rafailov et al., 2023 (https://arxiv.org/abs/2305.18290)
"""
```

---

## Common Patterns in This Repo

### 1. Configuration via Hydra

**Pattern**: Hierarchical configuration with type safety

```python
# configs/config_dpo.yaml
defaults:
  - model: gpt2
  - technique: dpo
  - data: preference

# Access in code
from omegaconf import DictConfig

def main(cfg: DictConfig):
    beta = cfg.technique.beta  # Type-safe nested access
    model_name = cfg.model.name
```

### 2. Lazy Loading

**Pattern**: Only load heavy objects when needed

```python
class LanguageModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # Not loaded yet
    
    @property
    def model(self):
        """Load model on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

### 3. Builder Pattern (for Complex Objects)

```python
class DPOTrainer:
    @classmethod
    def build(
        cls,
        policy_model,
        ref_model,
        config: DictConfig,
    ):
        """Build trainer from config."""
        return cls(
            model=policy_model,
            ref_model=ref_model,
            beta=config.technique.beta,
            loss_type=config.technique.loss_type,
            # ... many more parameters
        )
```

### 4. Optional Dependencies

```python
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
def log_metrics(metrics):
    if WANDB_AVAILABLE:
        wandb.log(metrics)
    else:
        print("wandb not available, skipping logging")
```

### 5. Device Agnostic Code

```python
def train(model, data, device=None):
    # Auto-detect device if not specified
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    for batch in data:
        # Move data to same device as model
        batch = {k: v.to(device) for k, v in batch.items()}
```

---

## Quick Reference

### When to Use What

| Feature | Use When | Example |
|---------|----------|---------|
| `@property` | Computed read-only value | `model.num_parameters` |
| `@classmethod` | Alternative constructor | `Model.from_pretrained()` |
| `@staticmethod` | Utility in class namespace | `Model.normalize()` |
| Type hints | Always! | `def func(x: int) -> str:` |
| `Optional[T]` | Value can be None | `device: Optional[str] = None` |
| `Union[A, B]` | Value can be multiple types | `Union[int, float]` |
| Dataclass | Configuration objects | `@dataclass class Config:` |
| Context manager | Resource management | `with open(...) as f:` |
| f-strings | String formatting | `f"Loss: {loss:.4f}"` |
| `**kwargs` | Variable keyword args | `def func(**kwargs):` |

### Code Style

Follow PEP 8 with these conventions:
- **Indentation**: 4 spaces
- **Line length**: 88 characters (Black formatter)
- **Imports**: Standard library, third-party, local (grouped)
- **Naming**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- **Quotes**: Prefer double quotes for strings

### Testing Your Understanding

Try to understand these snippets from the repo:

1. **LanguageModel wrapper**:
```python
# Why is this a good design?
class LanguageModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

2. **DPO loss function**:
```python
# What does the return type tell you?
def dpo_loss(...) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    if return_details:
        return loss, details
    return loss
```

3. **Configuration access**:
```python
# Why is this better than dict['technique']['beta']?
beta = cfg.technique.beta
```

---

## Further Reading

**Python Official Docs:**
- Type hints: https://docs.python.org/3/library/typing.html
- Dataclasses: https://docs.python.org/3/library/dataclasses.html
- Decorators: https://docs.python.org/3/glossary.html#term-decorator

**Style Guides:**
- PEP 8: https://pep8.org/
- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html

**Books:**
- "Fluent Python" by Luciano Ramalho
- "Effective Python" by Brett Slatkin

**Tools:**
- mypy: Static type checker
- black: Code formatter
- pylint: Linter

---

## Practice Exercises

To solidify your understanding, try:

1. **Add type hints** to a function in the repo that doesn't have them
2. **Create a @property** for a computed value in a class
3. **Write a docstring** in Google style for a new function
4. **Use a context manager** for a resource that needs cleanup
5. **Create a dataclass** for a configuration object

This repository uses modern Python idiomatically. Understanding these patterns will help you navigate and contribute to the codebase effectively!
