# Adding New Multimodal Models

Guide for integrating new vision-language models like JinaClip, SigLIP, and others into the training framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Currently Supported Models](#currently-supported-models)
3. [Model Categories](#model-categories)
4. [Adding a CLIP-like Model](#adding-a-clip-like-model)
5. [Adding a Generative Model](#adding-a-generative-model)
6. [Step-by-Step Integration](#step-by-step-integration)
7. [Testing Your Integration](#testing-your-integration)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The repository's multimodal training infrastructure is designed to be extensible. This guide shows how to add support for new vision-language models while maintaining compatibility with existing training scripts, data loaders, and evaluation tools.

**What you'll learn:**
- How to wrap HuggingFace models for our framework
- Integration patterns for different model architectures
- Configuration setup
- Testing procedures

---

## Currently Supported Models

### CLIP (Contrastive)
- **Model:** `openai/clip-vit-base-patch32`
- **Architecture:** Dual encoder (vision + text)
- **Training:** Contrastive learning (image-text alignment)
- **Use cases:** Zero-shot classification, retrieval, embedding

### LLaVA (Generative)
- **Model:** `llava-hf/llava-1.5-7b-hf`
- **Architecture:** Vision encoder + Language model
- **Training:** Instruction following, caption generation
- **Use cases:** Visual question answering, image captioning

---

## Model Categories

### Category 1: CLIP-like Models (Dual Encoder)

**Architecture:**
- Separate vision and text encoders
- Outputs: Image embeddings, text embeddings
- Loss: Contrastive (InfoNCE)

**Examples:**
- **OpenAI CLIP:** `openai/clip-vit-base-patch32`
- **JinaClip:** `jinaai/jina-clip-v1`
- **SigLIP:** `google/siglip-base-patch16-224`
- **OpenCLIP:** `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- **Chinese CLIP:** `OFA-Sys/chinese-clip-vit-base-patch16`

**Key characteristics:**
- Symmetric training (image ↔ text)
- No generation capability
- Fast inference
- Good for retrieval tasks

### Category 2: Generative Models (Vision-to-Text)

**Architecture:**
- Vision encoder → Cross-attention → Language decoder
- Outputs: Text sequences
- Loss: Causal language modeling

**Examples:**
- **LLaVA:** `llava-hf/llava-1.5-7b-hf`
- **Flamingo:** (not yet in HF)
- **BLIP-2:** `Salesforce/blip2-opt-2.7b`
- **InstructBLIP:** `Salesforce/instructblip-vicuna-7b`
- **Kosmos:** `microsoft/kosmos-2-patch14-224`

**Key characteristics:**
- Generates text from images
- More parameters (includes LLM)
- Slower inference
- Better for complex reasoning

---

## Adding a CLIP-like Model

Let's add **JinaClip** as an example. The process is similar for SigLIP and other CLIP variants.

### Step 1: Create Model Wrapper

**File:** `src/models/vision_language.py`

Add a new wrapper class:

```python
class JinaCLIPWrapper(nn.Module):
    """
    Wrapper for Jina CLIP models.

    JinaClip is an improved CLIP variant with:
    - Better multilingual support
    - Higher resolution support (up to 512x512)
    - Improved training on diverse datasets

    Example:
        model = JinaCLIPWrapper.from_pretrained("jinaai/jina-clip-v1")
        outputs = model(pixel_values=images, input_ids=texts)
    """

    def __init__(
        self,
        model: CLIPModel,  # JinaClip uses same architecture as CLIP
        processor: AutoProcessor,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.use_lora = use_lora

        if use_lora and lora_config:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA to text encoder (same as CLIP)."""
        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 16),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "v_proj"]  # Standard attention projections
            ),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Apply to text encoder only (same pattern as CLIP)
        self.model.text_model = get_peft_model(self.model.text_model, peft_config)
        print(f"✓ LoRA applied to JinaCLIP text encoder (r={peft_config.r})")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "jinaai/jina-clip-v1",
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        device: str = "auto",
    ) -> "JinaCLIPWrapper":
        """Load pretrained JinaCLIP model."""
        print(f"Loading JinaCLIP model: {model_name}")

        # Load using standard HuggingFace transformers
        # JinaCLIP uses the same CLIPModel class
        from transformers import CLIPModel, AutoProcessor

        model = CLIPModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)

        # Move to device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        wrapper = cls(
            model=model,
            processor=processor,
            use_lora=use_lora,
            lora_config=lora_config,
        )

        print(f"✓ JinaCLIP loaded on {device}")
        return wrapper

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> VisionLanguageModelOutput:
        """Forward pass (identical to CLIP)."""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=return_loss,
            **kwargs,
        )

        return VisionLanguageModelOutput(
            loss=outputs.loss if return_loss else None,
            logits=outputs.logits_per_image,
            image_embeds=outputs.image_embeds,
            text_embeds=outputs.text_embeds,
        )

    # Inherit or copy these methods from CLIPWrapper:
    # - encode_image()
    # - encode_text()
    # - compute_similarity()
    # - save_pretrained()
    # - device property
    # - config property
```

**Note:** If JinaCLIP is 100% compatible with CLIP's API, you can simply reuse `CLIPWrapper` and just change the model loading.

### Step 2: Update Factory Function

Add to `create_vision_language_model()` in `src/models/vision_language.py`:

```python
def create_vision_language_model(
    model_type: str,
    model_name: str,
    use_lora: bool = False,
    lora_config: Optional[Dict] = None,
    device: str = "auto",
    **kwargs,
) -> Union[CLIPWrapper, LLaVAWrapper, JinaCLIPWrapper]:  # Add new type
    """Factory function to create vision-language models."""
    model_type = model_type.lower()

    if model_type == "clip":
        return CLIPWrapper.from_pretrained(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            device=device,
        )
    elif model_type == "jinaclip":  # Add new model type
        return JinaCLIPWrapper.from_pretrained(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            device=device,
        )
    elif model_type in ["llava", "llava-1.5"]:
        return LLaVAWrapper.from_pretrained(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported: 'clip', 'jinaclip', 'llava'"  # Update error message
        )
```

### Step 3: Create Model Configuration

**File:** `configs/model/jinaclip.yaml`

```yaml
# JinaCLIP Model Configuration
# High-resolution CLIP variant with multilingual support

# Model identification
name: "jinaclip"
architecture: "jinaclip"  # Used by create_vision_language_model()
model_name_or_path: "jinaai/jina-clip-v1"

# Model size and capabilities
hidden_size: 768
vision_resolution: 224  # JinaClip supports up to 512
text_max_length: 77

# LoRA configuration (recommended for fine-tuning)
use_lora: true
lora_config:
  r: 16
  lora_alpha: 32
  target_modules:
    - q_proj
    - v_proj
  lora_dropout: 0.05
  bias: "none"

# Quantization (for memory efficiency)
use_4bit: false
use_8bit: false

# Training settings
gradient_checkpointing: false
freeze_vision_encoder: false  # Set true to only train text encoder
freeze_text_encoder: false
```

### Step 4: Create Experiment Configuration

**File:** `configs/experiment/jinaclip_image_caption.yaml`

```yaml
# JinaCLIP Image-Caption Training Experiment
# Contrastive learning for image-text alignment

# Inherit base config
defaults:
  - /model: jinaclip
  - /data: coco_captions  # Or custom_image_caption
  - /training: default_multimodal
  - _self_

# Experiment metadata
experiment_name: "jinaclip_image_caption"
description: "Fine-tune JinaCLIP on image-caption pairs"

# Model overrides
model:
  use_lora: true
  lora_config:
    r: 8
    lora_alpha: 16

# Data settings
data:
  dataset_name: "coco"  # or "custom" for your data
  max_train_samples: 5000
  max_eval_samples: 1000
  image_size: 224  # JinaClip default

# Training hyperparameters
training:
  num_epochs: 10
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64
  learning_rate: 1e-5
  warmup_steps: 500
  fp16: true
  output_dir: "./outputs/jinaclip_finetuned"

# Logging
logging:
  use_tensorboard: true
  use_wandb: false
```

### Step 5: Update Training Script (if needed)

The existing `scripts/train/train_multimodal.py` should work without changes if your wrapper follows the same interface as CLIP. Verify by checking:

1. **Model type detection:** Script checks `cfg.model.architecture`
2. **Processor compatibility:** JinaCLIP processor should work like CLIP's
3. **Data collator:** Should handle pixel_values and input_ids

**If modifications needed:**

```python
# In train_multimodal.py, verify model type
model_arch = cfg.model.get('architecture', '').lower()
if model_arch not in ['clip', 'jinaclip', 'llava']:  # Add jinaclip
    raise ValueError(f"Unsupported model architecture: {model_arch}")

# Load processor
if model_arch in ["clip", "jinaclip"]:  # Group CLIP-like models
    from transformers import AutoProcessor  # JinaCLIP uses AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.model.model_name_or_path)
```

---

## Adding SigLIP

**SigLIP** (Sigmoid Loss for Language-Image Pre-training) is Google's improved CLIP variant.

### Key Differences from CLIP:

1. **Loss function:** Uses sigmoid loss instead of softmax (InfoNCE)
2. **No temperature parameter:** Logit scale is learned differently
3. **Better efficiency:** Trains faster with same performance

### Implementation:

```python
class SigLIPWrapper(nn.Module):
    """
    Wrapper for SigLIP models.

    SigLIP improves on CLIP with sigmoid loss for better efficiency.

    Example:
        model = SigLIPWrapper.from_pretrained("google/siglip-base-patch16-224")
    """

    def __init__(self, model, processor, use_lora=False, lora_config=None):
        super().__init__()
        self.model = model
        self.processor = processor
        # ... (similar to CLIP)

    @classmethod
    def from_pretrained(cls, model_name="google/siglip-base-patch16-224", **kwargs):
        from transformers import AutoModel, AutoProcessor

        # SigLIP uses AutoModel (not CLIPModel)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # ... rest of initialization

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        # SigLIP forward pass is similar to CLIP
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Return format may differ - check SigLIP's output structure
        return VisionLanguageModelOutput(
            logits=outputs.logits_per_image,
            image_embeds=outputs.image_embeds,
            text_embeds=outputs.text_embeds,
        )
```

**Configuration:** `configs/model/siglip.yaml`

```yaml
name: "siglip"
architecture: "siglip"
model_name_or_path: "google/siglip-base-patch16-224"

# SigLIP-specific settings
hidden_size: 768
vision_resolution: 224
text_max_length: 64  # SigLIP uses shorter text

use_lora: true
lora_config:
  r: 16
  lora_alpha: 32
```

---

## Adding a Generative Model

Example: Adding **BLIP-2**

### Step 1: Create Wrapper

```python
class BLIP2Wrapper(nn.Module):
    """
    Wrapper for BLIP-2 models.

    BLIP-2 uses Q-Former to bridge vision and language models.
    """

    def __init__(self, model, processor, use_lora=False, lora_config=None):
        super().__init__()
        self.model = model
        self.processor = processor

        if use_lora:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA to language model only."""
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "v_proj"]  # LM attention
            ),
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply to language model component
        self.model.language_model = get_peft_model(
            self.model.language_model,
            peft_config
        )

    @classmethod
    def from_pretrained(cls, model_name="Salesforce/blip2-opt-2.7b", **kwargs):
        from transformers import Blip2ForConditionalGeneration, AutoProcessor

        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            load_in_4bit=kwargs.get('load_in_4bit', False),
            load_in_8bit=kwargs.get('load_in_8bit', False),
        )
        processor = AutoProcessor.from_pretrained(model_name)

        return cls(model, processor, **kwargs)

    def forward(self, pixel_values=None, input_ids=None, labels=None, **kwargs):
        """Forward pass for training."""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            **kwargs
        )

        return VisionLanguageModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )

    def generate(self, images, prompts, max_new_tokens=50, **kwargs):
        """Generate captions."""
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt"
        )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
```

---

## Step-by-Step Integration

### Complete Checklist

**1. Research the Model**
- [ ] Check HuggingFace model card
- [ ] Understand architecture (dual encoder vs generative)
- [ ] Note input/output formats
- [ ] Check for special requirements (trust_remote_code, etc.)

**2. Create Model Wrapper**
- [ ] Create wrapper class in `src/models/vision_language.py`
- [ ] Implement `__init__`, `from_pretrained`, `forward`
- [ ] Add LoRA support (if applicable)
- [ ] Add utility methods (encode_image, encode_text for CLIP-like)
- [ ] Add generation method (for generative models)

**3. Update Factory Function**
- [ ] Add model type to `create_vision_language_model()`
- [ ] Update type hints
- [ ] Update error messages

**4. Create Configurations**
- [ ] Model config: `configs/model/{model_name}.yaml`
- [ ] Experiment config: `configs/experiment/{model_name}_*.yaml`
- [ ] Test with sample config loading

**5. Update Training Scripts (if needed)**
- [ ] Add model type to supported list
- [ ] Handle processor loading
- [ ] Verify data collator compatibility

**6. Test Integration**
- [ ] Unit test: Load model
- [ ] Unit test: Forward pass
- [ ] Integration test: Short training run
- [ ] Verify checkpoint saving/loading

**7. Documentation**
- [ ] Add to README model list
- [ ] Document any special considerations
- [ ] Add example usage
- [ ] Update this guide with lessons learned

---

## Testing Your Integration

### Test 1: Model Loading

```python
# test_jinaclip_loading.py
from src.models.vision_language import create_vision_language_model

def test_jinaclip_loading():
    """Test JinaCLIP model loads correctly."""
    model = create_vision_language_model(
        model_type="jinaclip",
        model_name="jinaai/jina-clip-v1",
        device="cpu"  # Use CPU for testing
    )

    assert model is not None
    assert model.model is not None
    assert model.processor is not None
    print("✓ Model loads successfully")

if __name__ == "__main__":
    test_jinaclip_loading()
```

### Test 2: Forward Pass

```python
def test_jinaclip_forward():
    """Test forward pass with dummy data."""
    import torch
    from PIL import Image
    import numpy as np

    model = create_vision_language_model(
        model_type="jinaclip",
        model_name="jinaai/jina-clip-v1",
        device="cpu"
    )

    # Create dummy inputs
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_text = "A test image"

    # Process inputs
    inputs = model.processor(
        images=[dummy_image],
        text=[dummy_text],
        return_tensors="pt",
        padding=True
    )

    # Forward pass
    outputs = model(**inputs, return_loss=True)

    assert outputs.loss is not None
    assert outputs.logits is not None
    print("✓ Forward pass works")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")

if __name__ == "__main__":
    test_jinaclip_forward()
```

### Test 3: Short Training Run

```bash
# Test with minimal training
python scripts/train/train_multimodal.py \
    experiment=jinaclip_image_caption \
    data.max_train_samples=10 \
    data.max_eval_samples=5 \
    +training.max_steps=5 \
    training.num_epochs=null \
    training.output_dir=./test_outputs/jinaclip_test
```

Expected output:
```
Loading JinaCLIP model: jinaai/jina-clip-v1
✓ JinaCLIP loaded on cuda
Loading data...
✓ Loaded 10 train examples, 5 eval examples
STARTING TRAINING
...
✓ Training complete!
```

---

## Common Patterns

### Pattern 1: CLIP-Compatible Models

If your model is API-compatible with CLIP (same inputs/outputs), you can reuse `CLIPWrapper`:

```python
# No new wrapper needed!
model = create_vision_language_model(
    model_type="clip",  # Use existing CLIP wrapper
    model_name="jinaai/jina-clip-v1",  # Different checkpoint
    use_lora=True
)
```

Just create a config file:

```yaml
# configs/model/jinaclip.yaml
name: "jinaclip"
architecture: "clip"  # Reuse CLIP wrapper
model_name_or_path: "jinaai/jina-clip-v1"
```

### Pattern 2: Models Requiring trust_remote_code

Some models need `trust_remote_code=True`:

```python
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True  # Required for custom code
)
```

**Security note:** Only use with trusted models.

### Pattern 3: Models with Custom Processors

If the processor has special requirements:

```python
processor = AutoProcessor.from_pretrained(
    model_name,
    use_fast=True,  # Use fast tokenizer
    do_resize=True,
    size={"height": 224, "width": 224},  # Custom size
)
```

### Pattern 4: Handling Different Output Formats

Models may return different output structures:

```python
def forward(self, **inputs):
    outputs = self.model(**inputs)

    # Adapt to our standard format
    return VisionLanguageModelOutput(
        loss=getattr(outputs, 'loss', None),
        logits=getattr(outputs, 'logits_per_image', outputs.logits),
        image_embeds=getattr(outputs, 'image_embeds', None),
        text_embeds=getattr(outputs, 'text_embeds', None),
    )
```

---

## Troubleshooting

### Issue 1: "Model not found on HuggingFace"

**Solution:** Verify model identifier:
```python
from transformers import AutoModel
AutoModel.from_pretrained("model-name", trust_remote_code=True)
```

### Issue 2: "Unexpected keyword argument"

**Cause:** Model doesn't support all CLIP's arguments

**Solution:** Filter kwargs in forward():
```python
def forward(self, pixel_values=None, input_ids=None, **kwargs):
    # Filter unsupported kwargs
    supported_kwargs = {'attention_mask', 'return_dict'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

    return self.model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        **filtered_kwargs
    )
```

### Issue 3: "LoRA application fails"

**Cause:** Model structure differs from CLIP

**Solution:** Find correct target modules:
```python
# Print model structure
print(model)

# Identify attention projections
# Look for modules like: .attention.query, .attention.key, .attention.value

target_modules = ["query", "value"]  # Adjust based on actual structure
```

### Issue 4: "Dimension mismatch in loss"

**Cause:** Different contrastive loss formulation

**Solution:** Implement custom loss if needed:
```python
def compute_contrastive_loss(self, image_embeds, text_embeds):
    """Custom loss for models with different formulations."""
    # Normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Compute similarity
    logits_per_image = image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    # Custom loss (e.g., sigmoid for SigLIP)
    labels = torch.arange(len(image_embeds), device=image_embeds.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss
```

---

## Summary

**Adding a new model requires:**

1. **Wrapper class** implementing standard interface
2. **Factory function** update
3. **Configuration files** for model and experiments
4. **Testing** to verify integration
5. **Documentation** of any special requirements

**Most CLIP-like models** (JinaClip, SigLIP, OpenCLIP) can reuse much of the existing `CLIPWrapper` code.

**Generative models** (BLIP-2, InstructBLIP) need custom wrappers but follow similar patterns to `LLaVAWrapper`.

---

## Next Steps

1. **Try adding JinaClip:** Follow the guide above
2. **Test thoroughly:** Use provided test scripts
3. **Document your addition:** Update README and this guide
4. **Share your work:** Submit PR with new model support
5. **Add evaluation:** Benchmark on standard datasets

---

## Additional Resources

- **HuggingFace Model Hub:** https://huggingface.co/models?pipeline_tag=image-text-to-text
- **CLIP Paper:** https://arxiv.org/abs/2103.00020
- **SigLIP Paper:** https://arxiv.org/abs/2303.15343
- **BLIP-2 Paper:** https://arxiv.org/abs/2301.12597
- **Existing wrappers:** `src/models/vision_language.py`

---

## Questions or Issues?

- Review existing wrappers in `src/models/vision_language.py`
- Check HuggingFace model documentation
- Open GitHub issue with details about the model you're adding
- Include any error messages and model information
