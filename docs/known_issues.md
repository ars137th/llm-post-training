# Known Issues and Workarounds

## LoRA + CLIP Training Issue (Known Limitation)

### ⚠️ Recommendation: Don't Use LoRA with CLIP

**Default behavior:** LoRA is **disabled** for CLIP in `configs/experiment/clip_image_caption.yaml`

**Why:** PEFT has fundamental incompatibility with CLIP's dual-encoder architecture. Even partial LoRA (text encoder only) causes training errors.

**Alternative:** Train CLIP without LoRA. CLIP-ViT-B/32 is small enough (~150M params) to train on modest hardware.

```yaml
# configs/experiment/clip_image_caption.yaml
model:
  use_lora: false  # Recommended for CLIP
```

**Training command:**
```bash
python scripts/train/train_multimodal.py experiment=clip_image_caption
# LoRA is disabled by default in config, so this works out of the box
```

**TL;DR:** Multiple workarounds were attempted (separate encoder calls, isolated inputs, text-encoder-only LoRA, manual projection calls) but none work reliably. The PEFT library's kwargs routing is fundamentally incompatible with CLIP's architecture. Train without LoRA instead.

### Problem Details
When training CLIP with LoRA using HuggingFace's PEFT library, there are fundamental kwargs routing issues:

**Errors encountered:**
```
TypeError: CLIPVisionTransformer.forward() got an unexpected keyword argument 'input_ids'
TypeError: CLIPTextTransformer.forward() got an unexpected keyword argument 'inputs_embeds'
```

**Root Cause:**
1. CLIP's architecture has separate `vision_model` and `text_model` submodules
2. CLIP's main `forward()` method internally routes:
   - `pixel_values` → `vision_model.forward()`
   - `input_ids` → `text_model.forward()`
3. When PEFT wraps these submodules with LoRA adapters, it wraps each submodule's `forward()`
4. However, PEFT's wrapper receives ALL kwargs from the parent call
5. This causes `input_ids` to be passed to `vision_model.forward()`, which doesn't accept it

**Stack trace pattern:**
```
model(pixel_values=..., input_ids=...)
  → vision_model(pixel_values=..., input_ids=...)  # PEFT wrapper receives all kwargs
    → CLIPVisionTransformer.forward(...)  # Rejects input_ids
```

### Solution 1: Use Separate Encoder Methods with Isolated Inputs (Recommended)

Instead of calling `model.forward(pixel_values=..., input_ids=...)`, call the encoder methods separately with completely isolated input dictionaries:

```python
# ✓ Correct approach - use isolated input dicts with cloned tensors
# This prevents PEFT from seeing unrelated kwargs during adapter routing

image_inputs = {'pixel_values': inputs['pixel_values'].clone()}
image_embeds = model.get_image_features(**image_inputs)

text_inputs = {'input_ids': inputs['input_ids'].clone()}
if 'attention_mask' in inputs:
    text_inputs['attention_mask'] = inputs['attention_mask'].clone()
text_embeds = model.get_text_features(**text_inputs)

# Compute loss manually
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
logit_scale = model.logit_scale.exp()
logits_per_image = logit_scale * (image_embeds @ text_embeds.t())
loss = contrastive_loss(logits_per_image)
```

**Key points:**
- Create completely separate dictionaries for image and text inputs
- Clone tensors to ensure no shared references
- Use `**dict` unpacking to pass only the intended kwargs
- Never pass the full `inputs` dict directly to avoid context contamination

This is implemented in:
- `src/core/sft/multimodal_trainer.py` - `MultimodalSFTTrainer.compute_loss()` (lines 119-130)
- `src/models/vision_language.py` - `CLIPWrapper` methods

### Solution 2: Use CLIPWrapper Instead of Raw Model

The `CLIPWrapper` class provides clean methods that avoid this issue:

```python
from src.models.vision_language import create_vision_language_model

# Create CLIP with LoRA
clip_wrapper = create_vision_language_model(
    model_type="clip",
    model_name="openai/clip-vit-base-patch32",
    use_lora=True,
    lora_config={'r': 8, 'lora_alpha': 16},
)

# Use wrapper methods (handles LoRA correctly)
image_embeds = clip_wrapper.encode_image(images)  # ✓ Works
text_embeds = clip_wrapper.encode_text(texts)      # ✓ Works
similarity = clip_wrapper.compute_similarity(images, texts)  # ✓ Works
```

**When to use each approach:**

| Approach | Use Case |
|----------|----------|
| `MultimodalSFTTrainer` | Full training with HuggingFace Trainer infrastructure |
| `CLIPWrapper` methods | Custom training loops, inference, evaluation |
| Standard `Trainer` | Text-only models, LLaVA (no separate encoders) |

### Solution 3: Apply LoRA to Text Encoder Only (Current Workaround)

Due to PEFT's kwargs routing issues, we apply LoRA to **only the text encoder**, leaving the vision encoder frozen:

```python
# This is now the default behavior in src/models/vision_language.py
clip_model = create_vision_language_model(
    model_type="clip",
    model_name="openai/clip-vit-base-patch32",
    use_lora=True,  # Only text encoder gets LoRA
    lora_config={'r': 8, 'lora_alpha': 16},
)

# Output:
# ✓ LoRA applied to CLIP text encoder only (r=8)
#   Vision encoder: frozen (full precision)
#   Text encoder: LoRA adapters (trainable)
```

**Pros:**
- ✓ Avoids PEFT routing bug entirely
- ✓ Still provides memory savings (text encoder is ~50% of CLIP)
- ✓ Text encoder typically needs more adaptation for domain-specific captions
- ✓ Training works reliably

**Cons:**
- Vision encoder not trainable (unless using full fine-tuning)
- Less flexible than training both encoders

**To enable vision encoder LoRA (not recommended):**
```python
lora_config = {
    'r': 8,
    'lora_alpha': 16,
    'apply_to_vision_encoder': True,  # ⚠️  May cause training errors
}
```

### Solution 4: Train Without LoRA (Testing Only)

For quick tests where memory isn't an issue:

```python
clip_model = create_vision_language_model(
    model_type="clip",
    model_name="openai/clip-vit-base-patch32",
    use_lora=False,  # No LoRA = no PEFT wrapper = no issue
)
```

This is only suitable for:
- Unit tests
- Small models
- Systems with lots of GPU memory

**Not recommended for production** as CLIP models are relatively large and LoRA provides significant memory savings.

### Affected Components

**Working correctly:**
- ✓ `src/models/vision_language.py` - CLIPWrapper methods
- ✓ `src/core/sft/multimodal_trainer.py` - Uses separate encoder calls
- ✓ `src/evaluation/metrics/multimodal.py` - Uses get_image_features/get_text_features
- ✓ `scripts/train/train_multimodal.py` - Uses MultimodalSFTTrainer (when fixed)

**Workaround in tests:**
- `tests/test_multimodal_pipeline.py` - Uses `use_lora=False` for quick testing

### Future Fix

This issue may be resolved in future PEFT versions by:
1. Better kwargs filtering in PEFT wrappers
2. Explicit parameter routing in CLIP's forward pass
3. PEFT hooks that respect model architecture

Track: https://github.com/huggingface/peft/issues/...

---

## Other Known Issues

### Issue: Bus Error on macOS During Training Tests

**Problem:**
```
zsh: bus error  python tests/test_multimodal_pipeline.py
resource_tracker: There appear to be 1 leaked semaphore objects
```

**Cause:**
- macOS has stricter fork safety rules than Linux (see: `docs/PLATFORM_COMPATIBILITY.md`)
- HuggingFace Trainer uses multiprocessing internally beyond just DataLoader workers
- PyTorch operations can trigger fork-unsafe behavior on macOS
- Setting `dataloader_num_workers=0` fixes DataLoader issues but not Trainer internals
- This is a known macOS + PyTorch + transformers limitation

**Solution:**
Training scripts **automatically detect macOS** and apply workarounds:

```python
# scripts/train/train_multimodal.py automatically does this:
if platform.system() == "Darwin":
    logger.warning("⚠️  macOS detected - applying fork safety workarounds")
    # Sets dataloader_num_workers=0
    # Disables periodic evaluation (eval_steps)
    # Evaluation runs once at the end instead
```

**For macOS users - training now works automatically:**

```bash
# This now works on macOS (with automatic workarounds)
python scripts/train/train_multimodal.py experiment=clip_image_caption

# Or use macOS-specific config for explicit control
python scripts/train/train_multimodal.py experiment=clip_image_caption_macos
```

**What the script does on macOS:**
1. ✅ Sets `dataloader_num_workers=0` (no multiprocessing in DataLoader)
2. ✅ Disables `dataloader_pin_memory` (not useful on CPU)
3. ✅ Disables periodic evaluation (`eval_steps=null`)
4. ✅ Runs evaluation once at the end of training
5. ✅ Disables fp16 (CPU doesn't support it)

**For serious training, use Google Colab** (recommended):
- Free T4 GPU (10-20x faster)
- No macOS limitations
- See: `docs/google_colab_guide.md` for detailed setup

**What works on macOS:**
- ✅ Model loading
- ✅ Data processing
- ✅ Evaluation metrics
- ✅ CLIP inference
- ✅ **Production training scripts** (with auto-workarounds)
- ✅ Small-scale experiments and development
- ❌ Only unit test training loop has issues

**Related Documentation:**
- `docs/PLATFORM_COMPATIBILITY.md` - Full platform guide
- `docs/VERSION_COMPATIBILITY.md` - macOS typically uses transformers 4.35.x

**Note:** This only affects automated unit tests. Production training scripts work correctly on macOS when properly configured.

---

### Issue: Tied Weights Warning During Reward Model Saving

**Problem:**
```
UserWarning: Tensor has shared memory with other tensors. Clone may not behave as expected.
RuntimeError: Cannot save shared parameters that require gradient.
```

**Cause:** Language models often tie input and output embeddings (weight sharing).

**Solution:** See `docs/troubleshooting/reward_model_saving.md`

---

### Issue: YAML Config Ordering Sensitivity

**Problem:** Config values not being overridden as expected in Hydra experiments.

**Cause:** Hydra processes configs in order. The `_self_` directive controls when the current file is merged.

**Solution:** See `docs/hydra_config_guide.md` - Section on "Config Composition Order"

---

### Issue: Dataset Still Synthetic After Config Change

**Problem:** Training uses synthetic data even after setting `data.dataset_name=anthropic`.

**Cause:** Multiple config files with different merge orders.

**Solution:**
- Place `_self_` at the end of `defaults` list in experiment configs
- Use debug script to verify: `python configs/debug_config.py`
- See: `docs/troubleshooting/config_ordering.md`
