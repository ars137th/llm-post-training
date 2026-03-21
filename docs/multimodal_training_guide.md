# Multimodal Training Guide

Complete guide to training vision-language models (CLIP, LLaVA) with LoRA.

**Training on Google Colab?** See the dedicated [Google Colab Guide](google_colab_guide.md) for platform-specific optimizations, GPU recommendations, and memory management tips.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Setup and Downloads](#model-setup-and-downloads)
3. [Architecture Overview](#architecture-overview)
4. [Training Components](#training-components)
5. [LoRA + CLIP Implementation](#lora--clip-implementation)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)
8. [Platform-Specific Guides](#platform-specific-guides)

---

## Quick Start

### Train CLIP with SFT on Synthetic Data

```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption
```

### Train CLIP with LoRA on COCO

```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=coco \
    data.max_train_samples=50000 \
    model.use_lora=true \
    model.lora_config.r=8
```

### Train LLaVA (7B with 4-bit quantization)

```bash
python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    model.use_4bit=true \
    model.use_lora=true
```

### Train CLIP with DPO (Preference Learning)

```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo
```

### Train LLaVA with DPO

```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=llava_dpo \
    model.use_4bit=true
```

---

## Model Setup and Downloads

### How Model Loading Works

This repository uses **HuggingFace Hub** for model management. Models are automatically downloaded on first use and cached locally.

**No manual downloads needed!** Just run the training script and models will be fetched automatically.

### Supported Models

#### CLIP Models

**Model identifier**: `openai/clip-vit-base-patch32`

```python
from src.models.vision_language import create_vision_language_model

# First run: Downloads model (~600MB) and caches it
# Subsequent runs: Uses cached model (instant)
clip_model = create_vision_language_model(
    model_type="clip",
    model_name="openai/clip-vit-base-patch32",  # HuggingFace model ID
    use_lora=True,
    device="cuda",
)
```

**What happens on first run:**
1. Connects to HuggingFace Hub
2. Downloads model weights (~600MB for CLIP-ViT-B/32)
3. Caches in `~/.cache/huggingface/hub/`
4. Loads model into memory

**Subsequent runs:**
- Skips download (uses cached files)
- Loads directly from local cache (fast)

**Available CLIP variants:**
- `openai/clip-vit-base-patch32` (151M params, 224x224) - **Default**
- `openai/clip-vit-base-patch16` (151M params, 224x224, higher res patches)
- `openai/clip-vit-large-patch14` (428M params, 224x224, more capacity)

#### LLaVA Models

**Model identifier**: `llava-hf/llava-1.5-7b-hf`

```python
# First run: Downloads 7B model (~14GB with fp16, ~4GB with 4-bit)
llava_model = create_vision_language_model(
    model_type="llava",
    model_name="llava-hf/llava-1.5-7b-hf",  # HuggingFace model ID
    use_4bit=True,  # Reduces download and memory (4GB vs 14GB)
    use_lora=True,
    device="cuda",
)
```

**Available LLaVA variants:**
- `llava-hf/llava-1.5-7b-hf` (7B params, Vicuna-7B base) - **Default**
- `llava-hf/llava-1.5-13b-hf` (13B params, better quality, more VRAM)
- `llava-hf/bakLlava-v1-hf` (7B params, LLaMA-2 base)

### Cache Location

**Default cache**: `~/.cache/huggingface/hub/`

**Check cache size:**
```bash
du -sh ~/.cache/huggingface/
```

**Clear cache (if needed):**
```bash
rm -rf ~/.cache/huggingface/hub/
# Models will re-download on next use
```

**Custom cache location:**
```bash
export HF_HOME="/path/to/custom/cache"
python scripts/train/train_multimodal.py experiment=clip_image_caption
```

### Offline Usage

**Download models ahead of time:**

```python
from transformers import CLIPModel, CLIPProcessor

# Download and cache CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Now available offline
```

**Or use HuggingFace CLI:**
```bash
pip install huggingface_hub

# Download CLIP
huggingface-cli download openai/clip-vit-base-patch32

# Download LLaVA (large!)
huggingface-cli download llava-hf/llava-1.5-7b-hf
```

### Private/Custom Models

**Use your own fine-tuned model:**

```bash
# 1. Train and save a model
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.output_dir=./my_clip_model

# 2. Use the saved model (no download)
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    model.model_name_or_path=./my_clip_model  # Local path
```

**Use from HuggingFace Hub (your account):**

```python
# Upload your model first (requires authentication)
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./my_clip_model",
    repo_id="your-username/my-clip-model",
)

# Then use it
clip_model = create_vision_language_model(
    model_type="clip",
    model_name="your-username/my-clip-model",  # Your HF model
)
```

### Authentication (for private models)

**If you need to access private HuggingFace models:**

```bash
# Option 1: Login via CLI (saves token)
huggingface-cli login

# Option 2: Set token as environment variable
export HF_TOKEN="hf_your_token_here"
```

**In Python:**
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Network Requirements

**Internet connection needed for:**
- First download of each model
- Accessing private models (authentication)

**Internet NOT needed for:**
- Training with cached models
- Inference with cached models
- Using local model paths

### Model Storage Space

| Model | Size (fp16) | Size (4-bit) | Cache Location |
|-------|-------------|--------------|----------------|
| CLIP-ViT-B/32 | ~600 MB | N/A | `~/.cache/huggingface/` |
| CLIP-ViT-L/14 | ~1.7 GB | N/A | `~/.cache/huggingface/` |
| LLaVA-1.5-7B | ~14 GB | ~4 GB | `~/.cache/huggingface/` |
| LLaVA-1.5-13B | ~26 GB | ~7 GB | `~/.cache/huggingface/` |

**Recommendation:** Use 4-bit quantization for LLaVA models to save disk space and memory.

### Quick Reference

**Training with default CLIP (auto-downloads):**
```bash
python scripts/train/train_multimodal.py experiment=clip_image_caption
# Downloads openai/clip-vit-base-patch32 on first run
```

**Training with LLaVA (auto-downloads with 4-bit):**
```bash
python scripts/train/train_multimodal.py experiment=llava_instruction
# Downloads llava-hf/llava-1.5-7b-hf (4-bit version, ~4GB)
```

**Training with local model (no download):**
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    model.model_name_or_path=/path/to/my/model
```

---

## Architecture Overview

### CLIP (Contrastive Language-Image Pre-training)

```
Image ──→ Vision Encoder ──→ Image Embedding ──┐
                                                ├──→ Cosine Similarity ──→ Contrastive Loss
Text  ──→ Text Encoder   ──→ Text Embedding  ──┘
```

**Key Points:**
- Dual encoder architecture (separate vision and text encoders)
- Learns joint embedding space for images and text
- Training: Maximize similarity for matching pairs, minimize for non-matching
- LoRA applied to attention layers in both encoders

### LLaVA (Large Language and Vision Assistant)

```
Image ──→ CLIP Vision Encoder ──→ Visual Features ──┐
                                                     ├──→ LLaMA/Vicuna ──→ Response
Instruction ──→ Tokenizer ──→ Text Tokens ───────────┘
```

**Key Points:**
- Combines pre-trained CLIP vision encoder with LLaMA/Vicuna LLM
- Vision encoder usually frozen, LLM fine-tuned
- Training: Causal language modeling (predict response given image + instruction)
- LoRA applied to LLM attention layers

---

## Training Components

### 1. Model Wrappers

**`src/models/vision_language.py`**

```python
from src.models.vision_language import create_vision_language_model

# Create CLIP with LoRA
clip_model = create_vision_language_model(
    model_type="clip",
    model_name="openai/clip-vit-base-patch32",
    use_lora=True,
    lora_config={
        'r': 8,              # LoRA rank
        'lora_alpha': 16,    # LoRA scaling
        'target_modules': ['q_proj', 'v_proj'],  # Which layers to adapt
    },
    device="cuda",
)

# CLIPWrapper provides clean API
image_embeds = clip_model.encode_image(images)
text_embeds = clip_model.encode_text(texts)
similarity = clip_model.compute_similarity(images, texts)
```

### 2. Data Processing

**`src/data/processors/multimodal.py`**

```python
from src.data.processors.multimodal import MultimodalDataProcessor

processor = MultimodalDataProcessor()

# Load real datasets
coco_examples = processor.load_coco_captions(split="train", num_examples=10000)
flickr_examples = processor.load_flickr30k(split="train", num_examples=5000)

# Generate synthetic data (for testing)
synthetic_examples = processor.create_synthetic_data(num_examples=100)

# Create instruction format (for LLaVA)
instruction_data = processor.create_instruction_data(
    examples,
    instruction_template="Describe this image in detail:"
)

# Create preference pairs (for reward modeling/DPO)
preference_pairs = processor.create_preference_pairs(
    examples,
    augment_negatives=True  # Generate negative examples
)
```

### 3. Data Collators

**`src/data/collators/multimodal.py`**

```python
from src.data.collators.multimodal import create_multimodal_collator
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLIP collator: batches image+text pairs
clip_collator = create_multimodal_collator(
    model_type="clip",
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
)

# LLaVA collator: batches image+instruction+response
llava_collator = create_multimodal_collator(
    model_type="llava",
    tokenizer=tokenizer,
    image_processor=image_processor,
    instruction_template="Describe this image:",
)
```

### 4. Trainers

#### Supervised Fine-Tuning (SFT)

**`src/core/sft/multimodal_trainer.py`**

```python
from src.core.sft.multimodal_trainer import MultimodalSFTTrainer

trainer = MultimodalSFTTrainer(
    model=clip_model.model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    model_type="clip",  # or "llava"
)

trainer.train()
```

**Why MultimodalSFTTrainer?**
- Handles LoRA + CLIP kwargs routing issue
- Computes correct loss for each model type:
  - CLIP: Contrastive loss
  - LLaVA: Causal language modeling loss
- Logs multimodal-specific metrics

#### Direct Preference Optimization (DPO)

**`src/core/dpo/multimodal_trainer.py`**

```python
from src.core.dpo.multimodal_trainer import MultimodalDPOTrainer
from src.data.collators.multimodal import MultimodalDPODataCollator

# Load policy and reference models
policy_model = create_vision_language_model(...)
ref_model = create_vision_language_model(...)  # Frozen copy

# Create DPO data collator for preference pairs
dpo_collator = MultimodalDPODataCollator(
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
    max_length=77,
)

# Create DPO trainer
trainer = MultimodalDPOTrainer(
    model=policy_model.model,
    ref_model=ref_model.model,
    args=training_args,
    train_dataset=preference_dataset,  # Must contain chosen/rejected pairs
    data_collator=dpo_collator,
    tokenizer=processor.tokenizer,
    model_type="clip",  # or "llava"
    beta=0.1,  # DPO temperature
)

trainer.train()
```

**Why MultimodalDPOTrainer?**
- Learns from preference pairs without explicit reward model
- Two approaches:
  - **CLIP:** Uses contrastive similarity as implicit reward
  - **LLaVA:** Uses causal LM log probabilities
- More stable than PPO, simpler pipeline
- Lower learning rate required (5e-7 vs 5e-6 for SFT)

### 5. Evaluation Metrics

**`src/evaluation/metrics/multimodal.py`**

```python
from src.evaluation.metrics.multimodal import CLIPScoreMetric, ImageTextRetrievalMetric

# CLIP Score: measures image-text alignment
clip_metric = CLIPScoreMetric()
scores = clip_metric.compute(images, captions)
print(f"CLIP Score: {scores['clip_score']:.2f}")

# Retrieval: measures recall@K
retrieval_metric = ImageTextRetrievalMetric()
results = retrieval_metric.compute(images, texts, k_values=[1, 5, 10])
print(f"Text→Image R@1: {results['t2i_recall@1']:.1%}")
print(f"Image→Text R@1: {results['i2t_recall@1']:.1%}")
```

---

## LoRA + CLIP Implementation

### The Challenge

CLIP has separate `vision_model` and `text_model` submodules. When PEFT wraps these with LoRA:

```python
# ✗ Naive approach (fails with LoRA)
outputs = model(pixel_values=..., input_ids=...)
# Error: CLIPVisionTransformer.forward() got unexpected keyword argument 'input_ids'
```

**Why?** PEFT's wrapper receives ALL kwargs from the parent scope, but vision encoder only accepts `pixel_values`.

### The Solution (Text Encoder Only)

**Current workaround:** Apply LoRA to **text encoder only**, leave vision encoder frozen.

```python
# Default behavior in src/models/vision_language.py
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

**Why this works:**
- Avoids PEFT kwargs routing bug entirely
- Still provides memory savings (~50% of CLIP is text encoder)
- Text encoder typically needs more domain adaptation
- Training works reliably

**Training loop:** Call encoder methods separately:

```python
# ✓ Correct approach (in MultimodalSFTTrainer)
image_embeds = model.get_image_features(pixel_values=pixel_values)
text_embeds = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

# Compute contrastive loss manually
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

logit_scale = model.logit_scale.exp()
logits_per_image = logit_scale * (image_embeds @ text_embeds.t())

# Contrastive loss (cross-entropy with diagonal targets)
labels = torch.arange(batch_size, device=image_embeds.device)
loss_i = F.cross_entropy(logits_per_image, labels)
loss_t = F.cross_entropy(logits_per_image.t(), labels)
loss = (loss_i + loss_t) / 2
```

This is implemented in:
- `src/core/sft/multimodal_trainer.py` → `compute_loss()` method
- `src/models/vision_language.py` → `CLIPWrapper` methods

### Implementation Details

**File: `src/core/sft/multimodal_trainer.py`**

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if self.model_type == "clip":
        # CLIP: Call encoders separately (avoids LoRA routing issues)
        vision_outputs = model.get_image_features(pixel_values=inputs['pixel_values'])
        text_outputs = model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
        )

        # Create outputs object
        class CLIPOutputs:
            def __init__(self, image_embeds, text_embeds, logit_scale):
                self.image_embeds = image_embeds
                self.text_embeds = text_embeds
                self.logit_scale = logit_scale

        logit_scale = model.logit_scale.exp()
        outputs = CLIPOutputs(vision_outputs, text_outputs, logit_scale)

        # Compute contrastive loss
        loss = self._compute_clip_loss(outputs, inputs)

    else:  # llava
        # LLaVA: Standard forward pass works fine
        outputs = model(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            labels=inputs.get('labels'),
        )
        loss = outputs.loss

    return (loss, outputs) if return_outputs else loss
```

---

## Usage Examples

### Example 1: Train CLIP from Scratch

```bash
# Synthetic data (quick test)
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.num_epochs=3 \
    training.per_device_train_batch_size=16

# COCO Captions (production)
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=coco \
    data.max_train_samples=50000 \
    data.max_eval_samples=5000 \
    model.use_lora=true \
    model.lora_config.r=8 \
    training.num_epochs=10 \
    training.per_device_train_batch_size=32 \
    training.learning_rate=1e-5
```

### Example 2: Fine-tune CLIP for Specific Domain

```bash
# Medical images
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=medical_images \  # Custom dataset
    model.use_lora=true \
    model.lora_config.r=16 \  # Higher rank for more capacity
    training.learning_rate=5e-6  # Lower LR for fine-tuning
```

### Example 3: Train LLaVA on Instructions

```bash
python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    data.dataset_name=coco \
    data.use_instruction_format=true \
    data.instruction_template="Describe this image in detail:" \
    model.use_4bit=true \  # 4-bit quantization for 7B model
    model.use_lora=true \
    model.freeze_vision_encoder=true \  # Keep CLIP encoder frozen
    training.per_device_train_batch_size=2 \
    training.gradient_accumulation_steps=8  # Effective batch size = 16
```

### Example 4: Evaluate Trained Model

```bash
# CLIP model
python scripts/evaluate/evaluate_multimodal.py \
    --model_path ./outputs/clip_caption \
    --model_type clip \
    --dataset coco \
    --num_examples 1000

# LLaVA model
python scripts/evaluate/evaluate_multimodal.py \
    --model_path ./outputs/llava_instruction \
    --model_type llava \
    --dataset coco \
    --num_examples 100
```

### Example 5: Train CLIP with DPO on Preference Pairs

```bash
# Synthetic preferences (quick test)
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    training.num_epochs=1 \
    training.per_device_train_batch_size=4

# COCO with real preferences
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data.dataset_name=coco \
    data.max_train_samples=10000 \
    model.use_lora=true \
    model.lora_config.r=8 \
    technique.beta=0.1 \
    training.learning_rate=5e-7
```

### Example 6: Train LLaVA with DPO

```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=llava_dpo \
    data.dataset_name=synthetic \
    data.max_train_samples=200 \
    model.use_4bit=true \
    model.use_lora=true \
    technique.beta=0.1 \
    training.per_device_train_batch_size=1 \
    training.gradient_accumulation_steps=8
```

### Example 7: Custom Training Loop (SFT)

```python
from src.models.vision_language import create_vision_language_model
from src.data.processors.multimodal import MultimodalDataProcessor
import torch

# Setup
clip_model = create_vision_language_model(
    model_type="clip",
    model_name="openai/clip-vit-base-patch32",
    use_lora=True,
)

processor_obj = MultimodalDataProcessor()
examples = processor_obj.load_coco_captions(split="train", num_examples=1000)

optimizer = torch.optim.AdamW(clip_model.model.parameters(), lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images, texts = batch['image'], batch['text']

        # Use CLIPWrapper methods (handles LoRA correctly)
        image_embeds = clip_model.encode_image(images)
        text_embeds = clip_model.encode_text(texts)

        # Compute loss
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits = (image_embeds @ text_embeds.t()) * 100
        labels = torch.arange(len(images), device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Troubleshooting

### Issue: `CLIPVisionTransformer.forward() got unexpected keyword argument 'input_ids'`

**Cause:** Using standard `Trainer` with CLIP + LoRA.

**Solution:** Use `MultimodalSFTTrainer`:
```python
from src.core.sft.multimodal_trainer import MultimodalSFTTrainer

trainer = MultimodalSFTTrainer(
    model=model,
    args=training_args,
    model_type="clip",  # Important!
    ...
)
```

See: `docs/known_issues.md` - "LoRA + CLIP Training Issue"

### Issue: Out of memory when training LLaVA

**Solution 1:** Use 4-bit quantization
```yaml
model:
  use_4bit: true
  use_lora: true
```

**Solution 2:** Reduce batch size, increase gradient accumulation
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

**Solution 3:** Freeze vision encoder
```yaml
model:
  freeze_vision_encoder: true  # Only train LLM
```

### Issue: Low CLIP Score

**Possible causes:**
1. **Learning rate too high:** Try 1e-5 or lower
2. **Too few epochs:** CLIP needs substantial training
3. **Batch size too small:** CLIP benefits from larger batches (use gradient accumulation)
4. **LoRA rank too low:** Try r=16 or r=32 instead of r=8

### Issue: Synthetic data too simple

**Solution:** Use real datasets
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=coco  # or flickr30k
```

---

## Best Practices

### Memory Efficiency

1. **Use LoRA:** Reduces trainable parameters by 90%
   ```yaml
   model:
     use_lora: true
     lora_config:
       r: 8  # Start small, increase if needed
   ```

2. **Use quantization** (for 7B+ models):
   ```yaml
   model:
     use_4bit: true  # Reduces memory by ~4x
   ```

3. **Gradient accumulation:**
   ```yaml
   training:
     per_device_train_batch_size: 2
     gradient_accumulation_steps: 8  # Effective batch = 16
   ```

### Training Speed

1. **Use mixed precision:**
   ```yaml
   training:
     fp16: true  # or bf16 on Ampere GPUs
   ```

2. **Disable unnecessary logging:**
   ```yaml
   training:
     logging_steps: 50  # Log less frequently
   ```

3. **Use multiple workers:**
   ```yaml
   training:
     dataloader_num_workers: 4
   ```

### Hyperparameters

**CLIP:**
- Learning rate: 1e-5 to 5e-5
- Batch size: 32-256 (use gradient accumulation)
- Epochs: 5-30 depending on dataset size
- LoRA rank: 8-16 for fine-tuning, 16-32 for training from scratch

**LLaVA:**
- Learning rate: 2e-5
- Batch size: 2-8 per device (7B model)
- Gradient accumulation: 4-8 steps
- LoRA rank: 16-32
- Always use 4-bit quantization for 7B models

---

## Training Techniques Comparison

### Available: SFT and DPO

**Supervised Fine-Tuning (SFT):**
- Trains model on image-caption pairs
- Next-token prediction (LLaVA) or contrastive learning (CLIP)
- Simplest approach, good baseline
- Script: `scripts/train/train_multimodal.py`

**Direct Preference Optimization (DPO):**
- Learns from preference pairs (chosen vs rejected)
- No explicit reward model needed
- More stable than PPO, simpler pipeline
- Better alignment with human preferences
- Script: `scripts/train/train_multimodal_dpo.py`

### Future Work: PPO/RLHF

**Proximal Policy Optimization (PPO)** for multimodal models is planned for Phase 7:

**Why PPO for Multimodal?**
- Can optimize for complex rewards (image quality, caption accuracy, style)
- Explicit reward model provides interpretable feedback
- Can combine multiple reward signals (CLIP score + text quality + safety)

**Planned Implementation:**
- Reward model: Trained to score image-caption pairs
- Actor (policy): CLIP or LLaVA model being optimized
- Critic (value): Estimates expected rewards
- Reference model: Frozen copy for KL constraint

**Challenges:**
- More complex than DPO (4 models vs 2)
- Requires careful hyperparameter tuning
- Higher memory requirements
- Training instability (reward hacking, mode collapse)

**When to Use:**
- Complex reward functions (multi-objective optimization)
- Need interpretable reward signals
- Have computational resources for full RLHF pipeline
- Preference-based DPO insufficient for task

**Status:** Not yet implemented. Use DPO for preference learning in the meantime.

For text-only PPO, see: `docs/PPO_THEORY.md` and `src/core/ppo/trainer.py`

---

## Platform-Specific Guides

### Training on Google Colab

For detailed information about training multimodal models on Google Colab, including:
- GPU tier recommendations (Free, Pro, Pro+)
- LoRA compatibility issues on Colab
- Why CLIP training without LoRA works better on GPU
- Memory optimization strategies
- Session management and checkpointing
- Complete Colab notebook templates

See: **[Google Colab Guide](google_colab_guide.md)**

**Quick summary:**
- ✅ **CLIP**: Train without LoRA on GPU (fast and works)
- ❌ **CLIP + LoRA**: Same PEFT bugs on all platforms
- ✅ **LLaVA**: Use LoRA + 4-bit quantization (essential for 7B models)
- ✅ **Free tier**: Sufficient for CLIP and small models
- ✅ **Pro tier**: Recommended for LLaVA-7B

### Training on macOS

See: **[Platform Compatibility Guide](PLATFORM_COMPATIBILITY.md)**

**Key considerations:**
- CPU-only training (no Metal GPU support yet)
- Fork safety issues with multiprocessing
- Slower training but works for small models
- Recommended for development and testing

### Training on Linux/Windows with GPU

Standard CUDA training works out of the box. Follow Quick Start instructions.

---

## Adding New Multimodal Models

Want to train with **JinaClip**, **SigLIP**, **BLIP-2**, or other vision-language models?

The framework is designed to be extensible. See comprehensive guides:

### Quick Start: Add JinaClip (2 minutes)

**Create config:** `configs/model/jinaclip.yaml`
```yaml
name: "jinaclip"
architecture: "clip"  # Reuse CLIP wrapper
model_name_or_path: "jinaai/jina-clip-v1"
use_lora: true
```

**Train:**
```bash
python scripts/train/train_multimodal.py \
    model=jinaclip \
    data=custom_image_caption \
    data.train_file=/path/to/data.json
```

### Full Integration Guides

- **Complete Guide:** [Adding New Multimodal Models](ADDING_NEW_MULTIMODAL_MODELS.md)
  - Step-by-step integration process
  - CLIP-like models (JinaClip, SigLIP, OpenCLIP)
  - Generative models (BLIP-2, InstructBLIP)
  - Testing and troubleshooting

- **Quick Reference:** [Multimodal Models Reference](MULTIMODAL_MODELS_REFERENCE.md)
  - Popular models comparison table
  - Integration difficulty ratings
  - Model characteristics
  - Recommended additions

---

## Further Reading

- **Implementation Details:** `src/models/vision_language.py`, `src/core/sft/multimodal_trainer.py`, `src/core/dpo/multimodal_trainer.py`
- **Custom Data:** `docs/CUSTOM_DATA_GUIDE.md`
- **Known Issues:** `docs/known_issues.md`
- **Tutorial Notebook:** `notebooks/06_multimodal_training.ipynb`
- **DPO Theory:** `docs/DPO_THEORY.md`
- **PPO Theory (text-only):** `docs/PPO_THEORY.md`
- **CLIP Paper:** https://arxiv.org/abs/2103.00020
- **LLaVA Paper:** https://arxiv.org/abs/2304.08485
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **DPO Paper:** https://arxiv.org/abs/2305.18290
