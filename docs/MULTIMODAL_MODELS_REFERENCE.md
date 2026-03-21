# Multimodal Models Quick Reference

Quick reference for popular vision-language models and their integration status.

---

## Currently Supported

| Model | Type | HuggingFace ID | Status | Notes |
|-------|------|----------------|--------|-------|
| **CLIP** | Dual Encoder | `openai/clip-vit-base-patch32` | ✅ Supported | Baseline model |
| **LLaVA** | Generative | `llava-hf/llava-1.5-7b-hf` | ✅ Supported | Instruction following |

---

## CLIP-like Models (Easy to Add)

These models use similar architecture to CLIP and can often reuse existing code.

| Model | HuggingFace ID | Key Features | Integration Effort |
|-------|----------------|--------------|-------------------|
| **JinaClip** | `jinaai/jina-clip-v1` | Multilingual, high-res (512px) | 🟢 Low - Similar to CLIP |
| **SigLIP** | `google/siglip-base-patch16-224` | Sigmoid loss, more efficient | 🟢 Low - Similar to CLIP |
| **OpenCLIP** | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | Larger training data | 🟢 Low - Compatible API |
| **Chinese CLIP** | `OFA-Sys/chinese-clip-vit-base-patch16` | Chinese language support | 🟢 Low - CLIP architecture |
| **ALIGN** | `kakaobrain/align-base` | Google's image-text model | 🟡 Medium - Check API |
| **AltCLIP** | `BAAI/AltCLIP` | Multilingual CLIP | 🟢 Low - CLIP-based |

**Integration steps:**
1. Create config file (or reuse CLIP's)
2. Test with existing `CLIPWrapper`
3. Add to factory function if custom wrapper needed

---

## Generative Models (Medium Effort)

These models generate text from images, similar to LLaVA.

| Model | HuggingFace ID | Key Features | Integration Effort |
|-------|----------------|--------------|-------------------|
| **BLIP-2** | `Salesforce/blip2-opt-2.7b` | Q-Former bridge, efficient | 🟡 Medium - New wrapper needed |
| **InstructBLIP** | `Salesforce/instructblip-vicuna-7b` | Instruction-tuned BLIP-2 | 🟡 Medium - Similar to BLIP-2 |
| **GIT** | `microsoft/git-base` | Generative Image-to-Text | 🟡 Medium - Different architecture |
| **Flamingo** | Not on HF yet | Few-shot learning | 🔴 High - Complex architecture |
| **Kosmos-2** | `microsoft/kosmos-2-patch14-224` | Multimodal LLM | 🟡 Medium - Check compatibility |
| **Qwen-VL** | `Qwen/Qwen-VL-Chat` | Alibaba's VL model | 🟡 Medium - Custom components |

**Integration steps:**
1. Create custom wrapper class
2. Implement forward() and generate()
3. Add LoRA support
4. Create configs and test

---

## High-Resolution Models

Models optimized for high-resolution images.

| Model | HuggingFace ID | Max Resolution | Integration Effort |
|-------|----------------|----------------|-------------------|
| **JinaClip** | `jinaai/jina-clip-v1` | 512×512 | 🟢 Low |
| **CLIP ViT-L** | `openai/clip-vit-large-patch14` | 336×336 | 🟢 Low |
| **EVA-CLIP** | `QuPath/CLIP-ViT-H-14-laion2B-s32B-b79K` | 224×224 (but better) | 🟢 Low |

---

## Specialized Models

Models for specific domains or tasks.

| Model | Domain | HuggingFace ID | Integration Effort |
|-------|--------|----------------|-------------------|
| **BiomedCLIP** | Medical imaging | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 🟢 Low |
| **GeoRSCLIP** | Satellite imagery | Not widely available | 🟡 Medium |
| **FashionCLIP** | Fashion/e-commerce | Various finetuned versions | 🟢 Low |

---

## Model Characteristics Comparison

### CLIP vs Alternatives

| Feature | CLIP | JinaClip | SigLIP | BLIP-2 |
|---------|------|----------|--------|--------|
| **Architecture** | Dual encoder | Dual encoder | Dual encoder | Vision → Q-Former → LM |
| **Training Loss** | Contrastive (InfoNCE) | Contrastive | Sigmoid loss | Generative |
| **Max Resolution** | 224×224 | 512×512 | 224×224 | 224×224 |
| **Text Length** | 77 tokens | 77 tokens | 64 tokens | Variable |
| **Params (Base)** | 151M | ~150M | ~90M | 2.7B+ |
| **Generation** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Multilingual** | Limited | ✅ Yes | Limited | Depends on LM |
| **Training Speed** | Baseline | Similar | Faster | Slower |
| **Use Case** | General | Multilingual retrieval | Efficient retrieval | Captioning, VQA |

---

## Integration Difficulty Guide

### 🟢 Low Effort (1-2 hours)

**Requirements:**
- Model uses same API as CLIP or LLaVA
- Standard HuggingFace Transformers support
- No custom code requirements

**Models:**
- JinaClip, SigLIP, OpenCLIP (CLIP-like)
- Any model with `CLIPModel` class

**Process:**
1. Create config file
2. Update factory function (optional)
3. Test with existing wrapper

### 🟡 Medium Effort (4-8 hours)

**Requirements:**
- Custom architecture but standard components
- May need `trust_remote_code=True`
- Different forward() signature

**Models:**
- BLIP-2, InstructBLIP, GIT
- Models with Q-Former or custom bridges

**Process:**
1. Create new wrapper class
2. Implement forward(), generate()
3. Add LoRA support
4. Create configs
5. Test and validate

### 🔴 High Effort (1-3 days)

**Requirements:**
- Complex multi-stage architecture
- Custom preprocessing
- Not on HuggingFace or needs heavy modifications

**Models:**
- Flamingo (not available)
- Custom research models
- Models with non-standard APIs

**Process:**
1. Port model to HuggingFace format (if needed)
2. Create comprehensive wrapper
3. Custom data processors
4. Extensive testing
5. Documentation

---

## Recommended Models to Add

### Top Priorities

**1. JinaClip** 🟢
- **Why:** Better multilingual, higher resolution
- **Effort:** Low
- **Use case:** Improved retrieval, multilingual tasks

**2. SigLIP** 🟢
- **Why:** More efficient training, similar performance
- **Effort:** Low
- **Use case:** Faster training, efficiency research

**3. BLIP-2** 🟡
- **Why:** State-of-art captioning, smaller than LLaVA
- **Effort:** Medium
- **Use case:** Image captioning, VQA

### Domain-Specific

**4. BiomedCLIP** 🟢
- **Why:** Medical imaging applications
- **Effort:** Low
- **Use case:** Medical image retrieval

---

## Quick Start: Adding JinaClip

The fastest way to add a new model:

### Option 1: Reuse CLIP Wrapper (2 minutes)

**File:** `configs/model/jinaclip.yaml`
```yaml
name: "jinaclip"
architecture: "clip"  # Reuse existing wrapper
model_name_or_path: "jinaai/jina-clip-v1"
use_lora: true
```

**Usage:**
```bash
python scripts/train/train_multimodal.py \
    model=jinaclip \
    data=custom_image_caption \
    data.train_file=/path/to/data.json
```

### Option 2: Custom Wrapper (1-2 hours)

See `docs/ADDING_NEW_MULTIMODAL_MODELS.md` for complete guide.

---

## Testing Checklist

Before considering a model "integrated":

- [ ] Model loads successfully
- [ ] Forward pass works
- [ ] Training completes (short run)
- [ ] Checkpoints save/load correctly
- [ ] Config files created
- [ ] Basic documentation added
- [ ] Example usage provided

**Test command:**
```bash
python scripts/train/train_multimodal.py \
    model=YOUR_MODEL \
    data.max_train_samples=10 \
    +training.max_steps=5 \
    training.num_epochs=null
```

---

## Common Issues

### Issue: "Model not found"
**Solution:** Check HuggingFace model hub, verify model ID

### Issue: "trust_remote_code required"
**Solution:** Add `trust_remote_code=True` in from_pretrained()

### Issue: "Incompatible processor"
**Solution:** Use `AutoProcessor` instead of `CLIPProcessor`

### Issue: "LoRA fails"
**Solution:** Check target modules match actual model structure

---

## Resources

- **Full Integration Guide:** `docs/ADDING_NEW_MULTIMODAL_MODELS.md`
- **Existing Wrappers:** `src/models/vision_language.py`
- **HuggingFace Hub:** https://huggingface.co/models?pipeline_tag=image-text-to-text
- **Model Cards:** Check individual model documentation

---

## Contributing

Added support for a new model? Please:

1. **Test thoroughly** using checklist above
2. **Document** in this file and main guide
3. **Submit PR** with:
   - Wrapper code
   - Config files
   - Test results
   - Usage example

---

## Questions?

- Check `docs/ADDING_NEW_MULTIMODAL_MODELS.md` for detailed guide
- Review existing wrappers for examples
- Open GitHub issue with model details
