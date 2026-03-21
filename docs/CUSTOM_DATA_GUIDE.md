# Custom Data Guide: Image-Caption Training

Complete guide for training CLIP/LLaVA with your own image-caption data.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Formats](#data-formats)
3. [SFT Training Data](#sft-training-data)
4. [DPO Training Data](#dpo-training-data)
5. [Loading Your Data](#loading-your-data)
6. [Training Examples](#training-examples)
7. [Validation](#validation)
8. [Best Practices](#best-practices)

---

## Overview

This guide covers how to prepare and use your own image-caption data for:
- **SFT (Supervised Fine-Tuning)**: Image + caption pairs
- **DPO (Direct Preference Optimization)**: Image + chosen/rejected caption pairs

**Supported formats:**
- JSON (recommended)
- CSV
- JSONL (JSON Lines)
- Python dictionaries

---

## Data Formats

### Quick Reference

| Format | SFT | DPO | Best For |
|--------|-----|-----|----------|
| **JSON** | ✅ | ✅ | Small-medium datasets |
| **JSONL** | ✅ | ✅ | Large datasets (streaming) |
| **CSV** | ✅ | ✅ | Simple data, spreadsheets |
| **Python Dict** | ✅ | ✅ | Programmatic generation |

---

## SFT Training Data

SFT requires **image-caption pairs** where the model learns to align images with their descriptions.

### Format 1: JSON (Recommended)

**File:** `my_data/train.json`

```json
{
  "data": [
    {
      "image_path": "images/cat001.jpg",
      "caption": "A fluffy orange cat sitting on a windowsill"
    },
    {
      "image_path": "images/dog002.jpg",
      "caption": "A golden retriever playing in the park"
    },
    {
      "image_path": "images/sunset003.jpg",
      "caption": "Beautiful sunset over the ocean with orange and pink clouds"
    }
  ]
}
```

**Required fields:**
- `image_path`: Path to image (relative to data directory or absolute)
- `caption`: Text description of the image

**Optional fields:**
- `id`: Unique identifier
- `split`: "train", "val", or "test"
- `metadata`: Additional info (not used in training)

**Full example with optional fields:**
```json
{
  "data": [
    {
      "id": "img_001",
      "image_path": "/absolute/path/to/images/cat001.jpg",
      "caption": "A fluffy orange cat sitting on a windowsill",
      "split": "train",
      "metadata": {
        "photographer": "Jane Doe",
        "license": "CC-BY-4.0"
      }
    }
  ]
}
```

---

### Format 2: JSONL (JSON Lines)

**File:** `my_data/train.jsonl`

Each line is a separate JSON object:

```jsonl
{"image_path": "images/cat001.jpg", "caption": "A fluffy orange cat sitting on a windowsill"}
{"image_path": "images/dog002.jpg", "caption": "A golden retriever playing in the park"}
{"image_path": "images/sunset003.jpg", "caption": "Beautiful sunset over the ocean"}
```

**Advantages:**
- Streaming for large datasets
- Easy to append new data
- Memory efficient

---

### Format 3: CSV

**File:** `my_data/train.csv`

```csv
image_path,caption
images/cat001.jpg,"A fluffy orange cat sitting on a windowsill"
images/dog002.jpg,"A golden retriever playing in the park"
images/sunset003.jpg,"Beautiful sunset over the ocean with orange and pink clouds"
```

**Notes:**
- First row must be header
- Caption should be quoted if it contains commas
- Image paths can be absolute or relative

---

### Format 4: Directory Structure

Organize images and captions in a directory:

```
my_data/
├── images/
│   ├── cat001.jpg
│   ├── dog002.jpg
│   └── sunset003.jpg
└── captions.txt
```

**captions.txt format:**
```
cat001.jpg: A fluffy orange cat sitting on a windowsill
dog002.jpg: A golden retriever playing in the park
sunset003.jpg: Beautiful sunset over the ocean
```

Or **separate caption files:**
```
my_data/
├── images/
│   ├── cat001.jpg
│   ├── dog002.jpg
│   └── sunset003.jpg
└── captions/
    ├── cat001.txt
    ├── dog002.txt
    └── sunset003.txt
```

---

## DPO Training Data

DPO requires **preference pairs**: image with chosen (better) and rejected (worse) captions.

### Format 1: JSON with Preferences

**File:** `my_data/preferences.json`

```json
{
  "preferences": [
    {
      "image_path": "images/cat001.jpg",
      "chosen": "A fluffy orange tabby cat lounging peacefully on a sunny windowsill",
      "rejected": "A cat"
    },
    {
      "image_path": "images/dog002.jpg",
      "chosen": "A joyful golden retriever running through a lush green park on a sunny day",
      "rejected": "Dog outside"
    },
    {
      "image_path": "images/sunset003.jpg",
      "chosen": "Breathtaking sunset over calm ocean waters, with vibrant orange and pink clouds painting the sky",
      "rejected": "Sunset at beach"
    }
  ]
}
```

**Required fields:**
- `image_path`: Path to image
- `chosen`: Better/preferred caption (detailed, accurate)
- `rejected`: Worse caption (generic, less descriptive)

**Tips for creating preferences:**
- `chosen`: Detailed, specific, accurate
- `rejected`: Generic, vague, or inaccurate
- Clear quality difference between chosen/rejected

---

### Format 2: JSONL Preferences

**File:** `my_data/preferences.jsonl`

```jsonl
{"image_path": "images/cat001.jpg", "chosen": "A fluffy orange tabby cat lounging peacefully", "rejected": "A cat"}
{"image_path": "images/dog002.jpg", "chosen": "A joyful golden retriever running in park", "rejected": "Dog outside"}
```

---

### Format 3: CSV Preferences

**File:** `my_data/preferences.csv`

```csv
image_path,chosen,rejected
images/cat001.jpg,"A fluffy orange tabby cat lounging peacefully on a sunny windowsill","A cat"
images/dog002.jpg,"A joyful golden retriever running through a park","Dog outside"
```

---

### Format 4: Ranked Captions

If you have multiple captions with rankings:

**File:** `my_data/ranked_captions.json`

```json
{
  "ranked_data": [
    {
      "image_path": "images/cat001.jpg",
      "captions": [
        {
          "text": "A fluffy orange tabby cat lounging peacefully on a sunny windowsill",
          "rank": 1
        },
        {
          "text": "An orange cat sitting on a windowsill",
          "rank": 2
        },
        {
          "text": "A cat",
          "rank": 3
        }
      ]
    }
  ]
}
```

The data loader will automatically create preference pairs from ranked captions.

---

## Loading Your Data

### Method 1: Using Data Configs (Recommended)

Create a custom data config:

**File:** `configs/data/my_custom_data.yaml`

```yaml
# Custom image-caption dataset
_target_: src.data.loaders.CustomImageCaptionDataset

# Dataset name
name: "my_custom_dataset"

# Data format
format: "json"  # Options: json, jsonl, csv, directory

# File paths
train_file: "/path/to/my_data/train.json"
val_file: "/path/to/my_data/val.json"  # Optional
test_file: "/path/to/my_data/test.json"  # Optional

# Image directory (if paths in data are relative)
image_dir: "/path/to/my_data/images"

# Data settings
max_train_samples: null  # null = use all
max_val_samples: null
image_size: 224
max_caption_length: 77  # CLIP max length

# Data augmentation (optional)
augmentation:
  horizontal_flip: true
  random_crop: false
  color_jitter: false

# Validation
validate_on_load: true  # Check all images exist and can be loaded
```

**Use in training:**
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=my_custom_data \
    model.use_lora=false
```

---

### Method 2: Using Python API

Load data programmatically:

```python
from src.data.loaders import load_custom_image_caption_data

# Load from JSON
train_data, val_data = load_custom_image_caption_data(
    train_file="my_data/train.json",
    val_file="my_data/val.json",
    image_dir="my_data/images",
    format="json"
)

# Load from JSONL (streaming for large datasets)
train_data = load_custom_image_caption_data(
    train_file="my_data/train.jsonl",
    format="jsonl",
    streaming=True  # Memory efficient
)

# Load from CSV
train_data = load_custom_image_caption_data(
    train_file="my_data/train.csv",
    format="csv"
)
```

---

### Method 3: Using Command Line Override

```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=/path/to/train.json \
    data.val_file=/path/to/val.json \
    data.image_dir=/path/to/images \
    data.format=json
```

---

## Training Examples

### Example 1: SFT with Custom JSON Data

**1. Prepare data:**
```json
{
  "data": [
    {"image_path": "cat.jpg", "caption": "A fluffy orange cat"},
    {"image_path": "dog.jpg", "caption": "A playful dog"}
  ]
}
```

**2. Create config:**
```yaml
# configs/data/my_pets.yaml
format: "json"
train_file: "/Users/you/pets/train.json"
image_dir: "/Users/you/pets/images"
```

**3. Train:**
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=my_pets \
    training.num_epochs=10
```

---

### Example 2: DPO with Preference Pairs

**1. Prepare preference data:**
```json
{
  "preferences": [
    {
      "image_path": "cat.jpg",
      "chosen": "A fluffy orange tabby cat with bright green eyes",
      "rejected": "A cat"
    }
  ]
}
```

**2. Train DPO:**
```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data.train_file=/path/to/preferences.json \
    data.format=json \
    training.beta=0.1
```

---

### Example 3: Large Dataset with JSONL Streaming

**1. Prepare JSONL (one JSON per line):**
```jsonl
{"image_path": "img1.jpg", "caption": "Caption 1"}
{"image_path": "img2.jpg", "caption": "Caption 2"}
...  # 100,000 more lines
```

**2. Train with streaming:**
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=/path/to/large_data.jsonl \
    data.format=jsonl \
    data.streaming=true \
    training.dataloader_num_workers=4
```

---

## Validation

Validate your data before training:

### Built-in Validator

```python
from src.data.validators import validate_image_caption_data

# Validate JSON data
is_valid, errors = validate_image_caption_data(
    data_file="my_data/train.json",
    image_dir="my_data/images",
    format="json"
)

if is_valid:
    print("✓ Data is valid!")
else:
    print("✗ Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### Validation Script

```bash
python scripts/data/validate_custom_data.py \
    --data-file my_data/train.json \
    --image-dir my_data/images \
    --format json \
    --check-images  # Verify all images can be loaded
```

**Validation checks:**
- ✅ Required fields present
- ✅ Image files exist
- ✅ Images can be loaded (PIL)
- ✅ Captions are non-empty
- ✅ No duplicate image-caption pairs
- ✅ Image paths are valid

---

## Best Practices

### Data Quality

**Good captions:**
```json
{
  "image_path": "cat.jpg",
  "caption": "A fluffy orange tabby cat with green eyes sitting on a wooden chair"
}
```

**Bad captions (avoid):**
```json
{
  "caption": "Cat"  // Too generic
  "caption": "asdfkj"  // Not descriptive
  "caption": ""  // Empty
}
```

### Dataset Size

**Minimum recommended:**
- SFT: 500-1000 image-caption pairs
- DPO: 200-500 preference pairs

**Optimal:**
- SFT: 5,000-10,000 pairs
- DPO: 1,000-2,000 pairs

### Image Requirements

**Supported formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

**Recommended specs:**
- Resolution: 224x224 or higher
- File size: < 5 MB per image
- Color: RGB (will convert grayscale automatically)

### Caption Guidelines

**Length:**
- Minimum: 3 words
- Optimal: 10-30 words
- Maximum: 77 tokens (CLIP limit)

**Quality:**
- Be specific and descriptive
- Include objects, actions, colors, setting
- Use natural language
- Avoid generic descriptions

**Example progression:**
- ❌ "A cat" (too generic)
- ⚠️ "An orange cat" (better but still generic)
- ✅ "A fluffy orange cat sitting on a windowsill" (good)
- ✅ "A fluffy orange tabby cat with bright eyes lounging peacefully on a sunny windowsill" (excellent)

### DPO Preferences

**Creating good preference pairs:**

```json
{
  "chosen": "A golden retriever playing fetch with a tennis ball in a sunny park",
  "rejected": "Dog playing"
}
```

**Quality difference matters:**
- Large difference: Trains faster, clearer signal
- Small difference: Trains slower, more nuanced

**Sources of preferences:**
- Human ratings (best)
- Multiple annotators with rankings
- Model-generated vs human-written
- Detailed vs generic captions

### File Organization

**Recommended structure:**
```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       └── img003.jpg
├── train.json
├── val.json
└── preferences.json  # For DPO
```

---

## Troubleshooting

### Issue: "Image file not found"

**Cause:** Incorrect path in data file

**Solution:**
```json
// Use absolute paths
{"image_path": "/Users/you/data/images/cat.jpg"}

// Or relative paths with image_dir config
{"image_path": "cat.jpg"}
// + config: image_dir: "/Users/you/data/images"
```

### Issue: "Unable to load image"

**Cause:** Corrupted image file

**Solution:**
```python
# Validate images before training
from PIL import Image

for img_path in image_paths:
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        print(f"Corrupted: {img_path}")
```

### Issue: "Out of memory during loading"

**Cause:** Loading entire dataset into memory

**Solution:**
```bash
# Use JSONL with streaming
data.format=jsonl
data.streaming=true
```

### Issue: "Captions are truncated"

**Cause:** Exceeding CLIP's 77 token limit

**Solution:**
```yaml
# In config
data:
  max_caption_length: 77
  truncate_long_captions: true  # Automatically truncate
```

---

## Examples Repository

See `examples/custom_data/` for complete examples:

- `examples/custom_data/simple_sft.json` - Basic SFT format
- `examples/custom_data/dpo_preferences.json` - DPO format
- `examples/custom_data/prepare_data.py` - Script to create data
- `examples/custom_data/validate_data.py` - Validation script

---

## API Reference

### Data Loading Functions

```python
from src.data.loaders import (
    load_custom_image_caption_data,
    load_custom_preference_data,
    ImageCaptionDataset,
    PreferenceDataset
)

# Load SFT data
train_data = load_custom_image_caption_data(
    train_file="path/to/train.json",
    format="json",
    image_dir="path/to/images",
    max_samples=1000,
    validate=True
)

# Load DPO preference data
pref_data = load_custom_preference_data(
    data_file="path/to/preferences.json",
    format="json",
    image_dir="path/to/images"
)

# Create dataset directly
from datasets import Dataset
dataset = ImageCaptionDataset(
    data=train_data,
    image_dir="path/to/images",
    transform=transform
)
```

---

## Next Steps

1. **Prepare your data** using one of the formats above
2. **Validate** with the validation script
3. **Create a config** in `configs/data/`
4. **Train** using the training scripts
5. **Evaluate** on your test set

See also:
- `docs/multimodal_training_guide.md` - Full training guide
- `scripts/data/convert_to_custom_format.py` - Convert from other formats
- `examples/custom_data/` - Complete examples
