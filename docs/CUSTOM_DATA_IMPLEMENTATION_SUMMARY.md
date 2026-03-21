# Custom Data Loading - Implementation Summary

## Overview

This document summarizes the implementation of custom data loading functionality for training multimodal models (CLIP, LLaVA) with your own image-caption datasets.

**Date:** March 20, 2024

---

## What Was Implemented

### 1. Core Data Loaders (`src/data/loaders/custom.py`)

**Purpose:** Load custom image-caption and preference data in various formats.

**Key Classes:**

#### `CustomImageCaptionLoader`
Loads image-caption pairs for SFT training.

**Features:**
- Supports JSON, JSONL, CSV formats
- Path resolution (absolute vs relative)
- Data validation (images exist, captions non-empty)
- Handles multiple JSON structures
- Optional sample limiting

**Usage:**
```python
from src.data.loaders.custom import CustomImageCaptionLoader

loader = CustomImageCaptionLoader(
    data_file="train.json",
    image_dir="images/",
    format="json",
    validate=True
)
examples = loader.load()  # Returns List[Dict]
```

#### `CustomPreferenceLoader`
Loads preference pairs for DPO training.

**Features:**
- Supports JSON, JSONL, CSV, and ranked formats
- Validates chosen != rejected
- Filters invalid preferences
- Auto-creates pairs from ranked captions

**Usage:**
```python
from src.data.loaders.custom import CustomPreferenceLoader

loader = CustomPreferenceLoader(
    data_file="preferences.json",
    image_dir="images/",
    format="json",
    validate=True
)
preferences = loader.load()  # Returns List[Dict]
```

#### Convenience Functions

**`load_custom_image_caption_data()`**
- Loads train/val splits in one call
- Handles both single and split datasets

**`load_custom_preference_data()`**
- Simplified preference loading
- Same interface as image-caption loader

---

### 2. Configuration Files

#### `configs/data/custom_image_caption.yaml`

Configuration template for custom SFT data.

**Key Parameters:**
```yaml
# Required
train_file: /path/to/train.json
image_dir: /path/to/images
format: json  # or jsonl, csv

# Optional
val_file: /path/to/val.json
max_train_samples: null
validate_on_load: true
```

**Usage:**
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=/path/to/train.json \
    data.image_dir=/path/to/images
```

#### `configs/data/custom_preferences.yaml`

Configuration template for custom DPO data.

**Key Parameters:**
```yaml
# Required
train_file: /path/to/preferences.json
image_dir: /path/to/images
format: json  # or jsonl, csv, ranked

# Optional
val_file: /path/to/val_preferences.json
validate_preferences: true
```

**Usage:**
```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=/path/to/preferences.json \
    data.image_dir=/path/to/images
```

---

### 3. Training Script Updates

#### `scripts/train/train_multimodal.py`

**Added:**
- Custom data loading path (lines 220-287)
- Auto-detects `dataset_name="custom"` or `data.train_file` presence
- Loads data using `CustomImageCaptionLoader`
- Converts to `MultimodalExample` objects
- Handles train/val splitting if no val file provided

**Example Usage:**
```bash
# Method 1: Use config override
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=/path/to/train.json \
    data.image_dir=/path/to/images

# Method 2: Direct CLI override (any experiment)
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=/path/to/train.json \
    data.image_dir=/path/to/images \
    data.format=json
```

#### `scripts/train/train_multimodal_dpo.py`

**Added:**
- Custom preference data loading (lines 189-258)
- Auto-detects `dataset_name="custom"` or `data.train_file` presence
- Loads data using `CustomPreferenceLoader`
- Converts to preference pair format
- Handles train/val splitting

**Example Usage:**
```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=/path/to/preferences.json \
    data.image_dir=/path/to/images
```

---

### 4. Documentation

#### `docs/CUSTOM_DATA_GUIDE.md` (~600 lines)

**Complete user guide covering:**
- Data format specifications (JSON, JSONL, CSV)
- SFT format examples
- DPO preference format examples
- Loading methods (configs, Python API, CLI)
- Validation procedures
- Best practices for caption quality
- Dataset size recommendations
- Troubleshooting common issues
- API reference

**Sections:**
1. Overview
2. Data Formats (Quick Reference Table)
3. SFT Training Data (4 format examples)
4. DPO Training Data (4 format examples)
5. Loading Your Data (3 methods)
6. Training Examples
7. Validation
8. Best Practices
9. Troubleshooting
10. API Reference

---

### 5. Example Data & Scripts

#### `examples/custom_data/simple_sft.json`

10 example image-caption pairs showing the required JSON structure.

**Format:**
```json
{
  "data": [
    {
      "image_path": "images/cat001.jpg",
      "caption": "A fluffy orange tabby cat..."
    }
  ]
}
```

#### `examples/custom_data/dpo_preferences.json`

5 example preference pairs for DPO training.

**Format:**
```json
{
  "preferences": [
    {
      "image_path": "images/cat001.jpg",
      "chosen": "Detailed caption...",
      "rejected": "Generic caption"
    }
  ]
}
```

#### `examples/custom_data/prepare_data.py`

**Script to prepare custom data from image directory.**

**Features:**
- Converts directory of images to training format
- Loads existing captions or creates placeholders
- Generates JSON, JSONL, CSV formats
- Creates synthetic preference pairs
- Validates images can be loaded

**Usage:**
```bash
python prepare_data.py \
    --image-dir ./images \
    --output-dir ./processed_data \
    --caption-file ./captions.txt \
    --create-preferences \
    --create-jsonl
```

**Outputs:**
- `processed_data/train.json` - SFT data
- `processed_data/train.jsonl` - JSONL format
- `processed_data/train.csv` - CSV format
- `processed_data/preferences.json` - DPO preferences

#### `examples/custom_data/README.md`

Quick start guide for the examples directory with complete workflows.

---

### 6. Validation Script

#### `scripts/data/validate_custom_data.py`

**Standalone validation script for custom data.**

**Checks:**
- ✅ File format is correct
- ✅ Required fields present
- ✅ Images exist and can be loaded
- ✅ Captions are non-empty
- ✅ No duplicate entries
- ✅ Preference pairs have different chosen/rejected
- ✅ Image paths are valid

**Usage:**
```bash
# Validate SFT data
python scripts/data/validate_custom_data.py \
    --data-file my_data/train.json \
    --image-dir my_data/images \
    --format json \
    --type sft

# Validate DPO preferences
python scripts/data/validate_custom_data.py \
    --data-file my_data/preferences.json \
    --image-dir my_data/images \
    --format json \
    --type dpo
```

**Output:**
```
============================================================
Custom Data Validation
============================================================
Data file: my_data/train.json
Format: json
Type: sft
Image dir: my_data/images
Check images: True

Loading data from my_data/train.json...
✓ Loaded 100 examples

============================================================
SFT Data Validation Summary
============================================================
Total examples: 100
Valid examples: 98
Invalid examples: 2
Errors found: 2

❌ Found 2 errors
============================================================
Errors (showing first 20):
============================================================
  ❌ Example 5: Image not found: my_data/images/missing.jpg
  ❌ Example 12: Empty caption
```

---

## Data Format Specifications

### SFT (Supervised Fine-Tuning)

#### JSON Format
```json
{
  "data": [
    {
      "image_path": "path/to/image.jpg",
      "caption": "Descriptive caption",
      "id": "optional_id"
    }
  ]
}
```

#### JSONL Format
```jsonl
{"image_path": "path/to/image1.jpg", "caption": "Caption 1"}
{"image_path": "path/to/image2.jpg", "caption": "Caption 2"}
```

#### CSV Format
```csv
image_path,caption,id
path/to/image1.jpg,"Caption 1",img001
path/to/image2.jpg,"Caption 2",img002
```

### DPO (Direct Preference Optimization)

#### JSON Format
```json
{
  "preferences": [
    {
      "image_path": "path/to/image.jpg",
      "chosen": "High-quality caption",
      "rejected": "Low-quality caption",
      "id": "optional_id"
    }
  ]
}
```

#### Ranked Format
```json
{
  "ranked_data": [
    {
      "image_path": "path/to/image.jpg",
      "captions": [
        {"text": "Best caption", "rank": 1},
        {"text": "Good caption", "rank": 2},
        {"text": "OK caption", "rank": 3}
      ]
    }
  ]
}
```

*Note:* Loader automatically creates preference pairs from ranked data.

---

## Complete Usage Workflow

### Step 1: Prepare Your Data

**Option A: Manual Creation**

Create a JSON file with your image-caption pairs:

```json
{
  "data": [
    {"image_path": "img1.jpg", "caption": "Your caption here"},
    {"image_path": "img2.jpg", "caption": "Another caption"}
  ]
}
```

**Option B: Use Preparation Script**

```bash
python examples/custom_data/prepare_data.py \
    --image-dir /path/to/images \
    --caption-file /path/to/captions.txt \
    --output-dir ./my_dataset
```

### Step 2: Validate

```bash
python scripts/data/validate_custom_data.py \
    --data-file ./my_dataset/train.json \
    --image-dir /path/to/images \
    --format json \
    --type sft
```

### Step 3: Train (SFT)

```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=./my_dataset/train.json \
    data.image_dir=/path/to/images \
    data.format=json \
    training.num_epochs=10
```

### Step 4: Create Preferences (for DPO)

Either manually or using the script:

```bash
python examples/custom_data/prepare_data.py \
    --image-dir /path/to/images \
    --output-dir ./my_dataset \
    --create-preferences
```

### Step 5: Train (DPO)

```bash
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=./my_dataset/preferences.json \
    data.image_dir=/path/to/images \
    training.num_epochs=5
```

---

## Files Created

**Core Implementation:**
- `src/data/loaders/custom.py` (~560 lines)

**Configuration:**
- `configs/data/custom_image_caption.yaml` (~80 lines)
- `configs/data/custom_preferences.yaml` (~70 lines)

**Documentation:**
- `docs/CUSTOM_DATA_GUIDE.md` (~700 lines)
- `docs/CUSTOM_DATA_IMPLEMENTATION_SUMMARY.md` (this file)
- `examples/custom_data/README.md` (~250 lines)

**Examples:**
- `examples/custom_data/simple_sft.json` (10 examples)
- `examples/custom_data/dpo_preferences.json` (5 examples)
- `examples/custom_data/prepare_data.py` (~300 lines)

**Validation:**
- `scripts/data/validate_custom_data.py` (~350 lines)

**Updated Scripts:**
- `scripts/train/train_multimodal.py` (added custom data support)
- `scripts/train/train_multimodal_dpo.py` (added custom preference support)

**Total:** ~2,400 lines of code and documentation

---

## Key Features

1. **Multiple Format Support:** JSON, JSONL, CSV, ranked
2. **Flexible Path Resolution:** Absolute or relative image paths
3. **Built-in Validation:** Checks images exist and can be loaded
4. **Automatic Splitting:** Creates train/val split if no val file provided
5. **Streaming Support:** JSONL format for large datasets
6. **Error Handling:** Detailed error messages for debugging
7. **Config Integration:** Works with Hydra configuration system
8. **CLI Override Support:** Can specify paths via command line
9. **Comprehensive Documentation:** User guide, examples, API reference
10. **Validation Tools:** Standalone script to check data before training

---

## Testing

### Manual Tests Recommended

1. **Basic SFT Training:**
   ```bash
   python scripts/train/train_multimodal.py \
       experiment=clip_image_caption \
       data=custom_image_caption \
       data.train_file=examples/custom_data/simple_sft.json \
       data.image_dir=examples/custom_data/images \
       +training.max_steps=10 \
       training.num_epochs=null
   ```

2. **Basic DPO Training:**
   ```bash
   python scripts/train/train_multimodal_dpo.py \
       experiment=clip_dpo \
       data=custom_preferences \
       data.train_file=examples/custom_data/dpo_preferences.json \
       data.image_dir=examples/custom_data/images \
       +training.max_steps=10 \
       training.num_epochs=null
   ```

3. **Validation:**
   ```bash
   python scripts/data/validate_custom_data.py \
       --data-file examples/custom_data/simple_sft.json \
       --format json \
       --type sft \
       --no-check-images
   ```

---

## Next Steps (Future Enhancements)

### Potential Additions

1. **Auto-Captioning:** Script to generate captions using BLIP/LLaVA
2. **Data Augmentation:** Image augmentation utilities
3. **Quality Filtering:** Auto-filter low-quality captions
4. **Statistics:** Script to analyze dataset statistics
5. **Conversion Tools:** Convert between different formats
6. **Deduplication:** Remove duplicate image-caption pairs
7. **Dataset Merging:** Combine multiple datasets
8. **Active Learning:** Select best examples for annotation

### Integration with Other Tools

- HuggingFace Datasets integration
- WandB dataset versioning
- MLflow data logging
- DVC for data versioning

---

## Support and Troubleshooting

**Common Issues:**

1. **Image not found:** Check `image_dir` is correct and paths in data file
2. **Empty caption:** All captions must be non-empty strings
3. **Format error:** Verify JSON/JSONL/CSV structure matches specs
4. **Memory error:** Use JSONL format with streaming for large datasets

**Getting Help:**

1. Read `docs/CUSTOM_DATA_GUIDE.md` for detailed instructions
2. Check example files in `examples/custom_data/`
3. Run validation script to diagnose issues
4. Open GitHub issue with error details

---

## Summary

This implementation provides a complete, flexible system for training multimodal models with custom data. Users can:

✅ Use their own image-caption datasets
✅ Train both SFT and DPO models
✅ Support multiple data formats
✅ Validate data before training
✅ Follow comprehensive documentation
✅ Use provided examples as templates

The system integrates seamlessly with the existing training infrastructure while maintaining flexibility and ease of use.
