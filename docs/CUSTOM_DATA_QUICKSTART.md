# Custom Data Quick Start Guide

Get started with training on your own image-caption data in 5 minutes!

---

## Prerequisites

- Python 3.8+ with the repository installed
- A directory of images
- Captions for your images (or use the examples)

---

## Option 1: Use Example Data (Fastest)

The repository includes example data you can use immediately:

```bash
# Validate example data (optional - already valid)
python scripts/data/validate_custom_data.py \
    --data-file examples/custom_data/simple_sft.json \
    --format json \
    --type sft \
    --no-check-images

# Train CLIP on example data
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=examples/custom_data/simple_sft.json \
    data.image_dir=examples/custom_data/images \
    +training.max_steps=50 \
    training.num_epochs=null
```

**Note:** You'll need to create the `examples/custom_data/images/` directory and add images matching the filenames in `simple_sft.json`.

---

## Option 2: Use Your Own Data (3 Steps)

### Step 1: Create Data File

Create a JSON file with your image-caption pairs:

**File: `my_data.json`**
```json
{
  "data": [
    {
      "image_path": "cat.jpg",
      "caption": "A fluffy orange cat sitting on a windowsill"
    },
    {
      "image_path": "dog.jpg",
      "caption": "A golden retriever playing in the park"
    },
    {
      "image_path": "sunset.jpg",
      "caption": "Beautiful sunset over the ocean"
    }
  ]
}
```

**Requirements:**
- Each entry must have `image_path` and `caption`
- Image paths can be relative (to `image_dir`) or absolute
- Captions should be descriptive (10-30 words recommended)

### Step 2: Validate (Recommended)

Check your data before training:

```bash
python scripts/data/validate_custom_data.py \
    --data-file my_data.json \
    --image-dir /path/to/your/images \
    --format json \
    --type sft
```

**Expected output:**
```
✅ All data is valid!
Total examples: 3
Valid examples: 3
```

If you see errors, fix them before proceeding to training.

### Step 3: Train

```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=my_data.json \
    data.image_dir=/path/to/your/images \
    data.format=json \
    training.num_epochs=10
```

**That's it!** Your model will start training.

---

## Option 3: Prepare Data from Directory

If you have a directory of images but no data file yet:

```bash
# Prepare data file from images
python examples/custom_data/prepare_data.py \
    --image-dir /path/to/your/images \
    --output-dir ./processed_data \
    --caption-file captions.txt

# This creates:
# - processed_data/train.json
# - processed_data/train.jsonl (optional)
# - processed_data/preferences.json (optional)
```

**Caption file format (captions.txt):**
```
One caption per line, matching image order
```

**Then train:**
```bash
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=./processed_data/train.json \
    data.image_dir=/path/to/your/images \
    training.num_epochs=10
```

---

## Data Format Reference

### Minimal JSON (SFT)
```json
{
  "data": [
    {"image_path": "img1.jpg", "caption": "Caption 1"},
    {"image_path": "img2.jpg", "caption": "Caption 2"}
  ]
}
```

### Minimal JSON (DPO Preferences)
```json
{
  "preferences": [
    {
      "image_path": "img1.jpg",
      "chosen": "Detailed, high-quality caption",
      "rejected": "Generic caption"
    }
  ]
}
```

### JSONL Format (for large datasets)
```jsonl
{"image_path": "img1.jpg", "caption": "Caption 1"}
{"image_path": "img2.jpg", "caption": "Caption 2"}
```

### CSV Format
```csv
image_path,caption
img1.jpg,"Caption 1"
img2.jpg,"Caption 2"
```

---

## Common Patterns

### Pattern 1: Single Data File (Auto-Split)

```bash
# Train with auto train/val split
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=my_data.json \
    data.image_dir=/path/to/images
```

The script will automatically use 90% for training, 10% for validation.

### Pattern 2: Separate Train/Val Files

```bash
# Train with explicit train and val files
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=train.json \
    data.val_file=val.json \
    data.image_dir=/path/to/images
```

### Pattern 3: Limit Sample Size (For Testing)

```bash
# Train on subset for quick testing
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=my_data.json \
    data.image_dir=/path/to/images \
    data.max_train_samples=100 \
    +training.max_steps=50 \
    training.num_epochs=null
```

### Pattern 4: Large Dataset (JSONL Streaming)

```bash
# Use JSONL for memory-efficient loading
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.train_file=large_data.jsonl \
    data.image_dir=/path/to/images \
    data.format=jsonl \
    data.streaming=true
```

### Pattern 5: DPO Training on Preferences

```bash
# First train SFT, then DPO
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=preferences.json \
    data.image_dir=/path/to/images \
    training.num_epochs=5
```

---

## Troubleshooting

### Error: "Image not found"

**Problem:** Image paths in data file don't match actual image locations

**Solution:**
```bash
# Option 1: Use absolute paths in data file
{"image_path": "/full/path/to/image.jpg", "caption": "..."}

# Option 2: Use relative paths + image_dir parameter
{"image_path": "image.jpg", "caption": "..."}
# Then: data.image_dir=/path/to/images
```

### Error: "Empty caption"

**Problem:** Some captions are empty or whitespace-only

**Solution:** Ensure all captions have actual text:
```json
❌ {"image_path": "img.jpg", "caption": ""}
❌ {"image_path": "img.jpg", "caption": "   "}
✅ {"image_path": "img.jpg", "caption": "A real caption"}
```

### Error: "Validation failed"

**Problem:** Data doesn't match expected format

**Solution:** Run validation script to see specific errors:
```bash
python scripts/data/validate_custom_data.py \
    --data-file my_data.json \
    --image-dir /path/to/images \
    --format json \
    --type sft \
    --max-errors 50
```

### Warning: "Using placeholders"

**Problem:** `prepare_data.py` created placeholder captions

**Solution:** Edit the generated JSON file and replace placeholder captions with real descriptions.

---

## Next Steps

After getting basic training working:

1. **Improve Caption Quality**
   - Be specific and descriptive
   - Include objects, actions, colors, setting
   - Aim for 10-30 words

2. **Increase Dataset Size**
   - Minimum: 100-500 pairs for experimentation
   - Recommended: 1,000-5,000 pairs for good results

3. **Try DPO Training**
   - Create preference pairs (chosen vs rejected captions)
   - Train with DPO after SFT
   - Compare results

4. **Evaluate Results**
   - Generate captions on test images
   - Compare with baseline model
   - Measure CLIP Score

---

## Full Documentation

For complete details, see:
- **`docs/CUSTOM_DATA_GUIDE.md`** - Comprehensive guide
- **`examples/custom_data/README.md`** - Example workflows
- **`docs/CUSTOM_DATA_IMPLEMENTATION_SUMMARY.md`** - Technical details

---

## Command Reference

```bash
# Validate data
python scripts/data/validate_custom_data.py \
    --data-file FILE --image-dir DIR --format FORMAT --type TYPE

# Prepare data
python examples/custom_data/prepare_data.py \
    --image-dir DIR --output-dir DIR [--caption-file FILE]

# Train SFT
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=FILE \
    data.image_dir=DIR

# Train DPO
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=FILE \
    data.image_dir=DIR
```

---

## Questions?

- Review example files in `examples/custom_data/`
- Check full guide at `docs/CUSTOM_DATA_GUIDE.md`
- Run validation to diagnose issues
- Open GitHub issue for help
