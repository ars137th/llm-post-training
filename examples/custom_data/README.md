# Custom Data Examples

This directory contains example data files and scripts for training with your own custom image-caption datasets.

## Files

### Example Data Files

1. **`simple_sft.json`** - Example SFT (Supervised Fine-Tuning) data
   - 10 image-caption pairs
   - Shows the required JSON structure
   - Use as a template for your own data

2. **`dpo_preferences.json`** - Example DPO preference data
   - 5 preference pairs (chosen vs rejected captions)
   - Shows how to format preference data
   - Use for DPO training

### Scripts

3. **`prepare_data.py`** - Script to prepare your own custom data
   - Converts a directory of images to training format
   - Supports JSON, JSONL, CSV outputs
   - Can generate preference pairs

## Quick Start

### 1. Prepare Your Data

If you have a directory of images:

```bash
python prepare_data.py \
    --image-dir /path/to/your/images \
    --output-dir ./my_dataset \
    --caption-file /path/to/captions.txt \
    --create-preferences
```

This will create:
- `my_dataset/train.json` - SFT training data
- `my_dataset/preferences.json` - DPO preference pairs

### 2. Validate Your Data

Before training, validate that your data is correctly formatted:

```bash
python ../../scripts/data/validate_custom_data.py \
    --data-file ./my_dataset/train.json \
    --image-dir /path/to/your/images \
    --format json \
    --type sft
```

### 3. Train with SFT

Train CLIP or LLaVA on your custom data:

```bash
python ../../scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=./my_dataset/train.json \
    data.image_dir=/path/to/your/images \
    data.format=json \
    training.num_epochs=10
```

### 4. Train with DPO

After SFT, train with DPO on preference data:

```bash
python ../../scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=./my_dataset/preferences.json \
    data.image_dir=/path/to/your/images \
    data.format=json \
    training.num_epochs=5
```

## Data Format Examples

### SFT Format (JSON)

```json
{
  "data": [
    {
      "image_path": "images/image1.jpg",
      "caption": "Detailed description of the image"
    }
  ]
}
```

### DPO Format (JSON)

```json
{
  "preferences": [
    {
      "image_path": "images/image1.jpg",
      "chosen": "Detailed, high-quality caption",
      "rejected": "Generic caption"
    }
  ]
}
```

### JSONL Format

One JSON object per line:

```jsonl
{"image_path": "images/image1.jpg", "caption": "Caption 1"}
{"image_path": "images/image2.jpg", "caption": "Caption 2"}
```

### CSV Format

```csv
image_path,caption
images/image1.jpg,"Detailed caption"
images/image2.jpg,"Another caption"
```

## Tips

### Caption Quality

**Good captions:**
- Descriptive and specific
- 10-30 words
- Mention objects, actions, colors, setting
- Natural language

**Example:**
- ✅ "A fluffy orange tabby cat sitting on a sunny windowsill"
- ❌ "A cat"

### Preference Pairs

**For DPO training:**
- Chosen: Detailed, accurate, specific
- Rejected: Generic, vague, or less accurate
- Clear quality difference between chosen and rejected

### Dataset Size

**Minimum:**
- SFT: 100-500 pairs
- DPO: 50-200 preferences

**Recommended:**
- SFT: 1,000-5,000 pairs
- DPO: 500-2,000 preferences

### Image Requirements

- Formats: JPG, PNG, WebP
- Resolution: 224x224 or higher
- File size: < 5 MB per image
- Color: RGB (grayscale will be converted)

## Complete Guide

For full documentation, see: `../../docs/CUSTOM_DATA_GUIDE.md`

This includes:
- All supported data formats
- Loading methods (configs, API, CLI)
- Validation procedures
- Best practices
- Troubleshooting
- API reference

## Example Workflows

### Workflow 1: Train on Your Photos

1. Collect images in a directory
2. Write captions (manually or use a model)
3. Create `train.json` using the example format
4. Validate with `validate_custom_data.py`
5. Train with `train_multimodal.py`

### Workflow 2: Improve Existing Model

1. Start with SFT-trained model
2. Collect preference annotations (human ratings)
3. Create `preferences.json`
4. Train with DPO using `train_multimodal_dpo.py`
5. Compare with SFT baseline

### Workflow 3: Large Dataset

1. Organize images in directory
2. Create JSONL format for streaming
3. Use `prepare_data.py` with `--create-jsonl`
4. Train with `data.format=jsonl data.streaming=true`

## Support

Questions or issues?
- Check the full guide: `docs/CUSTOM_DATA_GUIDE.md`
- Review example data files in this directory
- Run validation script to diagnose problems
- Open an issue on GitHub

## Next Steps

1. **Review examples:** Look at `simple_sft.json` and `dpo_preferences.json`
2. **Prepare your data:** Use `prepare_data.py` or create manually
3. **Validate:** Run `validate_custom_data.py`
4. **Train:** Use custom data configs with training scripts
5. **Evaluate:** Compare results with baseline models
