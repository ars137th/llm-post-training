"""
Example script to prepare custom image-caption data for training.

This script demonstrates how to:
1. Create image-caption pairs from a directory of images
2. Generate synthetic captions (or load existing ones)
3. Save in the required format for training
4. Create preference pairs for DPO training

Usage:
    python prepare_data.py --image-dir ./images --output-dir ./processed_data
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from PIL import Image


def create_sft_data_from_directory(
    image_dir: Path,
    output_file: Path,
    caption_file: Path = None
) -> List[Dict]:
    """
    Create SFT training data from a directory of images.

    Args:
        image_dir: Directory containing images
        output_file: Path to save JSON file
        caption_file: Optional file with existing captions (one per line)

    Returns:
        List of image-caption pairs
    """
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob("*.jpg")) +
                        list(image_dir.glob("*.png")) +
                        list(image_dir.glob("*.jpeg")))

    print(f"Found {len(image_files)} images in {image_dir}")

    # Load captions if provided
    captions = []
    if caption_file and Path(caption_file).exists():
        with open(caption_file, 'r') as f:
            captions = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(captions)} captions from {caption_file}")

    # Create dataset
    data = []
    for i, img_path in enumerate(image_files):
        # Verify image can be loaded
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"⚠️  Skipping {img_path}: {e}")
            continue

        # Use provided caption or generate placeholder
        if i < len(captions):
            caption = captions[i]
        else:
            # Placeholder - replace with actual captions
            caption = f"Image of {img_path.stem}"
            print(f"⚠️  No caption for {img_path.name}, using placeholder")

        data.append({
            "image_path": str(img_path),
            "caption": caption,
            "id": img_path.stem
        })

    # Save as JSON
    output = {"data": data}
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved {len(data)} examples to {output_file}")
    return data


def create_jsonl_from_json(json_file: Path, jsonl_file: Path):
    """
    Convert JSON format to JSONL (one example per line).

    Useful for large datasets to enable streaming.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    examples = data.get('data', [])

    with open(jsonl_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Converted {len(examples)} examples to JSONL: {jsonl_file}")


def create_preference_pairs(
    sft_data: List[Dict],
    output_file: Path,
    quality_scores: Dict[str, float] = None
) -> List[Dict]:
    """
    Create preference pairs for DPO training.

    This example creates pairs by:
    - Using detailed captions as "chosen"
    - Creating generic captions as "rejected"

    In practice, you should have human annotations or model rankings.

    Args:
        sft_data: List of image-caption pairs
        output_file: Path to save preferences
        quality_scores: Optional dict of caption quality scores

    Returns:
        List of preference pairs
    """
    preferences = []

    for item in sft_data:
        image_path = item['image_path']
        chosen_caption = item['caption']

        # Create a worse version (simplified) as rejected
        # This is just an example - real preferences should come from annotations
        words = chosen_caption.split()
        if len(words) > 5:
            # Truncate to make it worse
            rejected_caption = ' '.join(words[:3])
        else:
            rejected_caption = words[0]  # Just the first word

        preferences.append({
            "image_path": image_path,
            "chosen": chosen_caption,
            "rejected": rejected_caption,
            "id": item.get('id', Path(image_path).stem)
        })

    # Save as JSON
    output = {"preferences": preferences}
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Created {len(preferences)} preference pairs: {output_file}")
    print("⚠️  Note: These are synthetic preferences. Use real human annotations for best results.")
    return preferences


def create_csv_format(json_file: Path, csv_file: Path):
    """
    Convert JSON format to CSV.

    Useful for editing in spreadsheet software.
    """
    import csv

    with open(json_file, 'r') as f:
        data = json.load(f)

    examples = data.get('data', [])

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'caption', 'id'])
        writer.writeheader()
        for example in examples:
            writer.writerow({
                'image_path': example['image_path'],
                'caption': example['caption'],
                'id': example.get('id', '')
            })

    print(f"✓ Converted {len(examples)} examples to CSV: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare custom data for training")
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                       help='Directory to save processed data')
    parser.add_argument('--caption-file', type=str, default=None,
                       help='File with captions (one per line, matching image order)')
    parser.add_argument('--create-preferences', action='store_true',
                       help='Also create preference pairs for DPO')
    parser.add_argument('--create-jsonl', action='store_true',
                       help='Also create JSONL format')
    parser.add_argument('--create-csv', action='store_true',
                       help='Also create CSV format')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Custom Training Data")
    print("=" * 60)

    # Create SFT data (JSON format)
    json_file = output_dir / "train.json"
    sft_data = create_sft_data_from_directory(
        image_dir=args.image_dir,
        output_file=json_file,
        caption_file=args.caption_file
    )

    # Create JSONL format if requested
    if args.create_jsonl:
        jsonl_file = output_dir / "train.jsonl"
        create_jsonl_from_json(json_file, jsonl_file)

    # Create CSV format if requested
    if args.create_csv:
        csv_file = output_dir / "train.csv"
        create_csv_format(json_file, csv_file)

    # Create preference pairs if requested
    if args.create_preferences:
        preferences_file = output_dir / "preferences.json"
        create_preference_pairs(sft_data, preferences_file)

    print("\n" + "=" * 60)
    print("✓ Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}")
    print("\nNext steps:")
    print("1. Review and edit captions (especially if using placeholders)")
    print("2. Validate data:")
    print(f"   python scripts/data/validate_custom_data.py --data-file {json_file}")
    print("3. Train model:")
    print(f"   python scripts/train/train_multimodal.py \\")
    print(f"       experiment=clip_image_caption \\")
    print(f"       data=custom_image_caption \\")
    print(f"       data.train_file={json_file} \\")
    print(f"       data.image_dir={args.image_dir}")


if __name__ == "__main__":
    main()
