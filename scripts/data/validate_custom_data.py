"""
Validate custom image-caption or preference data before training.

This script checks:
- File format is correct
- Required fields are present
- Images exist and can be loaded
- Captions are non-empty
- No duplicate entries
- Preference pairs have different chosen/rejected

Usage:
    # Validate SFT data
    python validate_custom_data.py \
        --data-file my_data/train.json \
        --image-dir my_data/images \
        --format json \
        --type sft

    # Validate DPO preferences
    python validate_custom_data.py \
        --data-file my_data/preferences.json \
        --image-dir my_data/images \
        --format json \
        --type dpo
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import json
import csv


def validate_sft_data(
    data: List[Dict],
    image_dir: Path = None,
    check_images: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate SFT image-caption data.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    valid_count = 0

    # Check structure
    if not isinstance(data, list):
        errors.append("Data must be a list of examples")
        return False, errors

    # Check each example
    seen_pairs = set()
    for i, example in enumerate(data):
        if not isinstance(example, dict):
            errors.append(f"Example {i}: Not a dictionary")
            continue

        # Check required fields
        if 'image_path' not in example:
            errors.append(f"Example {i}: Missing 'image_path' field")
            continue

        if 'caption' not in example:
            errors.append(f"Example {i}: Missing 'caption' field")
            continue

        image_path = example['image_path']
        caption = example['caption']

        # Check caption not empty
        if not caption or not caption.strip():
            errors.append(f"Example {i}: Empty caption for {image_path}")
            continue

        # Check for duplicates
        pair_key = (image_path, caption)
        if pair_key in seen_pairs:
            errors.append(f"Example {i}: Duplicate pair {image_path} + caption")
            continue
        seen_pairs.add(pair_key)

        # Check image exists and can be loaded
        if check_images:
            img_path = Path(image_path)
            if not img_path.is_absolute() and image_dir:
                img_path = image_dir / img_path

            if not img_path.exists():
                errors.append(f"Example {i}: Image not found: {img_path}")
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                errors.append(f"Example {i}: Cannot load image {img_path}: {e}")
                continue

        valid_count += 1

    # Summary
    total = len(data)
    is_valid = len(errors) == 0

    print(f"\n{'='*60}")
    print("SFT Data Validation Summary")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Valid examples: {valid_count}")
    print(f"Invalid examples: {total - valid_count}")
    print(f"Errors found: {len(errors)}")

    if is_valid:
        print("\n✅ All data is valid!")
    else:
        print(f"\n❌ Found {len(errors)} errors")

    return is_valid, errors


def validate_preference_data(
    data: List[Dict],
    image_dir: Path = None,
    check_images: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate DPO preference data.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    valid_count = 0

    # Check structure
    if not isinstance(data, list):
        errors.append("Data must be a list of preference pairs")
        return False, errors

    # Check each preference
    seen_pairs = set()
    for i, pref in enumerate(data):
        if not isinstance(pref, dict):
            errors.append(f"Preference {i}: Not a dictionary")
            continue

        # Check required fields
        required = ['image_path', 'chosen', 'rejected']
        missing = [f for f in required if f not in pref]
        if missing:
            errors.append(f"Preference {i}: Missing fields: {missing}")
            continue

        image_path = pref['image_path']
        chosen = pref['chosen']
        rejected = pref['rejected']

        # Check captions not empty
        if not chosen or not chosen.strip():
            errors.append(f"Preference {i}: Empty chosen caption")
            continue

        if not rejected or not rejected.strip():
            errors.append(f"Preference {i}: Empty rejected caption")
            continue

        # Check chosen != rejected
        if chosen.strip() == rejected.strip():
            errors.append(f"Preference {i}: Chosen and rejected are identical")
            continue

        # Check for duplicates
        pair_key = (image_path, chosen, rejected)
        if pair_key in seen_pairs:
            errors.append(f"Preference {i}: Duplicate preference pair")
            continue
        seen_pairs.add(pair_key)

        # Check image exists
        if check_images:
            img_path = Path(image_path)
            if not img_path.is_absolute() and image_dir:
                img_path = image_dir / img_path

            if not img_path.exists():
                errors.append(f"Preference {i}: Image not found: {img_path}")
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                errors.append(f"Preference {i}: Cannot load image {img_path}: {e}")
                continue

        valid_count += 1

    # Summary
    total = len(data)
    is_valid = len(errors) == 0

    print(f"\n{'='*60}")
    print("DPO Preference Data Validation Summary")
    print(f"{'='*60}")
    print(f"Total preferences: {total}")
    print(f"Valid preferences: {valid_count}")
    print(f"Invalid preferences: {total - valid_count}")
    print(f"Errors found: {len(errors)}")

    if is_valid:
        print("\n✅ All data is valid!")
    else:
        print(f"\n❌ Found {len(errors)} errors")

    return is_valid, errors


def load_data(
    data_file: Path,
    format: str
) -> List[Dict]:
    """Load data from file based on format."""

    if format == "json":
        with open(data_file, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "data" in data:
                return data["data"]
            elif "preferences" in data:
                return data["preferences"]
            elif "examples" in data:
                return data["examples"]
            else:
                raise ValueError("JSON must have 'data', 'preferences', or 'examples' key")
        else:
            raise ValueError("JSON must be list or dict")

    elif format == "jsonl":
        data = []
        with open(data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
        return data

    elif format == "csv":
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data

    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(description="Validate custom training data")
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to data file (JSON, JSONL, or CSV)')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Base directory for image paths (if relative)')
    parser.add_argument('--format', type=str, choices=['json', 'jsonl', 'csv'],
                       default='json', help='Data format')
    parser.add_argument('--type', type=str, choices=['sft', 'dpo'],
                       default='sft', help='Data type: sft or dpo')
    parser.add_argument('--no-check-images', action='store_true',
                       help='Skip image existence checks')
    parser.add_argument('--max-errors', type=int, default=20,
                       help='Maximum errors to display (default: 20)')

    args = parser.parse_args()

    data_file = Path(args.data_file)
    image_dir = Path(args.image_dir) if args.image_dir else None

    print(f"\n{'='*60}")
    print("Custom Data Validation")
    print(f"{'='*60}")
    print(f"Data file: {data_file}")
    print(f"Format: {args.format}")
    print(f"Type: {args.type}")
    print(f"Image dir: {image_dir or 'None (using absolute paths)'}")
    print(f"Check images: {not args.no_check_images}")

    # Check file exists
    if not data_file.exists():
        print(f"\n❌ Error: Data file not found: {data_file}")
        sys.exit(1)

    # Load data
    try:
        print(f"\nLoading data from {data_file}...")
        data = load_data(data_file, args.format)
        print(f"✓ Loaded {len(data)} examples")
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)

    # Validate data
    check_images = not args.no_check_images

    if args.type == "sft":
        is_valid, errors = validate_sft_data(data, image_dir, check_images)
    else:  # dpo
        is_valid, errors = validate_preference_data(data, image_dir, check_images)

    # Display errors
    if errors:
        print(f"\n{'='*60}")
        print(f"Errors (showing first {args.max_errors}):")
        print(f"{'='*60}")
        for error in errors[:args.max_errors]:
            print(f"  ❌ {error}")

        if len(errors) > args.max_errors:
            print(f"\n  ... and {len(errors) - args.max_errors} more errors")

    # Exit with appropriate code
    if is_valid:
        print(f"\n{'='*60}")
        print("✅ Validation passed! Data is ready for training.")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("❌ Validation failed. Please fix errors before training.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
