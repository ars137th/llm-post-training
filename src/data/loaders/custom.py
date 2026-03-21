"""
Custom Data Loaders for Image-Caption Training

Supports loading custom image-caption datasets in various formats:
- JSON
- JSONL (JSON Lines)
- CSV
- Directory structure

See docs/CUSTOM_DATA_GUIDE.md for format specifications.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CustomImageCaptionLoader:
    """
    Load custom image-caption datasets in various formats.

    Supports:
    - JSON: {"data": [{"image_path": "...", "caption": "..."}]}
    - JSONL: One JSON object per line
    - CSV: image_path,caption header with rows
    - Directory: images/ + captions.txt or captions/

    Example:
        >>> loader = CustomImageCaptionLoader(
        ...     data_file="train.json",
        ...     image_dir="images/",
        ...     format="json"
        ... )
        >>> examples = loader.load()
        >>> print(len(examples))
        1000
    """

    def __init__(
        self,
        data_file: str,
        image_dir: Optional[str] = None,
        format: str = "json",
        max_samples: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Initialize custom data loader.

        Args:
            data_file: Path to data file (JSON, JSONL, CSV)
            image_dir: Base directory for image paths (if relative paths used)
            format: Data format: "json", "jsonl", "csv"
            max_samples: Maximum number of samples to load (None = all)
            validate: Whether to validate data on load
        """
        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir) if image_dir else None
        self.format = format.lower()
        self.max_samples = max_samples
        self.validate = validate

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        if self.format not in ["json", "jsonl", "csv"]:
            raise ValueError(f"Unsupported format: {self.format}. Use: json, jsonl, csv")

    def load(self) -> List[Dict[str, str]]:
        """
        Load data from file.

        Returns:
            List of dicts with keys: image_path, caption, (optional) id
        """
        logger.info(f"Loading {self.format.upper()} data from {self.data_file}")

        if self.format == "json":
            examples = self._load_json()
        elif self.format == "jsonl":
            examples = self._load_jsonl()
        elif self.format == "csv":
            examples = self._load_csv()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Apply max_samples limit
        if self.max_samples is not None and len(examples) > self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples (from {len(examples)})")
            examples = examples[:self.max_samples]

        # Resolve image paths
        examples = self._resolve_image_paths(examples)

        # Validate if requested
        if self.validate:
            examples = self._validate_examples(examples)

        logger.info(f"✓ Loaded {len(examples)} examples")
        return examples

    def _load_json(self) -> List[Dict]:
        """Load from JSON file."""
        with open(self.data_file, 'r') as f:
            data = json.load(f)

        # Support multiple JSON structures
        if isinstance(data, list):
            # Direct list of examples
            return data
        elif isinstance(data, dict):
            if "data" in data:
                # {"data": [...]}
                return data["data"]
            elif "examples" in data:
                # {"examples": [...]}
                return data["examples"]
            elif "samples" in data:
                # {"samples": [...]}
                return data["samples"]
            else:
                raise ValueError(
                    "JSON file must be a list or dict with 'data'/'examples'/'samples' key"
                )
        else:
            raise ValueError("JSON file must contain list or dict")

    def _load_jsonl(self) -> List[Dict]:
        """Load from JSONL file (one JSON per line)."""
        examples = []
        with open(self.data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

        return examples

    def _load_csv(self) -> List[Dict]:
        """Load from CSV file."""
        examples = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Verify required columns
            if 'image_path' not in reader.fieldnames:
                raise ValueError("CSV must have 'image_path' column")
            if 'caption' not in reader.fieldnames:
                raise ValueError("CSV must have 'caption' column")

            for row_num, row in enumerate(reader, 2):  # Start at 2 (header is 1)
                if not row['image_path'] or not row['caption']:
                    logger.warning(f"Skipping empty row {row_num}")
                    continue

                examples.append({
                    'image_path': row['image_path'].strip(),
                    'caption': row['caption'].strip(),
                    'id': row.get('id', f'csv_row_{row_num}')
                })

        return examples

    def _resolve_image_paths(self, examples: List[Dict]) -> List[Dict]:
        """Resolve image paths (absolute vs relative)."""
        for ex in examples:
            img_path = Path(ex['image_path'])

            # If path is relative and image_dir provided, make it absolute
            if not img_path.is_absolute() and self.image_dir:
                ex['image_path'] = str(self.image_dir / img_path)
            else:
                ex['image_path'] = str(img_path)

        return examples

    def _validate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Validate examples and filter out invalid ones."""
        valid_examples = []
        errors = []

        for i, ex in enumerate(examples):
            # Check required fields
            if 'image_path' not in ex:
                errors.append(f"Example {i}: Missing 'image_path'")
                continue

            if 'caption' not in ex:
                errors.append(f"Example {i}: Missing 'caption'")
                continue

            # Check caption is not empty
            if not ex['caption'].strip():
                errors.append(f"Example {i}: Empty caption")
                continue

            # Check image file exists
            img_path = Path(ex['image_path'])
            if not img_path.exists():
                errors.append(f"Example {i}: Image not found: {img_path}")
                continue

            # Try to load image
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                errors.append(f"Example {i}: Cannot load image {img_path}: {e}")
                continue

            valid_examples.append(ex)

        # Log validation results
        if errors:
            logger.warning(f"Validation found {len(errors)} invalid examples:")
            for error in errors[:10]:  # Show first 10
                logger.warning(f"  {error}")
            if len(errors) > 10:
                logger.warning(f"  ... and {len(errors) - 10} more errors")

        logger.info(f"Validation: {len(valid_examples)}/{len(examples)} examples valid")
        return valid_examples


class CustomPreferenceLoader:
    """
    Load custom preference datasets for DPO training.

    Supports:
    - JSON: {"preferences": [{"image_path": "...", "chosen": "...", "rejected": "..."}]}
    - JSONL: One preference per line
    - CSV: image_path,chosen,rejected
    - Ranked: {"ranked_data": [{"image_path": "...", "captions": [...]}]}

    Example:
        >>> loader = CustomPreferenceLoader(
        ...     data_file="preferences.json",
        ...     image_dir="images/",
        ...     format="json"
        ... )
        >>> prefs = loader.load()
        >>> print(len(prefs))
        500
    """

    def __init__(
        self,
        data_file: str,
        image_dir: Optional[str] = None,
        format: str = "json",
        max_samples: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Initialize preference data loader.

        Args:
            data_file: Path to preference data file
            image_dir: Base directory for image paths
            format: Data format: "json", "jsonl", "csv", "ranked"
            max_samples: Maximum number of samples to load
            validate: Whether to validate data on load
        """
        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir) if image_dir else None
        self.format = format.lower()
        self.max_samples = max_samples
        self.validate = validate

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

    def load(self) -> List[Dict[str, str]]:
        """
        Load preference data.

        Returns:
            List of dicts with keys: image_path, chosen, rejected, (optional) id
        """
        logger.info(f"Loading {self.format.upper()} preference data from {self.data_file}")

        if self.format == "json":
            preferences = self._load_json()
        elif self.format == "jsonl":
            preferences = self._load_jsonl()
        elif self.format == "csv":
            preferences = self._load_csv()
        elif self.format == "ranked":
            preferences = self._load_ranked()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Apply max_samples limit
        if self.max_samples is not None and len(preferences) > self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples (from {len(preferences)})")
            preferences = preferences[:self.max_samples]

        # Resolve image paths
        preferences = self._resolve_image_paths(preferences)

        # Validate if requested
        if self.validate:
            preferences = self._validate_preferences(preferences)

        logger.info(f"✓ Loaded {len(preferences)} preference pairs")
        return preferences

    def _load_json(self) -> List[Dict]:
        """Load preferences from JSON."""
        with open(self.data_file, 'r') as f:
            data = json.load(f)

        # Support multiple structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "preferences" in data:
                return data["preferences"]
            elif "data" in data:
                return data["data"]
            else:
                raise ValueError("JSON must have 'preferences' or 'data' key")
        else:
            raise ValueError("JSON must be list or dict")

    def _load_jsonl(self) -> List[Dict]:
        """Load preferences from JSONL."""
        preferences = []
        with open(self.data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    pref = json.loads(line)
                    preferences.append(pref)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

        return preferences

    def _load_csv(self) -> List[Dict]:
        """Load preferences from CSV."""
        preferences = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Verify required columns
            required = ['image_path', 'chosen', 'rejected']
            for col in required:
                if col not in reader.fieldnames:
                    raise ValueError(f"CSV must have '{col}' column")

            for row_num, row in enumerate(reader, 2):
                if not all([row['image_path'], row['chosen'], row['rejected']]):
                    logger.warning(f"Skipping incomplete row {row_num}")
                    continue

                preferences.append({
                    'image_path': row['image_path'].strip(),
                    'chosen': row['chosen'].strip(),
                    'rejected': row['rejected'].strip(),
                    'id': row.get('id', f'csv_row_{row_num}')
                })

        return preferences

    def _load_ranked(self) -> List[Dict]:
        """Load from ranked captions and create preference pairs."""
        with open(self.data_file, 'r') as f:
            data = json.load(f)

        if "ranked_data" not in data:
            raise ValueError("Ranked format must have 'ranked_data' key")

        preferences = []
        for item in data["ranked_data"]:
            image_path = item["image_path"]
            captions = item["captions"]

            # Sort by rank
            captions_sorted = sorted(captions, key=lambda x: x["rank"])

            # Create pairs: best vs each worse caption
            best = captions_sorted[0]["text"]
            for worse in captions_sorted[1:]:
                preferences.append({
                    "image_path": image_path,
                    "chosen": best,
                    "rejected": worse["text"],
                    "rank_diff": worse["rank"] - captions_sorted[0]["rank"]
                })

        logger.info(f"Created {len(preferences)} preference pairs from {len(data['ranked_data'])} ranked examples")
        return preferences

    def _resolve_image_paths(self, preferences: List[Dict]) -> List[Dict]:
        """Resolve image paths."""
        for pref in preferences:
            img_path = Path(pref['image_path'])

            if not img_path.is_absolute() and self.image_dir:
                pref['image_path'] = str(self.image_dir / img_path)
            else:
                pref['image_path'] = str(img_path)

        return preferences

    def _validate_preferences(self, preferences: List[Dict]) -> List[Dict]:
        """Validate preference pairs."""
        valid_prefs = []
        errors = []

        for i, pref in enumerate(preferences):
            # Check required fields
            required = ['image_path', 'chosen', 'rejected']
            missing = [f for f in required if f not in pref]
            if missing:
                errors.append(f"Preference {i}: Missing fields: {missing}")
                continue

            # Check captions not empty
            if not pref['chosen'].strip() or not pref['rejected'].strip():
                errors.append(f"Preference {i}: Empty caption")
                continue

            # Check chosen != rejected
            if pref['chosen'].strip() == pref['rejected'].strip():
                errors.append(f"Preference {i}: Chosen and rejected are identical")
                continue

            # Check image exists
            img_path = Path(pref['image_path'])
            if not img_path.exists():
                errors.append(f"Preference {i}: Image not found: {img_path}")
                continue

            valid_prefs.append(pref)

        if errors:
            logger.warning(f"Validation found {len(errors)} invalid preferences:")
            for error in errors[:10]:
                logger.warning(f"  {error}")
            if len(errors) > 10:
                logger.warning(f"  ... and {len(errors) - 10} more errors")

        logger.info(f"Validation: {len(valid_prefs)}/{len(preferences)} preferences valid")
        return valid_prefs


def load_custom_image_caption_data(
    train_file: str,
    val_file: Optional[str] = None,
    image_dir: Optional[str] = None,
    format: str = "json",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    validate: bool = True,
) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
    """
    Convenience function to load image-caption data.

    Args:
        train_file: Path to training data file
        val_file: Path to validation data file (optional)
        image_dir: Base directory for images
        format: Data format (json, jsonl, csv)
        max_train_samples: Max training samples
        max_val_samples: Max validation samples
        validate: Whether to validate data

    Returns:
        If val_file provided: (train_data, val_data)
        Otherwise: train_data

    Example:
        >>> train, val = load_custom_image_caption_data(
        ...     train_file="train.json",
        ...     val_file="val.json",
        ...     image_dir="images/",
        ...     format="json"
        ... )
    """
    # Load training data
    train_loader = CustomImageCaptionLoader(
        data_file=train_file,
        image_dir=image_dir,
        format=format,
        max_samples=max_train_samples,
        validate=validate
    )
    train_data = train_loader.load()

    # Load validation data if provided
    if val_file:
        val_loader = CustomImageCaptionLoader(
            data_file=val_file,
            image_dir=image_dir,
            format=format,
            max_samples=max_val_samples,
            validate=validate
        )
        val_data = val_loader.load()
        return train_data, val_data

    return train_data


def load_custom_preference_data(
    data_file: str,
    image_dir: Optional[str] = None,
    format: str = "json",
    max_samples: Optional[int] = None,
    validate: bool = True,
) -> List[Dict]:
    """
    Convenience function to load preference data for DPO.

    Args:
        data_file: Path to preference data file
        image_dir: Base directory for images
        format: Data format (json, jsonl, csv, ranked)
        max_samples: Max samples to load
        validate: Whether to validate data

    Returns:
        List of preference dicts with image_path, chosen, rejected

    Example:
        >>> prefs = load_custom_preference_data(
        ...     data_file="preferences.json",
        ...     image_dir="images/",
        ...     format="json"
        ... )
    """
    loader = CustomPreferenceLoader(
        data_file=data_file,
        image_dir=image_dir,
        format=format,
        max_samples=max_samples,
        validate=validate
    )
    return loader.load()
