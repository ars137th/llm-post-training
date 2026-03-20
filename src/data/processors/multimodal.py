"""
Multimodal Data Processors

Handles image-text datasets for vision-language training:
- Image captioning (COCO, Flickr)
- Visual question answering (VQA)
- Image-text retrieval
- Multimodal instruction following

Converts various dataset formats into unified structure for training.
"""

import torch
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from PIL import Image
import io
import base64
from transformers import PreTrainedTokenizer, ProcessorMixin


@dataclass
class MultimodalExample:
    """Unified format for multimodal examples."""
    image: Image.Image
    text: str
    caption: Optional[str] = None  # For captioning tasks
    question: Optional[str] = None  # For VQA tasks
    answer: Optional[str] = None  # For VQA tasks
    metadata: Optional[Dict] = None


class MultimodalDataProcessor:
    """
    Process multimodal datasets into training format.

    Supports:
    - COCO Captions
    - Flickr30k
    - Conceptual Captions
    - Custom image-text pairs
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[ProcessorMixin] = None,
        max_length: int = 512,
        image_size: int = 224,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size

    def load_coco_captions(
        self,
        split: str = "train",
        num_examples: Optional[int] = None,
    ) -> List[MultimodalExample]:
        """
        Load COCO Captions dataset.

        Args:
            split: "train" or "validation"
            num_examples: Limit number of examples

        Returns:
            List of MultimodalExample objects
        """
        print(f"Loading COCO Captions ({split})...")

        # COCO is available through HuggingFace datasets
        dataset = load_dataset("HuggingFaceM4/COCO", split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        examples = []
        for item in dataset:
            # COCO has multiple captions per image, use first one
            caption = item['sentences']['raw'][0] if 'sentences' in item else item.get('caption', '')

            examples.append(MultimodalExample(
                image=item['image'],
                text=caption,
                caption=caption,
                metadata={'image_id': item.get('cocoid', None)},
            ))

        print(f"✓ Loaded {len(examples)} COCO examples")
        return examples

    def load_flickr30k(
        self,
        split: str = "train",
        num_examples: Optional[int] = None,
    ) -> List[MultimodalExample]:
        """Load Flickr30k dataset."""
        print(f"Loading Flickr30k ({split})...")

        dataset = load_dataset("nlphuji/flickr30k", split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        examples = []
        for item in dataset:
            # Flickr has multiple captions, use first one
            caption = item['caption'][0] if isinstance(item['caption'], list) else item['caption']

            examples.append(MultimodalExample(
                image=item['image'],
                text=caption,
                caption=caption,
                metadata={'image_id': item.get('img_id', None)},
            ))

        print(f"✓ Loaded {len(examples)} Flickr30k examples")
        return examples

    def load_conceptual_captions(
        self,
        split: str = "train",
        num_examples: Optional[int] = None,
    ) -> List[MultimodalExample]:
        """Load Conceptual Captions dataset."""
        print(f"Loading Conceptual Captions ({split})...")

        dataset = load_dataset("conceptual_captions", split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        examples = []
        for item in dataset:
            examples.append(MultimodalExample(
                image=item['image'],
                text=item['caption'],
                caption=item['caption'],
            ))

        print(f"✓ Loaded {len(examples)} Conceptual Captions examples")
        return examples

    def create_synthetic_data(
        self,
        num_examples: int = 100,
    ) -> List[MultimodalExample]:
        """
        Create synthetic multimodal data for testing.

        Generates simple colored images with descriptive captions.
        Useful for quick validation without downloading large datasets.
        """
        import numpy as np

        print(f"Generating {num_examples} synthetic multimodal examples...")

        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        shapes = ["square", "circle", "triangle"]
        examples = []

        for i in range(num_examples):
            # Generate random colored image
            color_idx = i % len(colors)
            shape_idx = (i // len(colors)) % len(shapes)

            color = colors[color_idx]
            shape = shapes[shape_idx]

            # Create simple colored image (RGB)
            color_map = {
                "red": [255, 0, 0],
                "blue": [0, 0, 255],
                "green": [0, 255, 0],
                "yellow": [255, 255, 0],
                "purple": [128, 0, 128],
                "orange": [255, 165, 0],
            }

            img_array = np.ones((224, 224, 3), dtype=np.uint8) * color_map[color]
            image = Image.fromarray(img_array)

            # Generate caption
            captions = [
                f"A {color} {shape}",
                f"This is a {color} colored {shape}",
                f"An image showing a {color} {shape}",
            ]
            caption = captions[i % len(captions)]

            examples.append(MultimodalExample(
                image=image,
                text=caption,
                caption=caption,
                metadata={'synthetic': True, 'color': color, 'shape': shape},
            ))

        print(f"✓ Generated {len(examples)} synthetic examples")
        return examples

    def create_instruction_data(
        self,
        examples: List[MultimodalExample],
        instruction_template: str = "Describe this image:",
    ) -> List[Dict]:
        """
        Convert captioning data to instruction format.

        Args:
            examples: Multimodal examples
            instruction_template: Prompt template

        Returns:
            List of dicts with image, instruction, and response
        """
        instruction_data = []

        for ex in examples:
            instruction_data.append({
                'image': ex.image,
                'instruction': instruction_template,
                'response': ex.caption or ex.text,
                'metadata': ex.metadata,
            })

        return instruction_data

    def create_preference_pairs(
        self,
        examples: List[MultimodalExample],
        augment_negatives: bool = True,
    ) -> List[Dict]:
        """
        Create preference pairs for reward modeling/DPO.

        Args:
            examples: Multimodal examples with captions
            augment_negatives: Whether to create synthetic negative examples

        Returns:
            List of preference pairs (image, chosen_caption, rejected_caption)
        """
        pairs = []

        if augment_negatives:
            # Simple augmentation: shuffle captions for negatives
            for i, ex in enumerate(examples):
                # Positive: original caption
                chosen = ex.caption or ex.text

                # Negative: caption from different image
                negative_idx = (i + 1) % len(examples)
                rejected = examples[negative_idx].caption or examples[negative_idx].text

                pairs.append({
                    'image': ex.image,
                    'chosen': chosen,
                    'rejected': rejected,
                    'metadata': ex.metadata,
                })

        print(f"✓ Created {len(pairs)} preference pairs")
        return pairs

    def tokenize_multimodal_batch(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize batch of images and texts.

        Args:
            images: List of PIL images
            texts: List of text strings

        Returns:
            Dict with pixel_values and text tokens
        """
        if self.processor is not None:
            # Use processor (for models like CLIP, LLaVA)
            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        elif self.tokenizer is not None:
            # Separate image and text processing
            # (Image processing would need separate vision model)
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            # Note: Would need vision encoder to process images
            inputs['pixel_values'] = None  # Placeholder
        else:
            raise ValueError("Either processor or tokenizer must be provided")

        return inputs


def create_multimodal_collator(
    processor: ProcessorMixin,
    max_length: int = 512,
):
    """
    Create data collator for multimodal training.

    Args:
        processor: Model processor (handles both image and text)
        max_length: Maximum sequence length

    Returns:
        Collator function for DataLoader
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of multimodal examples."""
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]

        # Process batch
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add labels if present
        if 'labels' in batch[0]:
            inputs['labels'] = torch.tensor([item['labels'] for item in batch])

        return inputs

    return collate_fn


def load_multimodal_dataset(
    dataset_name: str,
    split: str = "train",
    num_examples: Optional[int] = None,
    processor: Optional[MultimodalDataProcessor] = None,
) -> List[MultimodalExample]:
    """
    Load multimodal dataset by name.

    Args:
        dataset_name: "coco", "flickr30k", "conceptual", or "synthetic"
        split: Dataset split
        num_examples: Limit number of examples
        processor: Data processor instance

    Returns:
        List of multimodal examples
    """
    if processor is None:
        processor = MultimodalDataProcessor()

    dataset_name = dataset_name.lower()

    if dataset_name == "coco":
        return processor.load_coco_captions(split=split, num_examples=num_examples)
    elif dataset_name in ["flickr30k", "flickr"]:
        return processor.load_flickr30k(split=split, num_examples=num_examples)
    elif dataset_name == "conceptual":
        return processor.load_conceptual_captions(split=split, num_examples=num_examples)
    elif dataset_name == "synthetic":
        num_examples = num_examples or 100
        return processor.create_synthetic_data(num_examples=num_examples)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: 'coco', 'flickr30k', 'conceptual', 'synthetic'"
        )


if __name__ == "__main__":
    # Test processor
    print("Testing multimodal data processor...")

    processor = MultimodalDataProcessor()

    # Test synthetic data
    examples = processor.create_synthetic_data(num_examples=10)
    print(f"✓ Created {len(examples)} synthetic examples")
    print(f"  Example: {examples[0].caption}")

    # Test instruction formatting
    instruction_data = processor.create_instruction_data(examples[:5])
    print(f"✓ Created {len(instruction_data)} instruction examples")

    # Test preference pairs
    pairs = processor.create_preference_pairs(examples[:5])
    print(f"✓ Created {len(pairs)} preference pairs")

    print("\n✓ All multimodal data processors working!")
