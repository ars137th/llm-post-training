"""
Multimodal Data Collators

Data collators for batching image+text examples for vision-language models.
"""

from typing import Dict, List, Any, Optional, Union
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from PIL import Image


@dataclass
class MultimodalDataCollator:
    """
    Data collator for multimodal (image + text) models.

    Handles batching of examples containing both images and text,
    suitable for models like CLIP and LLaVA.

    Example:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        collator = MultimodalDataCollator(
            tokenizer=processor.tokenizer,
            image_processor=processor.image_processor,
        )

        batch = collator(examples)
        # Returns: {
        #   'pixel_values': Tensor[B, C, H, W],
        #   'input_ids': Tensor[B, L],
        #   'attention_mask': Tensor[B, L],
        #   'labels': Tensor[B, L]  (if available)
        # }
    """

    tokenizer: PreTrainedTokenizer
    image_processor: Any  # CLIPImageProcessor or similar
    max_length: int = 512
    padding: str = "max_length"
    return_tensors: str = "pt"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multimodal examples.

        Args:
            examples: List of dicts with keys:
                - 'image': PIL Image
                - 'text' or 'caption': Text string
                - 'labels' (optional): Label tensor

        Returns:
            Batch dict with 'pixel_values', 'input_ids', 'attention_mask', 'labels'
        """
        # Separate images and texts
        images = []
        texts = []
        labels = []
        has_labels = 'labels' in examples[0]

        for example in examples:
            # Get image
            if 'image' in example:
                images.append(example['image'])
            elif 'pixel_values' in example:
                # Already processed
                images.append(example['pixel_values'])
            else:
                raise ValueError("Example must contain 'image' or 'pixel_values'")

            # Get text
            if 'text' in example:
                texts.append(example['text'])
            elif 'caption' in example:
                texts.append(example['caption'])
            elif 'input_ids' in example:
                # Already tokenized - we'll handle this separately
                texts.append(None)
            else:
                raise ValueError("Example must contain 'text', 'caption', or 'input_ids'")

            # Get labels if present
            if has_labels:
                labels.append(example['labels'])

        # Process images
        if isinstance(images[0], Image.Image):
            # Raw PIL images - process them
            image_inputs = self.image_processor(
                images=images,
                return_tensors=self.return_tensors,
            )
            pixel_values = image_inputs['pixel_values']
        elif isinstance(images[0], torch.Tensor):
            # Already processed tensors
            pixel_values = torch.stack(images)
        else:
            raise ValueError(f"Unsupported image type: {type(images[0])}")

        # Process texts
        if texts[0] is not None:
            # Raw text - tokenize it
            text_inputs = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
            )
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
        else:
            # Already tokenized - stack them
            input_ids = torch.stack([ex['input_ids'] for ex in examples])
            attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        # Build batch
        batch = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        # Add labels
        if has_labels:
            if isinstance(labels[0], torch.Tensor):
                batch['labels'] = torch.stack(labels)
            else:
                # Use input_ids as labels for language modeling
                batch['labels'] = input_ids.clone()
        else:
            # No explicit labels - use input_ids for language modeling
            batch['labels'] = input_ids.clone()

        return batch


@dataclass
class CLIPDataCollator:
    """
    Specialized data collator for CLIP contrastive learning.

    CLIP learns by matching image-text pairs. This collator prepares
    batches where images[i] corresponds to texts[i].

    Example:
        collator = CLIPDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        batch = collator(examples)
        # Used for: similarity = model(pixel_values, input_ids)
    """

    tokenizer: PreTrainedTokenizer
    image_processor: Any
    max_length: int = 77  # CLIP's default context length
    padding: str = "max_length"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch for CLIP training.

        Args:
            examples: List of dicts with 'image' and 'text'/'caption'

        Returns:
            Batch with 'pixel_values' and 'input_ids'
        """
        images = [ex['image'] for ex in examples]

        # Get text from various possible keys
        texts = []
        for ex in examples:
            if 'text' in ex:
                texts.append(ex['text'])
            elif 'caption' in ex:
                texts.append(ex['caption'])
            else:
                raise ValueError("Example must contain 'text' or 'caption'")

        # Process images
        image_inputs = self.image_processor(images=images, return_tensors="pt")

        # Process texts
        text_inputs = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            'pixel_values': image_inputs['pixel_values'],
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
        }


@dataclass
class LLaVADataCollator:
    """
    Data collator for LLaVA instruction following.

    LLaVA is trained on (image, instruction, response) tuples.
    This collator formats them properly with:
    - Image → pixel_values
    - Instruction + Response → input_ids
    - Labels masked for instruction tokens (only predict response)

    Example:
        collator = LLaVADataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            instruction_template="Describe this image:",
        )
    """

    tokenizer: PreTrainedTokenizer
    image_processor: Any
    max_length: int = 2048
    instruction_template: Optional[str] = None
    ignore_index: int = -100  # Standard PyTorch ignore index

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch for LLaVA training.

        Args:
            examples: List of dicts with:
                - 'image': PIL Image
                - 'instruction': Instruction text
                - 'response': Response text

        Returns:
            Batch with pixel_values, input_ids, attention_mask, labels
            (labels have instruction tokens masked)
        """
        images = [ex['image'] for ex in examples]

        # Process images
        image_inputs = self.image_processor(images=images, return_tensors="pt")

        # Build instruction-response pairs
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for ex in examples:
            # Get instruction and response
            if 'instruction' in ex:
                instruction = ex['instruction']
            elif self.instruction_template:
                instruction = self.instruction_template
            else:
                instruction = "Describe this image:"

            response = ex.get('response', ex.get('caption', ''))

            # Tokenize instruction (we'll mask these in labels)
            instruction_tokens = self.tokenizer(
                instruction,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length // 2,
            )

            # Tokenize response (we'll learn to predict these)
            response_tokens = self.tokenizer(
                response,
                add_special_tokens=False,  # No special tokens in middle
                truncation=True,
                max_length=self.max_length // 2,
            )

            # Combine instruction + response
            input_ids = instruction_tokens['input_ids'] + response_tokens['input_ids']
            attention_mask = [1] * len(input_ids)

            # Create labels: mask instruction, predict response
            instruction_len = len(instruction_tokens['input_ids'])
            labels = [self.ignore_index] * instruction_len + response_tokens['input_ids']

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        # Pad sequences
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for input_ids, labels, attention_mask in zip(
            input_ids_list, labels_list, attention_mask_list
        ):
            padding_len = max_len - len(input_ids)

            # Pad to max_len
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * padding_len
            )
            padded_labels.append(
                labels + [self.ignore_index] * padding_len
            )
            padded_attention_mask.append(
                attention_mask + [0] * padding_len
            )

        return {
            'pixel_values': image_inputs['pixel_values'],
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }


@dataclass
class MultimodalDPODataCollator:
    """
    Data collator for multimodal DPO training.

    Batches preference pairs with images for Direct Preference Optimization.

    Example:
        collator = MultimodalDPODataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        batch = collator(preference_examples)
        # Returns: {
        #   'chosen_pixel_values', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
        #   'rejected_pixel_values', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'
        # }
    """

    tokenizer: PreTrainedTokenizer
    image_processor: Any
    max_length: int = 512
    padding: str = "max_length"
    ignore_index: int = -100

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate multimodal preference pairs for DPO.

        Args:
            examples: List of dicts with:
                - 'chosen_image': PIL Image for chosen
                - 'chosen_text': Text for chosen
                - 'rejected_image': PIL Image for rejected
                - 'rejected_text': Text for rejected

        Returns:
            Batch dict with chosen/rejected pixel_values, input_ids, attention_mask, labels
        """
        # Separate chosen and rejected
        chosen_images = [ex['chosen_image'] for ex in examples]
        chosen_texts = [ex['chosen_text'] for ex in examples]
        rejected_images = [ex['rejected_image'] for ex in examples]
        rejected_texts = [ex['rejected_text'] for ex in examples]

        # Process chosen images
        chosen_image_inputs = self.image_processor(
            images=chosen_images,
            return_tensors="pt",
        )

        # Process rejected images
        rejected_image_inputs = self.image_processor(
            images=rejected_images,
            return_tensors="pt",
        )

        # Process chosen texts
        chosen_text_inputs = self.tokenizer(
            chosen_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Process rejected texts
        rejected_text_inputs = self.tokenizer(
            rejected_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (for LLaVA-style models)
        # For CLIP, labels aren't used (DPO on similarity scores)
        chosen_labels = chosen_text_inputs['input_ids'].clone()
        rejected_labels = rejected_text_inputs['input_ids'].clone()

        return {
            'chosen_pixel_values': chosen_image_inputs['pixel_values'],
            'chosen_input_ids': chosen_text_inputs['input_ids'],
            'chosen_attention_mask': chosen_text_inputs['attention_mask'],
            'chosen_labels': chosen_labels,
            'rejected_pixel_values': rejected_image_inputs['pixel_values'],
            'rejected_input_ids': rejected_text_inputs['input_ids'],
            'rejected_attention_mask': rejected_text_inputs['attention_mask'],
            'rejected_labels': rejected_labels,
        }


def create_multimodal_collator(
    model_type: str,
    tokenizer: PreTrainedTokenizer,
    image_processor: Any,
    **kwargs,
):
    """
    Factory function to create appropriate collator for model type.

    Args:
        model_type: "clip", "llava", or "generic"
        tokenizer: Text tokenizer
        image_processor: Image processor
        **kwargs: Additional arguments for collator
            - max_length: Max sequence length
            - padding: Padding strategy
            - instruction_template: (LLaVA only) Instruction format
            - ignore_index: (LLaVA only) Label masking index

    Returns:
        Appropriate data collator

    Example:
        collator = create_multimodal_collator(
            model_type="clip",
            tokenizer=clip_processor.tokenizer,
            image_processor=clip_processor.image_processor,
        )
    """
    if model_type.lower() == "clip":
        # CLIP only accepts: max_length, padding
        clip_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'padding']
        }
        return CLIPDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **clip_kwargs,
        )
    elif model_type.lower() == "llava":
        # LLaVA accepts: max_length, instruction_template, ignore_index
        llava_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'instruction_template', 'ignore_index']
        }
        return LLaVADataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **llava_kwargs,
        )
    else:
        # Generic accepts: max_length, padding, return_tensors
        generic_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_length', 'padding', 'return_tensors']
        }
        return MultimodalDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **generic_kwargs,
        )


if __name__ == "__main__":
    # Test collators with synthetic data
    print("Testing multimodal data collators...\n")

    from transformers import CLIPProcessor
    from PIL import Image
    import numpy as np

    # Load CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Create synthetic examples
    examples = []
    for i in range(4):
        # Create a random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        examples.append({
            'image': img,
            'text': f"This is test image number {i}",
        })

    # Test CLIP collator
    print("1. Testing CLIPDataCollator...")
    clip_collator = CLIPDataCollator(
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
    )

    batch = clip_collator(examples)
    print(f"   pixel_values shape: {batch['pixel_values'].shape}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")

    # Test generic multimodal collator
    print("\n2. Testing MultimodalDataCollator...")
    generic_collator = MultimodalDataCollator(
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        max_length=128,
    )

    batch = generic_collator(examples)
    print(f"   pixel_values shape: {batch['pixel_values'].shape}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")

    # Test LLaVA collator
    print("\n3. Testing LLaVADataCollator...")
    llava_examples = [
        {
            'image': ex['image'],
            'instruction': "Describe this image:",
            'response': f"This is a test image with random pixels.",
        }
        for ex in examples
    ]

    llava_collator = LLaVADataCollator(
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
    )

    batch = llava_collator(llava_examples)
    print(f"   pixel_values shape: {batch['pixel_values'].shape}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")

    # Check that instruction tokens are masked
    num_masked = (batch['labels'][0] == -100).sum().item()
    print(f"   Masked tokens (instruction): {num_masked}")
    print(f"   Unmasked tokens (response): {(batch['labels'][0] != -100).sum().item()}")

    print("\n✓ All collators working!")
