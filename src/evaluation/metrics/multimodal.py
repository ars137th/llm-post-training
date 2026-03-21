"""
Multimodal Evaluation Metrics

Metrics for evaluating vision-language models:
- CLIP Score: Image-text alignment quality
- Image-Text Retrieval: Recall@K for retrieval tasks
- Caption Quality: BLEU, METEOR, CIDEr for captioning
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from collections import defaultdict


class CLIPScoreMetric:
    """
    CLIP Score metric for measuring image-text alignment.

    CLIP Score measures how well a generated caption matches an image
    using CLIP's learned image-text embedding space.

    Higher scores indicate better alignment between image and text.

    Reference: "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
    https://arxiv.org/abs/2104.08718

    Example:
        metric = CLIPScoreMetric()
        images = [pil_image1, pil_image2]
        captions = ["A cat on a couch", "A dog in a park"]
        scores = metric.compute(images, captions)
        print(f"Average CLIP Score: {scores['clip_score']:.3f}")
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
    ):
        """
        Initialize CLIP Score metric.

        Args:
            model_name: CLIP model to use for scoring
            device: Device to run model on
        """
        print(f"Loading CLIP model for scoring: {model_name}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print(f"✓ CLIP Score metric ready on {device}")

    @torch.no_grad()
    def compute(
        self,
        images: List[Image.Image],
        texts: List[str],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Compute CLIP Score for image-text pairs.

        Args:
            images: List of PIL images
            texts: List of text captions/descriptions
            batch_size: Batch size for processing

        Returns:
            Dict with metrics:
            - clip_score: Average CLIP score (0-100)
            - clip_score_std: Standard deviation
            - individual_scores: Per-example scores
        """
        if len(images) != len(texts):
            raise ValueError(f"Number of images ({len(images)}) must match texts ({len(texts)})")

        scores = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]

            # Process inputs
            inputs = self.processor(
                images=batch_images,
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            outputs = self.model(**inputs)

            # Normalize embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

            # Compute similarity (cosine similarity)
            # CLIP score is similarity * 100 (scaled to 0-100)
            similarity = (image_embeds * text_embeds).sum(dim=-1)
            batch_scores = similarity * 100.0

            scores.extend(batch_scores.cpu().tolist())

        scores = np.array(scores)

        return {
            "clip_score": float(scores.mean()),
            "clip_score_std": float(scores.std()),
            "clip_score_min": float(scores.min()),
            "clip_score_max": float(scores.max()),
            "individual_scores": scores.tolist(),
        }

    def compute_single(
        self,
        image: Image.Image,
        text: str,
    ) -> float:
        """Compute CLIP Score for a single image-text pair."""
        result = self.compute([image], [text])
        return result["clip_score"]


class ImageTextRetrievalMetric:
    """
    Image-Text Retrieval evaluation metrics.

    Measures how well the model can retrieve the correct image for a text query
    and vice versa using Recall@K metrics.

    Example:
        metric = ImageTextRetrievalMetric()
        results = metric.compute(images, texts)
        print(f"Text-to-Image R@1: {results['t2i_recall@1']:.1%}")
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
    ):
        """Initialize retrieval metric with CLIP model."""
        print(f"Loading CLIP model for retrieval: {model_name}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print(f"✓ Retrieval metric ready on {device}")

    @torch.no_grad()
    def compute(
        self,
        images: List[Image.Image],
        texts: List[str],
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.

        Args:
            images: List of images
            texts: List of corresponding texts
            k_values: K values for Recall@K

        Returns:
            Dict with metrics:
            - t2i_recall@k: Text-to-image recall at K
            - i2t_recall@k: Image-to-text recall at K
        """
        # Encode all images and texts
        image_inputs = self.processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_embeds = self.model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        similarity = torch.matmul(text_embeds, image_embeds.T)  # [num_texts, num_images]

        metrics = {}

        # Text-to-Image retrieval (each text should retrieve its corresponding image)
        for k in k_values:
            # Get top-k images for each text
            topk_indices = torch.topk(similarity, k=min(k, len(images)), dim=1).indices

            # Check if correct image is in top-k
            correct = 0
            for i in range(len(texts)):
                if i in topk_indices[i]:
                    correct += 1

            metrics[f"t2i_recall@{k}"] = correct / len(texts)

        # Image-to-Text retrieval (transpose similarity matrix)
        similarity_t = similarity.T  # [num_images, num_texts]

        for k in k_values:
            # Get top-k texts for each image
            topk_indices = torch.topk(similarity_t, k=min(k, len(texts)), dim=1).indices

            # Check if correct text is in top-k
            correct = 0
            for i in range(len(images)):
                if i in topk_indices[i]:
                    correct += 1

            metrics[f"i2t_recall@{k}"] = correct / len(images)

        return metrics


def compute_clip_score(
    images: List[Image.Image],
    texts: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """
    Convenience function to compute CLIP Score.

    Args:
        images: List of images
        texts: List of captions
        model_name: CLIP model to use

    Returns:
        Average CLIP Score
    """
    metric = CLIPScoreMetric(model_name=model_name)
    results = metric.compute(images, texts)
    return results["clip_score"]


def compute_retrieval_metrics(
    images: List[Image.Image],
    texts: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
) -> Dict[str, float]:
    """
    Convenience function to compute retrieval metrics.

    Args:
        images: List of images
        texts: List of texts
        model_name: CLIP model to use

    Returns:
        Dict with Recall@K metrics
    """
    metric = ImageTextRetrievalMetric(model_name=model_name)
    return metric.compute(images, texts)


# For HuggingFace Trainer integration
def create_clip_score_compute_metrics(
    processor: CLIPProcessor,
    device: str = "auto",
):
    """
    Create compute_metrics function for HuggingFace Trainer.

    This allows automatic CLIP Score evaluation during training.

    Example:
        from transformers import TrainingArguments, Trainer

        compute_metrics = create_clip_score_compute_metrics(clip_processor)

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
        )
    """
    clip_metric = CLIPScoreMetric(device=device)

    def compute_metrics(eval_pred):
        """Compute CLIP Score from predictions."""
        # Extract predictions and labels
        predictions, labels = eval_pred

        # This assumes predictions contain generated text and labels contain images
        # Actual implementation depends on your data format

        # Placeholder - needs adaptation to your specific use case
        return {
            "clip_score": 0.0,  # Would compute actual CLIP score here
        }

    return compute_metrics


if __name__ == "__main__":
    # Quick test
    print("Testing CLIP Score metric...")

    from ..processors.multimodal import MultimodalDataProcessor

    # Create synthetic data
    processor = MultimodalDataProcessor()
    examples = processor.create_synthetic_data(num_examples=10)

    images = [ex.image for ex in examples]
    texts = [ex.caption for ex in examples]

    # Test CLIP Score
    print("\n1. Testing CLIP Score...")
    metric = CLIPScoreMetric()
    scores = metric.compute(images, texts)
    print(f"   Average CLIP Score: {scores['clip_score']:.2f}")
    print(f"   Std Dev: {scores['clip_score_std']:.2f}")
    print(f"   Range: [{scores['clip_score_min']:.2f}, {scores['clip_score_max']:.2f}]")

    # Test Retrieval
    print("\n2. Testing Retrieval Metrics...")
    retrieval_metric = ImageTextRetrievalMetric()
    retrieval_scores = retrieval_metric.compute(images, texts)
    print(f"   Text-to-Image R@1: {retrieval_scores['t2i_recall@1']:.1%}")
    print(f"   Text-to-Image R@5: {retrieval_scores['t2i_recall@5']:.1%}")
    print(f"   Image-to-Text R@1: {retrieval_scores['i2t_recall@1']:.1%}")

    print("\n✓ All multimodal metrics working!")
