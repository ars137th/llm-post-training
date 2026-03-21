#!/usr/bin/env python3
"""
Evaluate Multimodal Models

Compute evaluation metrics for vision-language models:
- CLIP Score: Image-text alignment quality
- Retrieval Metrics: Recall@K for image-text retrieval
- Generation Quality: For LLaVA-style models

Usage:
    # Evaluate CLIP model
    python scripts/evaluate/evaluate_multimodal.py \
        model_path=./outputs/clip_caption \
        model_type=clip \
        dataset=synthetic

    # Evaluate LLaVA model
    python scripts/evaluate/evaluate_multimodal.py \
        model_path=./outputs/llava_instruction \
        model_type=llava \
        dataset=coco
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from typing import List
import argparse
from tqdm import tqdm

from src.models.vision_language import create_vision_language_model
from src.data.processors.multimodal import MultimodalDataProcessor
from src.evaluation.metrics.multimodal import (
    CLIPScoreMetric,
    ImageTextRetrievalMetric,
    compute_clip_score,
    compute_retrieval_metrics,
)


def evaluate_clip(
    model_path: str,
    dataset: str = "synthetic",
    num_examples: int = 500,
    device: str = "auto",
):
    """
    Evaluate CLIP model on image-text alignment.

    Args:
        model_path: Path to trained model
        dataset: Dataset to evaluate on
        num_examples: Number of examples to evaluate
        device: Device to run on
    """
    print("=" * 80)
    print("EVALUATING CLIP MODEL")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset}")
    print(f"Num examples: {num_examples}")
    print("=" * 80 + "\n")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("Loading data...")
    processor = MultimodalDataProcessor()

    if dataset == "synthetic":
        examples = processor.create_synthetic_data(num_examples=num_examples)
    elif dataset == "coco":
        examples = processor.load_coco_captions(split="validation", num_examples=num_examples)
    elif dataset == "flickr30k":
        examples = processor.load_flickr30k(split="test", num_examples=num_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    images = [ex.image for ex in examples]
    texts = [ex.caption for ex in examples]

    print(f"Loaded {len(examples)} examples\n")

    # Evaluate CLIP Score
    print("1. Computing CLIP Score...")
    clip_metric = CLIPScoreMetric(model_name=model_path, device=device)
    scores = clip_metric.compute(images, texts, batch_size=32)

    print(f"\nCLIP Score Results:")
    print(f"  Average: {scores['clip_score']:.2f}")
    print(f"  Std Dev: {scores['clip_score_std']:.2f}")
    print(f"  Min: {scores['clip_score_min']:.2f}")
    print(f"  Max: {scores['clip_score_max']:.2f}")

    # Evaluate Retrieval
    print("\n2. Computing Retrieval Metrics...")
    retrieval_metric = ImageTextRetrievalMetric(model_name=model_path, device=device)
    retrieval_scores = retrieval_metric.compute(images, texts, k_values=[1, 5, 10])

    print(f"\nRetrieval Results:")
    print(f"  Text-to-Image R@1: {retrieval_scores['t2i_recall@1']:.1%}")
    print(f"  Text-to-Image R@5: {retrieval_scores['t2i_recall@5']:.1%}")
    print(f"  Text-to-Image R@10: {retrieval_scores['t2i_recall@10']:.1%}")
    print(f"  Image-to-Text R@1: {retrieval_scores['i2t_recall@1']:.1%}")
    print(f"  Image-to-Text R@5: {retrieval_scores['i2t_recall@5']:.1%}")
    print(f"  Image-to-Text R@10: {retrieval_scores['i2t_recall@10']:.1%}")

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)

    return {
        'clip_score': scores,
        'retrieval': retrieval_scores,
    }


def evaluate_llava(
    model_path: str,
    dataset: str = "synthetic",
    num_examples: int = 100,
    device: str = "auto",
):
    """
    Evaluate LLaVA model on instruction following.

    Args:
        model_path: Path to trained model
        dataset: Dataset to evaluate on
        num_examples: Number of examples to evaluate
        device: Device to run on
    """
    print("=" * 80)
    print("EVALUATING LLAVA MODEL")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset}")
    print(f"Num examples: {num_examples}")
    print("=" * 80 + "\n")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading model...")
    model_wrapper = create_vision_language_model(
        model_type="llava",
        model_name=model_path,
        device=device,
    )

    # Load data
    print("Loading data...")
    processor = MultimodalDataProcessor()

    if dataset == "synthetic":
        examples = processor.create_synthetic_data(num_examples=num_examples)
    elif dataset == "coco":
        examples = processor.load_coco_captions(split="validation", num_examples=num_examples)
    elif dataset == "flickr30k":
        examples = processor.load_flickr30k(split="test", num_examples=num_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Loaded {len(examples)} examples\n")

    # Generate responses
    print("Generating responses...")
    prompts = ["Describe this image in detail:"] * len(examples)
    images = [ex.image for ex in examples]
    ground_truth = [ex.caption for ex in examples]

    generated_texts = []
    for i in tqdm(range(0, len(examples), 8)):  # Batch size 8
        batch_images = images[i:i+8]
        batch_prompts = prompts[i:i+8]

        outputs = model_wrapper.generate(
            images=batch_images,
            prompts=batch_prompts,
            max_new_tokens=100,
            temperature=0.7,
        )

        generated_texts.extend(outputs)

    # Compute CLIP Score between images and generated captions
    print("\nComputing CLIP Score for generated captions...")
    clip_metric = CLIPScoreMetric(device=device)
    scores = clip_metric.compute(images, generated_texts, batch_size=32)

    print(f"\nCLIP Score Results (Generated):")
    print(f"  Average: {scores['clip_score']:.2f}")
    print(f"  Std Dev: {scores['clip_score_std']:.2f}")

    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE GENERATIONS")
    print("=" * 80)

    num_samples = min(5, len(examples))
    for i in range(num_samples):
        print(f"\nExample {i+1}:")
        print(f"  Ground Truth: {ground_truth[i]}")
        print(f"  Generated: {generated_texts[i]}")
        print(f"  CLIP Score: {scores['individual_scores'][i]:.2f}")

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)

    return {
        'clip_score': scores,
        'generated_texts': generated_texts,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_type", type=str, required=True, choices=["clip", "llava"],
                       help="Model type")
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "coco", "flickr30k"],
                       help="Dataset to evaluate on")
    parser.add_argument("--num_examples", type=int, default=500,
                       help="Number of examples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on")

    args = parser.parse_args()

    if args.model_type == "clip":
        results = evaluate_clip(
            model_path=args.model_path,
            dataset=args.dataset,
            num_examples=args.num_examples,
            device=args.device,
        )
    else:  # llava
        results = evaluate_llava(
            model_path=args.model_path,
            dataset=args.dataset,
            num_examples=args.num_examples,
            device=args.device,
        )


if __name__ == "__main__":
    main()
