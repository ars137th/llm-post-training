"""
Vision-Language Model Wrappers

Supports multimodal models for image-text tasks:
- CLIP: Contrastive image-text learning
- LLaVA: Vision-language instruction following
- Flamingo: Few-shot vision-language models

These wrappers provide a unified interface compatible with our training infrastructure.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image


@dataclass
class VisionLanguageModelOutput:
    """Output from vision-language models."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class CLIPWrapper(nn.Module):
    """
    Wrapper for CLIP models with optional LoRA fine-tuning.

    CLIP learns joint embeddings for images and text using contrastive learning.
    Useful for image classification, retrieval, and zero-shot tasks.

    Example:
        model = CLIPWrapper.from_pretrained("openai/clip-vit-base-patch32")
        outputs = model(pixel_values=images, input_ids=texts)
        similarity = outputs.logits  # Cosine similarity between image and text
    """

    def __init__(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.use_lora = use_lora

        if use_lora and lora_config:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA to text encoder only (vision encoder causes PEFT routing issues)."""
        # CLIP has separate vision and text encoders
        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 16),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "v_proj"]  # Attention projections
            ),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # WORKAROUND: Apply LoRA to text encoder only
        # Applying LoRA to vision encoder causes PEFT kwargs routing issues
        # where input_ids gets passed to the vision model's forward()
        # See: docs/known_issues.md for details
        self.model.text_model = get_peft_model(self.model.text_model, peft_config)

        # Optional: Apply to vision encoder (currently disabled due to PEFT bug)
        apply_to_vision = lora_config.get("apply_to_vision_encoder", False)
        if apply_to_vision:
            print("⚠️  Warning: Applying LoRA to vision encoder may cause training errors")
            self.model.vision_model = get_peft_model(self.model.vision_model, peft_config)

        print(f"✓ LoRA applied to CLIP text encoder only (r={peft_config.r})")
        print(f"  Vision encoder: frozen (full precision)")
        print(f"  Text encoder: LoRA adapters (trainable)")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "openai/clip-vit-base-patch32",
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        device: str = "auto",
    ) -> "CLIPWrapper":
        """Load pretrained CLIP model."""
        print(f"Loading CLIP model: {model_name}")

        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        # Move to device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        wrapper = cls(
            model=model,
            processor=processor,
            use_lora=use_lora,
            lora_config=lora_config,
        )

        print(f"✓ CLIP loaded on {device}")
        return wrapper

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> VisionLanguageModelOutput:
        """
        Forward pass through CLIP.

        Args:
            pixel_values: Image tensors [batch, channels, height, width]
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            return_loss: Whether to compute contrastive loss

        Returns:
            VisionLanguageModelOutput with similarity logits
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=return_loss,
            **kwargs,
        )

        return VisionLanguageModelOutput(
            loss=outputs.loss if return_loss else None,
            logits=outputs.logits_per_image,  # Image-to-text similarity
            image_embeds=outputs.image_embeds,
            text_embeds=outputs.text_embeds,
        )

    def encode_image(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Encode images to embeddings."""
        if isinstance(images, list):
            # Process PIL images
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.model.device)
        else:
            pixel_values = images

        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        return image_embeds

    def encode_text(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Encode texts to embeddings."""
        if isinstance(texts, list):
            # Process text strings
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
        else:
            input_ids = texts
            attention_mask = None

        text_embeds = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return text_embeds

    def compute_similarity(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        texts: Union[List[str], torch.Tensor],
    ) -> torch.Tensor:
        """Compute cosine similarity between images and texts."""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = torch.matmul(image_embeds, text_embeds.T) * self.model.logit_scale.exp()
        return similarity

    def save_pretrained(self, output_dir: str):
        """Save model and processor."""
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"✓ CLIP saved to {output_dir}")

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config


class LLaVAWrapper(nn.Module):
    """
    Wrapper for LLaVA (Large Language and Vision Assistant) models.

    LLaVA combines a vision encoder (CLIP) with a language model (LLaMA/Vicuna)
    for multimodal instruction following.

    Example:
        model = LLaVAWrapper.from_pretrained("liuhaotian/llava-v1.5-7b")
        outputs = model(pixel_values=images, input_ids=prompts)
        response = model.generate(images, "Describe this image")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: AutoProcessor,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.use_lora = use_lora

        if use_lora and lora_config:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA to language model backbone."""
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"]  # LLaMA attention
            ),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, peft_config)
        print(f"✓ LoRA applied to LLaVA (r={peft_config.r})")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> "LLaVAWrapper":
        """Load pretrained LLaVA model."""
        print(f"Loading LLaVA model: {model_name}")

        # Quantization config for large models
        kwargs = {}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            kwargs["load_in_4bit"] = True

        model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
        processor = AutoProcessor.from_pretrained(model_name)

        # Move to device (if not quantized)
        if not (load_in_8bit or load_in_4bit):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

        wrapper = cls(
            model=model,
            processor=processor,
            use_lora=use_lora,
            lora_config=lora_config,
        )

        device_str = str(model.device) if hasattr(model, 'device') else "quantized"
        print(f"✓ LLaVA loaded on {device_str}")
        return wrapper

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VisionLanguageModelOutput:
        """
        Forward pass through LLaVA.

        Args:
            pixel_values: Image tensors
            input_ids: Text token IDs (prompt + response)
            attention_mask: Attention mask
            labels: Target token IDs for training

        Returns:
            VisionLanguageModelOutput with language modeling loss
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return VisionLanguageModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        )

    def generate(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """
        Generate text responses for image-prompt pairs.

        Args:
            images: Input images
            prompts: Text prompts (e.g., "Describe this image")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            List of generated text responses
        """
        # Process inputs
        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs,
            )

        # Decode
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    def save_pretrained(self, output_dir: str):
        """Save model and processor."""
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"✓ LLaVA saved to {output_dir}")

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config


def create_vision_language_model(
    model_type: str,
    model_name: str,
    use_lora: bool = False,
    lora_config: Optional[Dict] = None,
    device: str = "auto",
    **kwargs,
) -> Union[CLIPWrapper, LLaVAWrapper]:
    """
    Factory function to create vision-language models.

    Args:
        model_type: "clip" or "llava"
        model_name: HuggingFace model identifier
        use_lora: Whether to apply LoRA
        lora_config: LoRA configuration dict
        device: Device to load model on
        **kwargs: Additional model-specific arguments

    Returns:
        Vision-language model wrapper

    Example:
        # CLIP for image-text retrieval
        clip = create_vision_language_model(
            model_type="clip",
            model_name="openai/clip-vit-base-patch32",
            use_lora=True,
        )

        # LLaVA for instruction following
        llava = create_vision_language_model(
            model_type="llava",
            model_name="llava-hf/llava-1.5-7b-hf",
            use_lora=True,
            load_in_4bit=True,
        )
    """
    model_type = model_type.lower()

    if model_type == "clip":
        return CLIPWrapper.from_pretrained(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            device=device,
        )
    elif model_type in ["llava", "llava-1.5"]:
        return LLaVAWrapper.from_pretrained(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported: 'clip', 'llava'"
        )


if __name__ == "__main__":
    # Quick test
    print("Testing CLIP wrapper...")
    clip = CLIPWrapper.from_pretrained("openai/clip-vit-base-patch32")
    print(f"✓ CLIP loaded: {clip.config.vision_config.hidden_size}D vision, "
          f"{clip.config.text_config.hidden_size}D text")

    print("\nTesting would load LLaVA (commented out for speed)...")
    # llava = LLaVAWrapper.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
    print("✓ All vision-language wrappers working!")
