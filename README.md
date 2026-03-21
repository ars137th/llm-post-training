# LLM Post-Training Experimentation Repository

A comprehensive, educational repository for learning and experimenting with state-of-the-art LLM post-training techniques including Supervised Fine-Tuning (SFT), Reward Modeling, RLHF (PPO), DPO, and IPO.

## Overview

This repository provides a **hybrid approach** to learning LLM alignment techniques:
- **Educational**: Custom implementations of core algorithms to understand how they work
- **Practical**: Leverages battle-tested frameworks (HuggingFace, PEFT, TRL) for infrastructure
- **Multimodal**: Supports both text-only and vision-language models
- **Scalable**: Designed for small models (< 1B params) that run on consumer hardware

## Key Features

- **Five Post-Training Techniques**:
  - Supervised Fine-Tuning (SFT)
  - Reward Modeling
  - RLHF with PPO (Proximal Policy Optimization)
  - DPO (Direct Preference Optimization)
  - IPO (Identity Preference Optimization)

- **Multimodal Support**:
  - Text-only models (GPT-2, LLaMA, Mistral, OPT)
  - Vision-language models (CLIP, LLaVA)
  - Unified interface for both modalities

- **Educational Tools**:
  - Detailed Jupyter notebooks explaining each technique
  - Side-by-side comparisons
  - Visualization utilities
  - Extensive documentation

- **Flexible Configuration**:
  - Hydra-based config management
  - Easy experiment sweeps
  - Modular architecture

## Quick Start

### Installation

**Platform-specific** installation (choose one):

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training

# For macOS (Apple Silicon) - Method 1: Use requirements file
pip install -r requirements/base.txt
pip install -e .

# For macOS (Apple Silicon) - Method 2: Use extras
pip install -e ".[macos]"

# For Linux/Colab/Cloud with GPU (uses latest PyTorch/transformers)
pip install -e ".[gpu]"
```

**📖 See [INSTALLATION.md](INSTALLATION.md) for detailed platform-specific instructions.**

**Optional features**:
```bash
# RLHF, multimodal, experiment tracking, dev tools
pip install -e ".[all]"

# Everything with GPU optimization
pip install -e ".[all-gpu]"
```

### Quick Example: Supervised Fine-Tuning

```python
from src.models.language import LanguageModel
from src.core.sft.trainer import SFTTrainer
from src.data.loaders import load_dataset

# Load a small model (e.g., GPT-2)
model = LanguageModel.from_pretrained("gpt2", use_lora=True)

# Load conversation dataset
dataset = load_dataset("daily_dialog", split="train[:1000]")

# Train with SFT
trainer = SFTTrainer(model=model, dataset=dataset)
trainer.train()
```

### Minimal Example

See `examples/minimal_sft.py` for a complete working example in under 50 lines.

## Platform Compatibility

This repository works on **all major platforms**, with some considerations:

| Platform | Status | Best For | Notes |
|----------|--------|----------|-------|
| **Linux + NVIDIA GPU** | ✅ Fully Supported | Production training | All features, best performance |
| **Google Colab** | ✅ Fully Supported | Experiments, learning | Free GPU, recommended |
| **macOS (Apple Silicon)** | ⚠️ CPU Recommended | Development, testing | MPS has limitations (see below) |
| **Windows + NVIDIA GPU** | ✅ Fully Supported | Production training | Same as Linux |

### Apple Silicon (MPS) Limitations

**What is MPS?** Metal Performance Shaders - Apple's GPU framework for M1/M2/M3/M4 chips.

**Current Issues:**
- Some PyTorch operations not yet implemented on MPS (e.g., `log_sigmoid`)
- Affects reward modeling and some advanced techniques
- PyTorch team actively adding support: https://github.com/pytorch/pytorch/issues/77764

**Recommended Approach for macOS:**
```bash
# Use CPU for stable training (works perfectly)
python scripts/train/train_reward_model.py device=cpu

# Or use Google Colab for GPU acceleration (free)
# See notebooks/02_reward_modeling_colab.ipynb
```

**Performance:** CPU training on M1/M2 is 10-15x slower than NVIDIA GPU, but sufficient for:
- ✅ Small models (GPT-2, OPT-350m)
- ✅ Development and testing
- ✅ Learning and experimentation

For production training, use Google Colab (free) or cloud GPUs.

📖 **See [docs/PLATFORM_COMPATIBILITY.md](docs/PLATFORM_COMPATIBILITY.md) for detailed platform guide.**

## Repository Structure

```
llm-post-training/
├── src/
│   ├── core/              # Custom implementations (SFT, PPO, DPO, etc.)
│   ├── models/            # Model wrappers (text & multimodal)
│   ├── data/              # Data loading and preprocessing
│   ├── evaluation/        # Metrics and benchmarks
│   └── utils/             # Utilities and visualization
│
├── scripts/
│   ├── train/             # Training scripts for each technique
│   ├── evaluate/          # Evaluation scripts
│   └── data/              # Data preparation
│
├── notebooks/             # Educational Jupyter notebooks
├── examples/              # Minimal working examples
├── configs/               # Hydra configuration files
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Supported Techniques

### 1. Supervised Fine-Tuning (SFT)
Adapt pre-trained models to specific tasks using instruction-following data.
- Custom loss masking (only predict response tokens)
- Support for multiple conversation formats
- LoRA/QLoRA integration for efficiency

### 2. Reward Modeling
Train models to predict human preferences using pairwise comparisons.
- Bradley-Terry ranking loss
- Custom reward model architecture
- Evaluation metrics for preference accuracy

### 3. RLHF with PPO
Full reinforcement learning from human feedback pipeline.
- **Custom PPO implementation** for educational value
- Four-model setup: Actor, Critic, Reference, Reward Model
- Rollout and update phases with detailed logging
- KL divergence constraints

### 4. Direct Preference Optimization (DPO)
Simplified RLHF that directly optimizes from preferences.
- Single-stage training (no separate reward model)
- More stable than PPO
- Custom implementation with comparisons to TRL

### 5. Identity Preference Optimization (IPO)
Variant of DPO with different loss formulation.
- Squared loss instead of log-sigmoid
- More robust to overconfident predictions

## Multimodal Support

All techniques support both text-only and vision-language models:

**Text Models**: GPT-2, OPT, LLaMA, Mistral
**Vision-Language**: CLIP (contrastive), LLaVA (generative)

**Want to add more models?** The framework is designed to be extensible. See `docs/ADDING_NEW_MULTIMODAL_MODELS.md` for a complete guide on integrating models like:
- JinaClip
- SigLIP
- BLIP-2
- InstructBLIP
- And other CLIP-like or generative vision-language models

```python
# Text-only example
model = LanguageModel.from_pretrained("gpt2")

# Multimodal example
model = VisionLanguageModel.from_pretrained("openai/clip-vit-base-patch32")
```

## Custom Data Support

Train on your own image-caption datasets! The repository supports loading custom data in multiple formats:

**Supported Formats:**
- JSON (recommended)
- JSONL (for large datasets/streaming)
- CSV (easy to create/edit)

**Quick Example:**
```bash
# Train CLIP on your custom image-caption pairs
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data=custom_image_caption \
    data.train_file=/path/to/your/train.json \
    data.image_dir=/path/to/your/images \
    data.format=json

# Train DPO on your preference annotations
python scripts/train/train_multimodal_dpo.py \
    experiment=clip_dpo \
    data=custom_preferences \
    data.train_file=/path/to/your/preferences.json \
    data.image_dir=/path/to/your/images
```

**See full guide:** `docs/CUSTOM_DATA_GUIDE.md`
- Data format specifications
- Validation tools
- Example data preparation scripts
- Best practices for caption quality

## Educational Resources

### Cloud Training Templates

**Ready-to-use notebooks for cloud training:**

- **Google Colab:** `notebooks/colab_training_template.ipynb`
  - Complete setup for Colab Free, Pro, Pro+
  - CLIP, LLaVA, GPT-2, DPO examples
  - GPU optimization tips
  - Storage and checkpointing
  - See: `docs/google_colab_guide.md` and `docs/CLOUD_PLATFORMS_GUIDE.md`

- **Databricks:** `notebooks/databricks_training_template.py`
  - Cluster configuration guide
  - MLflow integration
  - Model registry
  - Distributed training
  - See: `docs/CLOUD_PLATFORMS_GUIDE.md`

**Quick start (Colab):**
```python
# Open in Colab: notebooks/colab_training_template.ipynb
# Run setup cells, then:
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.fp16=true
```

### Educational Notebooks

1. `00_setup_and_quickstart.ipynb` - Get started quickly
2. `01_understanding_sft.ipynb` - Deep dive into SFT
3. `02_reward_modeling.ipynb` - Learn reward modeling
4. `03_ppo_rlhf_deep_dive.ipynb` - Understand RLHF/PPO
5. `04_dpo_simplified_rlhf.ipynb` - DPO vs PPO comparison
6. `06_multimodal_training.ipynb` - Train vision-language models
7. `09_comparing_techniques.ipynb` - Side-by-side comparison

### Documentation
- Technique explanations: `docs/techniques/`
- User guides: `docs/guides/`
- API reference: `docs/api/`
- Architecture overview: `docs/architecture/`

## Configuration Management

Uses Hydra for hierarchical configuration:

```bash
# Train SFT on GPT-2 with conversation data
python scripts/train/train_sft.py \
    model=gpt2 \
    data=conversation \
    technique.learning_rate=5e-5

# Train DPO on preference data
python scripts/train/train_dpo.py \
    experiment=dpo_llama_preferences
```

Configuration files in `configs/`:
- `model/`: Model-specific configs (gpt2, llama, clip, etc.)
- `technique/`: Technique-specific hyperparameters
- `data/`: Dataset configurations
- `experiment/`: Full experiment presets

## Evaluation

Comprehensive evaluation suite:

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, metrics=["bleu", "rouge", "reward"])
results = evaluator.evaluate(test_dataset)
```

**Text Metrics**: BLEU, ROUGE, perplexity, diversity
**Preference Metrics**: Win rate, ranking accuracy
**Multimodal Metrics**: CLIP score, image-text alignment

## Training Scripts

Each technique has a dedicated training script:

```bash
# Supervised Fine-Tuning
python scripts/train/train_sft.py

# Reward Model
python scripts/train/train_reward_model.py

# PPO/RLHF
python scripts/train/train_ppo.py

# DPO
python scripts/train/train_dpo.py

# IPO
python scripts/train/train_ipo.py
```

## Framework Choices

### Why These Frameworks?

- **transformers**: Industry standard, extensive model support, great documentation
- **peft**: Essential for training on consumer hardware (reduces memory by 90%)
- **accelerate**: Simplifies distributed training and mixed precision
- **trl**: Reference implementations to compare against
- **hydra**: Clean configuration management for experiments
- **wandb**: Best-in-class experiment tracking

### Custom vs Framework Code

**We implement from scratch**:
- Core training loops (PPO, DPO, IPO) for educational value
- Loss functions with detailed explanations
- Advantage estimation (GAE)
- Custom logging and metrics

**We leverage frameworks**:
- Model loading and tokenization (transformers)
- LoRA/QLoRA adapters (peft)
- Distributed training (accelerate)
- Dataset loading (datasets)

This hybrid approach maximizes learning while maintaining practicality.

## Hardware Requirements

Designed for **consumer hardware**:
- Small models (< 1B params) run on 8-16GB GPU
- CPU training supported for very small models
- LoRA reduces memory requirements by ~90%
- Gradient checkpointing for larger models

**Recommended**:
- GPU: NVIDIA RTX 3090/4090 (24GB) or A100
- RAM: 32GB+ system memory
- Storage: 50GB+ for models and datasets

**Minimum**:
- GPU: NVIDIA GTX 1080 Ti (11GB) or similar
- RAM: 16GB system memory
- Storage: 20GB

## Contributing

Contributions welcome! See `CONTRIBUTING.md` for guidelines.

Areas for contribution:
- Additional post-training techniques (RLAIF, Constitutional AI)
- More evaluation metrics
- Additional model architectures
- Documentation improvements
- Bug fixes and optimizations

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{llm_post_training,
  title = {LLM Post-Training Experimentation Repository},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-post-training}
}
```

## License

MIT License - see `LICENSE` file for details.

## Acknowledgments

Built on top of excellent open-source projects:
- HuggingFace Transformers, PEFT, TRL, Datasets
- PyTorch
- Research papers on RLHF, DPO, PPO

## Resources

**Papers**:
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)

**Related Projects**:
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

## Contact

Questions? Issues? Open an issue on GitHub or reach out at your.email@example.com
