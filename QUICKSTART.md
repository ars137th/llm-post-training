# Quick Start Guide

Get up and running with LLM post-training in 5 minutes.

## Installation (1 minute)

```bash
cd llm-post-training
pip install -r requirements/base.txt
pip install -e .
```

## Run Minimal Example (2 minutes)

```bash
python examples/minimal_sft.py
```

This will:
- Load GPT-2 with LoRA adapters
- Create a tiny dataset
- Show how to process data for SFT
- Demonstrate text generation

## What's Implemented (Phase 1 - Foundation)

✅ **Core Infrastructure**:
- `src/models/language.py` - Unified interface for text models (GPT-2, LLaMA, Mistral)
  - LoRA/QLoRA support
  - Generation capabilities
  - Log probability computation for DPO/PPO
- `src/models/base.py` - Protocol defining model interface
- `src/data/loaders.py` - Dataset loading from HuggingFace/local files
- `src/data/processors/text.py` - Text tokenization and preprocessing

✅ **Configuration System**:
- Hydra-based hierarchical configs
- Model configs: `configs/model/` (gpt2, opt-350m)
- Technique configs: `configs/technique/` (sft, dpo)
- Data configs: `configs/data/` (conversation, preference)

✅ **Documentation**:
- `README.md` - Project overview
- `SETUP.md` - Detailed installation guide
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT license

✅ **Examples**:
- `examples/minimal_sft.py` - Working SFT example

## What's Next (Phase 2-7)

### Phase 2: Supervised Fine-Tuning
- Implement `src/core/sft/trainer.py`
- Create `scripts/train/train_sft.py`
- Add evaluation metrics
- Tutorial notebook

### Phase 3: Reward Modeling
- Implement `src/core/reward_modeling/trainer.py`
- Preference data processing
- Ranking evaluation

### Phase 4: DPO
- Implement `src/core/dpo/trainer.py`
- Training script
- Comparison experiments

### Phase 5: PPO/RLHF
- Implement full `src/core/ppo/` module
- Rollout and update phases
- Integration with reward model

### Phase 6: Multimodal
- Implement `src/models/vision_language.py`
- CLIP and LLaVA support
- Multimodal processors

### Phase 7: Polish
- Complete notebooks
- Full documentation
- Additional examples

## Directory Structure

```
llm-post-training/
├── src/                    # Source code
│   ├── core/              # Training implementations (SFT, PPO, DPO)
│   ├── models/            # Model wrappers (✅ language.py done)
│   ├── data/              # Data loading (✅ loaders.py, text.py done)
│   ├── evaluation/        # Metrics and benchmarks
│   └── utils/             # Utilities
│
├── configs/               # Hydra configurations (✅ base configs done)
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Educational Jupyter notebooks
├── examples/              # Minimal examples (✅ minimal_sft.py done)
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Quick Commands

```bash
# Install dependencies
pip install -r requirements/base.txt

# Run minimal example
python examples/minimal_sft.py

# Run tests (when implemented)
pytest tests/

# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/
```

## Key Files Created (Phase 1)

### Core Components
1. **src/models/language.py** (450 lines)
   - Load models from HuggingFace
   - LoRA/QLoRA support
   - Generation and log probability computation

2. **src/data/loaders.py** (250 lines)
   - Load datasets from various sources
   - Split datasets
   - Apply preprocessing

3. **src/data/processors/text.py** (280 lines)
   - Tokenization utilities
   - SFT data processing
   - Prompt templates

### Configuration
4. **configs/config.yaml** - Base configuration
5. **configs/model/gpt2.yaml** - GPT-2 model config
6. **configs/technique/sft.yaml** - SFT technique config
7. **configs/data/conversation.yaml** - Conversation data config

### Documentation
8. **README.md** - Project overview and features
9. **SETUP.md** - Installation instructions
10. **CONTRIBUTING.md** - Contribution guidelines

## Usage Example

```python
from src.models.language import LanguageModel
from src.data.processors.text import TextProcessor

# Load model with LoRA
model = LanguageModel.from_pretrained("gpt2", use_lora=True)

# Create processor
processor = TextProcessor(model.tokenizer, max_length=512)

# Process data for SFT
data = processor.process_for_sft(
    prompt="What is AI?",
    response="AI is artificial intelligence...",
    mask_prompt=True  # Only compute loss on response
)

# Generate text
encoded = processor.tokenize("Hello, world!")
output = model.generate(encoded["input_ids"], max_new_tokens=50)
text = processor.decode(output[0])
```

## Framework Choices Explained

- **transformers**: Model loading, tokenization
  - *Why*: Industry standard, extensive model support
- **peft**: LoRA/QLoRA adapters
  - *Why*: 90% memory reduction, essential for consumer hardware
- **torch**: Deep learning framework
  - *Why*: Flexibility for custom implementations
- **hydra**: Configuration management
  - *Why*: Clean, hierarchical configs for experiments
- **datasets**: Dataset loading and processing
  - *Why*: Memory-mapped, efficient, HuggingFace ecosystem

## Hardware Requirements

**Minimum**:
- CPU only (slow but works)
- 8GB RAM
- GPT-2, small OPT models

**Recommended**:
- NVIDIA GPU with 8-16GB VRAM
- 16GB+ system RAM
- Enables GPT-2, OPT-350M, small LLaMA with LoRA

## Next Steps

1. **Explore the code**:
   - Read `src/models/language.py` to understand model interface
   - Check `src/data/loaders.py` for data loading
   - Review configs in `configs/`

2. **Run examples**:
   - `python examples/minimal_sft.py`

3. **Contribute**:
   - See `CONTRIBUTING.md` for guidelines
   - Pick a task from Phase 2-7
   - Open a PR!

4. **Learn**:
   - Read the documentation in `docs/`
   - Explore Jupyter notebooks (coming in Phase 2+)

## Getting Help

- **Installation issues**: See `SETUP.md` troubleshooting
- **Usage questions**: Check `README.md` and docs
- **Bugs**: Open an issue on GitHub
- **Feature requests**: Open an issue with "feature" label

## What Makes This Repository Special

1. **Educational**: Custom implementations to understand internals
2. **Practical**: Uses proven frameworks for infrastructure
3. **Flexible**: Works with small models on consumer hardware
4. **Multimodal**: Support for both text and vision-language models
5. **Well-documented**: Extensive docs, examples, and notebooks
6. **Modular**: Easy to extend with new techniques

---

**Status**: Phase 1 (Foundation) Complete ✅

**Next**: Phase 2 (Supervised Fine-Tuning Implementation)

Start implementing by opening `src/core/sft/trainer.py` and following the plan!
