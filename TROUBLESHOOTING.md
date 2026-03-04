# Troubleshooting Guide

Common issues and solutions for the LLM Post-Training repository.

## Generation Issues

### Issue: "attention mask is not set" warning + Bus Error

**Symptoms:**
```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
zsh: bus error
```

**Cause:**
- Input tensors are on CPU but model is on GPU (or vice versa)
- Missing attention_mask in generation
- GPT-2 uses eos_token as pad_token, causing ambiguity

**Solution:**

```python
# ❌ WRONG - Missing device handling
encoded = processor.tokenize(prompt, return_tensors="pt")
output = model.generate(encoded["input_ids"])

# ✅ CORRECT - With device handling and attention_mask
encoded = processor.tokenize(prompt, return_tensors="pt")
encoded = {k: v.to(model.device) for k, v in encoded.items()}

output = model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    pad_token_id=model.tokenizer.pad_token_id,
)
```

**Key Points:**
1. Always move tensors to model device: `{k: v.to(model.device) for k, v in encoded.items()}`
2. Always pass `attention_mask` to generate
3. Explicitly set `pad_token_id` in generation

---

## CUDA / GPU Issues

### Issue: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python scripts/train/train_sft.py training.per_device_train_batch_size=1
   ```

2. **Use gradient accumulation:**
   ```bash
   python scripts/train/train_sft.py \
       training.per_device_train_batch_size=1 \
       training.gradient_accumulation_steps=8
   ```

3. **Enable 4-bit quantization:**
   ```bash
   python scripts/train/train_sft.py model.use_4bit=true
   ```

4. **Reduce sequence length:**
   ```bash
   python scripts/train/train_sft.py tokenizer.max_length=256
   ```

5. **Use gradient checkpointing:**
   ```python
   model.model.gradient_checkpointing_enable()
   ```

### Issue: Model loads to CPU instead of GPU

**Symptoms:**
```
Model device: cpu
```

**Check:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
```

**Solutions:**
- Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/
- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA version matches PyTorch

---

## Installation Issues

### Issue: bitsandbytes fails to install

**Symptoms:**
```
ERROR: Could not build wheels for bitsandbytes
```

**Solutions:**

**On macOS:**
- bitsandbytes requires CUDA, not supported on Mac
- Remove from requirements or use Linux/Windows with NVIDIA GPU

**On Windows:**
- May need to build from source
- Alternative: Use WSL2 with Linux

**On Linux:**
- Ensure CUDA toolkit installed: `nvcc --version`
- Install CUDA 11.8 or 12.1: https://developer.nvidia.com/cuda-downloads

### Issue: flash-attn compilation fails

**Symptoms:**
```
ERROR: Failed building wheel for flash-attn
```

**Solutions:**
- Compilation takes 10-15 minutes (be patient)
- Requires NVIDIA GPU with compute capability 7.5+
- Install with: `pip install flash-attn --no-build-isolation`
- If fails, skip flash-attn (it's optional)

### Issue: transformers import error

**Symptoms:**
```
ImportError: cannot import name 'X' from 'transformers'
```

**Solution:**
```bash
pip install --upgrade transformers
```

---

## Training Issues

### Issue: Loss is NaN

**Symptoms:**
```
train/loss: nan
```

**Causes & Solutions:**

1. **Learning rate too high:**
   ```bash
   python scripts/train/train_sft.py training.learning_rate=1e-5
   ```

2. **Mixed precision issues:**
   ```bash
   # Try disabling mixed precision
   python scripts/train/train_sft.py training.fp16=false
   ```

3. **Gradient clipping:**
   ```bash
   python scripts/train/train_sft.py training.max_grad_norm=0.5
   ```

### Issue: Training very slow

**Solutions:**

1. **Enable mixed precision:**
   ```bash
   python scripts/train/train_sft.py training.fp16=true
   ```

2. **Increase batch size:**
   ```bash
   python scripts/train/train_sft.py training.per_device_train_batch_size=8
   ```

3. **Use smaller max_length:**
   ```bash
   python scripts/train/train_sft.py tokenizer.max_length=256
   ```

4. **Check dataloader workers:**
   ```bash
   python scripts/train/train_sft.py num_workers=4
   ```

### Issue: Model not improving

**Symptoms:**
- Loss not decreasing
- Accuracy not increasing
- Poor generation quality

**Solutions:**

1. **Check data quality:**
   - Inspect processed examples
   - Verify prompt masking is correct
   - Ensure labels aren't all -100

2. **Try different learning rate:**
   ```bash
   # Try higher
   python scripts/train/train_sft.py training.learning_rate=1e-4

   # Or lower
   python scripts/train/train_sft.py training.learning_rate=1e-6
   ```

3. **Train longer:**
   ```bash
   python scripts/train/train_sft.py training.num_epochs=10
   ```

4. **Increase LoRA rank:**
   ```bash
   python scripts/train/train_sft.py model.lora_config.r=16
   ```

5. **Check if model is frozen:**
   ```python
   # Verify trainable parameters > 0
   print(f"Trainable: {model.num_trainable_parameters:,}")
   ```

---

## Data Issues

### Issue: Dataset not found

**Symptoms:**
```
ValueError: Failed to load dataset 'dataset_name'
```

**Solutions:**

1. **Check dataset name:**
   ```python
   from datasets import load_dataset
   dataset = load_dataset("Anthropic/hh-rlhf")  # Correct name
   ```

2. **Use local files:**
   ```bash
   python scripts/train/train_sft.py \
       data.dataset_name=json \
       data.data_files=path/to/data.json
   ```

3. **Check HuggingFace authentication:**
   ```bash
   huggingface-cli login
   ```

### Issue: KeyError in dataset processing

**Symptoms:**
```
KeyError: 'prompt'
```

**Cause:** Dataset has different column names than expected

**Solution:** Modify `train_sft.py` to adapt to your dataset format:

```python
def process_example(example):
    # Adapt to your dataset structure
    if 'question' in example and 'answer' in example:
        prompt = example['question']
        response = example['answer']
    # Add more formats as needed
    return {'prompt': prompt, 'response': response}
```

---

## Model Loading Issues

### Issue: Model not found on HuggingFace

**Symptoms:**
```
OSError: model 'X' is not a valid model identifier
```

**Solutions:**

1. **Check model name:**
   ```python
   # Correct names
   "gpt2"                    # ✓
   "facebook/opt-350m"       # ✓
   "meta-llama/Llama-2-7b-hf"  # ✓

   # Incorrect
   "gpt-2"                   # ✗ (should be "gpt2")
   "opt-350m"                # ✗ (needs "facebook/" prefix)
   ```

2. **Check if model requires authentication:**
   ```bash
   huggingface-cli login
   # Then accept license on HuggingFace website
   ```

3. **Use local model:**
   ```python
   model = LanguageModel.from_pretrained(
       "/path/to/local/model",
       use_lora=True,
   )
   ```

### Issue: PEFT/LoRA not working

**Symptoms:**
```
AttributeError: 'NoneType' object has no attribute 'lora_alpha'
```

**Solutions:**

1. **Check LoRA config:**
   ```python
   lora_config = {
       "r": 8,
       "lora_alpha": 16,
       "target_modules": ["q_proj", "v_proj"],
       "lora_dropout": 0.05,
       "bias": "none",
   }
   ```

2. **Verify PEFT installed:**
   ```bash
   pip install peft>=0.7.0
   ```

---

## Configuration Issues

### Issue: Hydra config override not working

**Symptoms:**
Config value not changing despite command line override

**Solutions:**

1. **Check syntax:**
   ```bash
   # Correct
   python scripts/train/train_sft.py training.learning_rate=1e-4

   # Incorrect
   python scripts/train/train_sft.py training.learning_rate 1e-4  # Missing =
   ```

2. **Check nested configs:**
   ```bash
   # For nested values, use dots
   python scripts/train/train_sft.py model.lora_config.r=16
   ```

3. **View resolved config:**
   The training script prints the full config at start

---

## Performance Tips

### Running on CPU

If you don't have a GPU:

```python
# Use smallest models
model = LanguageModel.from_pretrained("gpt2")  # 124M params

# Reduce batch size
training.per_device_train_batch_size=1

# Reduce sequence length
tokenizer.max_length=128

# Use fewer training samples
data.max_train_samples=100
```

### Running on Google Colab

```python
# Check GPU
!nvidia-smi

# Clone repo
!git clone https://github.com/yourusername/llm-post-training
%cd llm-post-training

# Install
!pip install -r requirements/base.txt

# Train
!python scripts/train/train_sft.py
```

### Running on AWS/Cloud

```bash
# Use larger instance
# p3.2xlarge (V100), g5.xlarge (A10G), etc.

# Enable mixed precision
training.fp16=true

# Use wandb for logging
logging.use_wandb=true
logging.wandb_project=my-project
```

---

## Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check device placement:

```python
print(f"Model device: {model.device}")
print(f"Input device: {input_ids.device}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

Check tokenizer:

```python
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
```

Check data:

```python
# Inspect processed example
example = train_dataset[0]
print(f"Input IDs: {example['input_ids'][:20]}")
print(f"Labels: {example['labels'][:20]}")
print(f"Attention mask: {example['attention_mask'][:20]}")

# Count masked tokens
masked = (example['labels'] == -100).sum()
total = len(example['labels'])
print(f"Masked: {masked}/{total} ({100*masked/total:.1f}%)")
```

---

## Getting Help

If you're still stuck:

1. **Check logs:** Look at `outputs/logs/` for detailed error messages
2. **Search issues:** https://github.com/yourusername/llm-post-training/issues
3. **Open an issue:** Provide:
   - Full error message
   - Your environment (OS, Python version, GPU)
   - Minimal code to reproduce
   - Config file used
4. **Community:** Ask in discussions

---

## Quick Diagnostic Checklist

When encountering issues, check:

- [ ] PyTorch installed correctly: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA available (if using GPU): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Transformers version: `pip show transformers`
- [ ] Model loads: `python -c "from src.models.language import LanguageModel; LanguageModel.from_pretrained('gpt2')"`
- [ ] Dataset accessible: Check HuggingFace Hub or local file path
- [ ] Config valid: Check for typos in YAML files
- [ ] Enough disk space: Check `df -h`
- [ ] Enough memory: Check `free -h` (Linux) or Activity Monitor (Mac)

---

## Useful Commands

```bash
# Check GPU usage
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check Python environment
which python
python --version
pip list | grep torch

# Check disk space
df -h

# Check memory
free -h  # Linux
top      # All systems

# Find CUDA version
nvcc --version

# Test imports
python -c "import torch, transformers, peft; print('All imports OK')"
```
