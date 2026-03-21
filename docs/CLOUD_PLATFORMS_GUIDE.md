# Cloud Platforms Training Guide

Complete guide for training LLM models on Google Colab and Databricks.

## Quick Links

- **Google Colab Notebook:** `notebooks/colab_training_template.ipynb`
- **Databricks Notebook:** `notebooks/databricks_training_template.py`
- **Detailed Colab Guide:** `docs/google_colab_guide.md`

---

## Platform Comparison

| Feature | Google Colab | Databricks |
|---------|-------------|-----------|
| **Best For** | Individual research, prototyping | Team collaboration, production |
| **Free Tier** | ✅ Yes (T4 GPU, 12 hours) | ❌ No (trial only) |
| **Pricing** | Free / $10 / $50 per month | Pay-per-use (DBU-based) |
| **GPU Options** | T4, V100, A100 | G4dn, G5, P3, P4d instances |
| **Storage** | Google Drive | DBFS (Databricks File System) |
| **Experiment Tracking** | Manual (TensorBoard/WandB) | MLflow (built-in) |
| **Collaboration** | Share notebook links | Workspace collaboration |
| **Deployment** | Manual | Model Registry + Serving |
| **Session Limit** | 12 hours (Free), 24 hours (Pro) | Cluster lifetime (configurable) |

---

## Google Colab

### Setup (5 minutes)

1. **Open Notebook:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `notebooks/colab_training_template.ipynb`
   - Or: File → Open notebook → GitHub → Paste repo URL

2. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (Free), V100 (Pro), A100 (Pro+)

3. **Run Setup Cells:**
   ```python
   # Cell 1: Clone repo
   !git clone https://github.com/your-repo/llm-post-training.git
   %cd llm-post-training

   # Cell 2: Install dependencies
   !pip install -r requirements/base.txt -r requirements/multimodal.txt

   # Cell 3: Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### Training Examples

#### CLIP (2-3 minutes on T4)
```bash
!python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.output_dir=/content/drive/MyDrive/llm_outputs/clip \
    training.num_epochs=3 \
    training.fp16=true
```

#### LLaVA-7B (30-40 minutes on T4)
```bash
!python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    model.use_lora=true \
    model.use_4bit=true \
    training.output_dir=/content/drive/MyDrive/llm_outputs/llava
```

#### GPT-2 (5-10 minutes on T4)
```bash
!python scripts/train/train_sft.py \
    experiment=gpt2_conversation \
    model.use_lora=true \
    training.output_dir=/content/drive/MyDrive/llm_outputs/gpt2
```

### Colab Tips

**Prevent Disconnects:**
```javascript
// Run in browser console (F12)
function ClickConnect(){
    console.log("Keeping session alive");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

**Download Models:**
```python
from google.colab import files
!zip -r model.zip /content/drive/MyDrive/llm_outputs/clip
files.download('model.zip')
```

**Monitor GPU:**
```bash
!nvidia-smi -l 5  # Update every 5 seconds
```

---

## Databricks

### Setup (10 minutes)

1. **Create Cluster:**
   - Workspace → Compute → Create Cluster
   - Runtime: ML Runtime 14.3+ (GPU)
   - Worker type: `g4dn.xlarge` (1 GPU, 16GB)
   - Workers: 1 (for CLIP/GPT-2) or 2-4 (for LLaVA)
   - Auto-termination: 120 minutes

2. **Import Notebook:**
   - Workspace → Import
   - Upload `notebooks/databricks_training_template.py`
   - Or: Import from URL/GitHub

3. **Attach Cluster:**
   - Open notebook → Select cluster from dropdown
   - Wait for cluster to start

### Training Examples

#### CLIP (2-3 minutes on g4dn.xlarge)
```bash
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.output_dir=/dbfs/mnt/llm_outputs/clip \
    training.num_epochs=3 \
    training.fp16=true
```

#### LLaVA-7B (30-40 minutes on g4dn.xlarge)
```bash
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    model.use_lora=true \
    model.use_4bit=true \
    training.output_dir=/dbfs/mnt/llm_outputs/llava
```

### MLflow Tracking (Built-in)

```python
import mlflow

# View experiments
experiments = mlflow.list_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# Get run details
run = mlflow.get_run("run-id-here")
print(run.data.metrics)
print(run.data.params)
```

### Model Registry

```python
import mlflow

# Register model
model_uri = "dbfs:/mnt/llm_outputs/clip/synthetic"
mlflow.register_model(model_uri, "clip-finetuned")

# Load from registry
model = mlflow.pyfunc.load_model("models:/clip-finetuned/Production")
```

### Databricks Tips

**Check Cluster Logs:**
- Compute → Select cluster → Driver Logs/Event Log

**Monitor Costs:**
- Account Console → Usage → Detailed Usage

**Save Checkpoints to DBFS:**
```bash
training.output_dir=/dbfs/mnt/llm_outputs/model_name
```

---

## Cost Comparison

### Google Colab

| Tier | Price | GPU | VRAM | Session | Best For |
|------|-------|-----|------|---------|----------|
| **Free** | $0 | T4 | 16GB | 12h | CLIP, GPT-2 |
| **Pro** | $10/mo | V100 | 16GB | 24h | LLaVA-7B |
| **Pro+** | $50/mo | A100 | 40GB | ∞ | LLaMA-13B+ |

**Estimated training costs (Pro tier):**
- CLIP (1000 samples): ~$0.01 (3 minutes)
- LLaVA-7B (1000 samples): ~$0.10 (30 minutes)
- GPT-2 (5000 samples): ~$0.05 (10 minutes)

### Databricks

Pricing based on DBUs (Databricks Units) + EC2 instance costs.

**Example: g4dn.xlarge (1 GPU)**
- EC2 cost: ~$0.526/hour (on-demand)
- DBU cost: ~$0.30/hour (standard tier)
- Total: ~$0.83/hour

**Estimated training costs:**
- CLIP (1000 samples): ~$0.05 (3 minutes)
- LLaVA-7B (1000 samples): ~$0.50 (30 minutes)
- GPT-2 (5000 samples): ~$0.15 (10 minutes)

**Cost optimization:**
- Use spot instances (50-70% savings)
- Auto-terminate clusters
- Use smaller instances for small models

---

## Model Size Guide

Which platform for which model?

| Model | Parameters | Memory | Colab Free | Colab Pro | Databricks |
|-------|-----------|--------|-----------|-----------|-----------|
| **CLIP-ViT-B/32** | 151M | ~1.2 GB | ✅ Fast | ✅ Fastest | ✅ Fast |
| **GPT-2** | 124M | ~1 GB | ✅ Fast | ✅ Fastest | ✅ Fast |
| **GPT-2 Medium** | 355M | ~2.8 GB | ✅ OK | ✅ Fast | ✅ Fast |
| **LLaMA-1B** | 1.1B | ~4 GB (4-bit) | ✅ OK | ✅ Fast | ✅ Fast |
| **LLaVA-7B** | 7B | ~6 GB (4-bit LoRA) | ⚠️ Slow/OOM | ✅ Good | ✅ Best |
| **LLaMA-7B** | 7B | ~4 GB (4-bit LoRA) | ⚠️ Slow/OOM | ✅ Good | ✅ Best |
| **LLaMA-13B** | 13B | ~7 GB (4-bit LoRA) | ❌ OOM | ⚠️ Slow | ✅ Good |

**Legend:**
- ✅ Fast: Trains efficiently
- ✅ Good: Trains well but slower
- ⚠️ Slow: Works but may timeout
- ❌ OOM: Out of memory

---

## Storage Management

### Google Colab

**Google Drive Storage:**
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Save outputs
training.output_dir=/content/drive/MyDrive/llm_outputs/model_name
```

**HuggingFace Hub:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./outputs/model",
    repo_id="your-username/model-name"
)
```

**Download Locally:**
```python
from google.colab import files
!zip -r model.zip ./outputs/model
files.download('model.zip')
```

### Databricks

**DBFS Storage:**
```bash
# Save to DBFS
training.output_dir=/dbfs/mnt/llm_outputs/model_name

# Access from notebook
%sh ls -lh /dbfs/mnt/llm_outputs/
```

**Model Registry:**
```python
# Register in Databricks
mlflow.register_model(
    model_uri="file:///dbfs/mnt/llm_outputs/model",
    name="model-name"
)
```

**Export to S3:**
```python
# Copy DBFS to S3
dbutils.fs.cp(
    "dbfs:/mnt/llm_outputs/model",
    "s3://your-bucket/models/model-name",
    recurse=True
)
```

---

## Troubleshooting

### Common Issues on Both Platforms

#### Out of Memory
**Symptoms:** CUDA out of memory, killed process

**Solutions:**
1. Reduce batch size: `training.per_device_train_batch_size=4`
2. Enable 4-bit: `model.use_4bit=true`
3. Gradient accumulation: `training.gradient_accumulation_steps=4`
4. Gradient checkpointing: `model.gradient_checkpointing=true`

#### Training Too Slow
**Symptoms:** Low GPU utilization, long training times

**Solutions:**
1. Increase batch size (if memory allows)
2. Enable fp16: `training.fp16=true`
3. Use larger GPU tier
4. Reduce dataset size for testing

#### Session Disconnected (Colab)
**Prevention:**
- Save to Google Drive
- Frequent checkpoints: `training.save_steps=100`
- Upgrade to Pro

**Recovery:**
```bash
training.resume_from_checkpoint=/content/drive/MyDrive/llm_outputs/model/checkpoint-500
```

#### Cluster Terminated (Databricks)
**Prevention:**
- Increase auto-termination time
- Monitor cluster utilization
- Save to DBFS

**Recovery:**
- Restart cluster
- Resume from checkpoint in DBFS

---

## Best Practices

### Development Workflow

**1. Local Development (macOS/Linux):**
- Write code and tests
- Debug logic
- No training (use Colab/Databricks)

**2. Colab for Training:**
- Quick experiments
- Hyperparameter tuning
- Small-scale production

**3. Databricks for Production:**
- Team collaboration
- Large-scale training
- Model versioning and deployment

### Experiment Tracking

**Colab:**
```python
# Use WandB or TensorBoard
training.logging_steps=10
logging.use_tensorboard=true
# Or
logging.use_wandb=true
logging.wandb_project=my-project
```

**Databricks:**
```python
# MLflow automatic
# View in Databricks UI: Experiments tab
```

### Checkpoint Strategy

**Frequent saves during training:**
```yaml
training:
  save_steps: 100  # Save every 100 steps
  save_total_limit: 3  # Keep last 3 checkpoints
```

**Resume from checkpoint:**
```bash
training.resume_from_checkpoint=path/to/checkpoint-500
```

---

## Quick Reference

### Colab Commands

```python
# Check GPU
!nvidia-smi

# Install packages
!pip install package-name

# Run training
!python scripts/train/train_sft.py experiment=exp_name

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Download file
from google.colab import files
files.download('file.zip')
```

### Databricks Commands

```python
# Check GPU
%sh nvidia-smi

# Install packages
%pip install package-name

# Run training
%sh python scripts/train/train_sft.py experiment=exp_name

# DBFS operations
dbutils.fs.ls("/mnt/llm_outputs")
dbutils.fs.cp("src", "dest")

# MLflow
import mlflow
mlflow.search_runs()
```

---

## Additional Resources

- **Colab Detailed Guide:** `docs/google_colab_guide.md`
- **Model Selection:** `docs/model_selection_guide.md`
- **Configuration:** `docs/CONFIGURATION_GUIDE.md`
- **Known Issues:** `docs/known_issues.md`
- **Platform Compatibility:** `docs/PLATFORM_COMPATIBILITY.md`

## Support

- **Issues:** https://github.com/your-username/llm-post-training/issues
- **Discussions:** https://github.com/your-username/llm-post-training/discussions
