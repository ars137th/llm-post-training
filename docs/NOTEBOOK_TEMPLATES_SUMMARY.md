# Cloud Training Notebook Templates - Summary

## What Was Created

Complete training templates for two major cloud platforms: Google Colab and Databricks.

---

## Files Created

### 1. Google Colab Notebook
**File:** `notebooks/colab_training_template.ipynb`

**Format:** Jupyter Notebook (.ipynb)

**Sections:**
- Header and overview
- Table of contents
- Setup and installation
- GPU verification
- Google Drive storage setup
- Training examples:
  - CLIP image-text training
  - LLaVA-7B vision-language training
  - GPT-2 text training
  - DPO preference learning
- Evaluation
- Model download
- Troubleshooting
- Cleanup

**Features:**
- Complete code cells ready to run
- Markdown documentation in each cell
- Memory and time estimates for each example
- GPU tier recommendations
- Storage strategies (Drive, HuggingFace Hub, local download)
- Session management tips
- Cost estimates

**Usage:**
```bash
# Open in Google Colab:
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Select notebooks/colab_training_template.ipynb
4. Runtime → Change runtime type → GPU
5. Run cells sequentially
```

---

### 2. Databricks Notebook
**File:** `notebooks/databricks_training_template.py`

**Format:** Python file with Databricks magic commands

**Sections:**
- Header and overview (MAGIC %md)
- Table of contents
- Cluster configuration guide
- Setup and installation (%sh, %pip)
- GPU verification
- DBFS storage setup
- Training examples (same as Colab)
- MLflow experiment tracking
- Evaluation
- Model Registry integration
- Distributed training (advanced)
- Cost monitoring
- Troubleshooting
- Cleanup

**Features:**
- Databricks-specific commands (%sh, %pip, %md)
- MLflow integration (automatic)
- DBFS storage patterns
- Model Registry workflow
- Multi-GPU and multi-node training examples
- Cluster configuration recommendations
- DBU cost estimation

**Usage:**
```bash
# Import to Databricks:
1. Login to Databricks workspace
2. Workspace → Import
3. Upload notebooks/databricks_training_template.py
4. Create GPU cluster (g4dn.xlarge or better)
5. Attach notebook to cluster
6. Run cells sequentially
```

---

### 3. Comprehensive Guide
**File:** `docs/CLOUD_PLATFORMS_GUIDE.md`

**Contents:**
- Platform comparison table (Colab vs Databricks)
- Detailed setup instructions for both platforms
- Training examples with commands
- Cost comparison and analysis
- Model size guide (which platform for which model)
- Storage management strategies
- Troubleshooting common issues
- Best practices
- Quick reference commands

**When to Use:**
- Reference guide while working in notebooks
- Comparing platforms before choosing
- Cost planning
- Troubleshooting

---

### 4. macOS Limitation Document
**File:** `MACOS_LIMITATION.md`

**Contents:**
- Explanation of why training fails on macOS
- Root cause analysis (fork safety)
- Solutions and alternatives
- What works vs doesn't work on macOS
- Cost-benefit analysis
- Recommendations

**When to Use:**
- Understanding macOS training errors
- Deciding between local and cloud training
- Planning development workflow

---

### 5. Code Review Document
**File:** `docs/CODE_REVIEW_MACOS_FIX.md`

**Contents:**
- Review of compat layer usage
- Hydra config ordering verification
- Before/after comparisons
- Testing procedures
- Related documentation links

**When to Use:**
- Code review reference
- Understanding the macOS fix implementation
- Learning compat layer patterns

---

## Platform Comparison

| Feature | Google Colab | Databricks |
|---------|-------------|-----------|
| **Best For** | Individual research, learning | Teams, production pipelines |
| **Free Tier** | ✅ Yes (T4 GPU) | ❌ No (trial only) |
| **Pricing** | $0 / $10 / $50 per month | ~$0.83/hour (g4dn.xlarge) |
| **GPU Options** | T4, V100, A100 | All AWS GPU instances |
| **Storage** | Google Drive | DBFS |
| **Experiment Tracking** | TensorBoard/WandB | MLflow (built-in) |
| **Collaboration** | Share links | Full workspace |
| **Deployment** | Manual | Model Registry + Serving |
| **Session Limit** | 12h (Free) / 24h (Pro) | Configurable |
| **Setup Time** | ~5 minutes | ~10 minutes |
| **Learning Curve** | Easy | Moderate |

---

## Training Time Estimates

**Google Colab (T4 GPU):**
- CLIP (1000 samples): 2-3 minutes
- LLaVA-7B (1000 samples): 30-40 minutes
- GPT-2 (5000 samples): 5-10 minutes
- DPO (2000 pairs): 10-15 minutes

**Databricks (g4dn.xlarge):**
- CLIP (1000 samples): 2-3 minutes
- LLaVA-7B (1000 samples): 30-40 minutes
- GPT-2 (5000 samples): 5-10 minutes
- DPO (2000 pairs): 10-15 minutes

**macOS (M1/M2 CPU):**
- CLIP (1000 samples): ❌ Bus error
- Any training: ❌ Fails (fork safety issues)

---

## Cost Estimates

### Google Colab

**Free Tier:** $0
- T4 GPU, 12-hour sessions
- Perfect for CLIP, GPT-2

**Pro Tier:** $10/month
- V100 GPU, 24-hour sessions
- Can handle LLaVA-7B

**Pro+ Tier:** $50/month
- A100 GPU, unlimited sessions
- Best for LLaMA-13B+

### Databricks

**g4dn.xlarge:** ~$0.83/hour
- EC2: $0.526/hour
- DBU: $0.30/hour

**Example costs:**
- CLIP training (3 min): ~$0.05
- LLaVA training (30 min): ~$0.50
- GPT-2 training (10 min): ~$0.15

**Cost optimization:**
- Use spot instances (50-70% savings)
- Auto-terminate clusters
- Smaller instances for small models

---

## Quick Start Guide

### For Beginners (Colab Recommended)

1. **Open Colab notebook:**
   ```
   File → Open notebook → Upload
   Select: notebooks/colab_training_template.ipynb
   ```

2. **Enable GPU:**
   ```
   Runtime → Change runtime type → GPU
   ```

3. **Run setup cells:**
   - Clone repo
   - Install dependencies
   - Mount Google Drive

4. **Choose training example:**
   - Start with CLIP (fastest)
   - Try GPT-2 next
   - Then LLaVA if on Pro tier

5. **Train model:**
   ```python
   !python scripts/train/train_multimodal.py \
       experiment=clip_image_caption \
       training.fp16=true
   ```

### For Teams/Production (Databricks)

1. **Create cluster:**
   - Runtime: ML 14.3+ (GPU)
   - Instance: g4dn.xlarge
   - Workers: 1-4

2. **Import notebook:**
   ```
   Workspace → Import
   Upload: notebooks/databricks_training_template.py
   ```

3. **Attach cluster:**
   - Select cluster from dropdown
   - Wait for startup

4. **Run cells:**
   - Setup and install
   - Choose training example
   - Monitor with MLflow

---

## Documentation Structure

```
docs/
├── CLOUD_PLATFORMS_GUIDE.md       # Main guide (both platforms)
├── google_colab_guide.md           # Detailed Colab guide
├── CODE_REVIEW_MACOS_FIX.md       # Code review of fixes
└── NOTEBOOK_TEMPLATES_SUMMARY.md  # This file

notebooks/
├── colab_training_template.ipynb   # Colab notebook
└── databricks_training_template.py # Databricks notebook

MACOS_LIMITATION.md                 # Why macOS training fails
```

---

## Usage Recommendations

### Development Workflow

1. **Local macOS:**
   - Write code
   - Unit tests
   - Data exploration
   - ❌ No training

2. **Google Colab:**
   - Experiments
   - Hyperparameter tuning
   - Learning
   - Small-scale production

3. **Databricks:**
   - Team collaboration
   - Large-scale training
   - Production pipelines
   - Model versioning

### Model Selection by Platform

**Colab Free (T4, 16GB):**
- ✅ CLIP-ViT-B/32
- ✅ GPT-2
- ✅ GPT-2 Medium
- ⚠️ LLaVA-7B (slow)

**Colab Pro (V100, 16GB):**
- ✅ All above
- ✅ LLaVA-7B (good)
- ✅ LLaMA-7B with 4-bit LoRA

**Databricks (g4dn.xlarge, 16GB):**
- ✅ All Colab Pro models
- ✅ Better for production
- ✅ Team features

**Databricks (g5.2xlarge, 24GB):**
- ✅ All above
- ✅ LLaMA-13B with 4-bit LoRA
- ✅ Larger batch sizes

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size
- Enable 4-bit quantization
- Use gradient accumulation

**"Session disconnected" (Colab)**
- Save to Google Drive
- Use checkpoints
- Upgrade to Pro

**"Cluster terminated" (Databricks)**
- Increase timeout
- Save to DBFS
- Resume from checkpoint

**"Training too slow"**
- Check GPU utilization
- Enable fp16
- Increase batch size

---

## Next Steps

1. **Try Colab first** (free, easy)
2. **Read CLOUD_PLATFORMS_GUIDE.md** for details
3. **Train a small model** (CLIP or GPT-2)
4. **Scale up** to LLaVA or DPO
5. **Consider Databricks** for teams

---

## Support

- **Documentation:** `docs/google_colab_guide.md`, `docs/CLOUD_PLATFORMS_GUIDE.md`
- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions

---

## Additional Notes

### Why Two Notebooks?

**Different formats:**
- Colab: Standard Jupyter (.ipynb)
- Databricks: Python script with magic commands

**Different features:**
- Colab: Simple, free tier, individual
- Databricks: MLflow, Model Registry, teams

### Why Not AWS SageMaker / Azure ML?

These templates focus on the **two most accessible platforms**:
- Colab: Free tier, no account setup complexity
- Databricks: Standard in many enterprises

SageMaker and Azure ML can be added later if there's demand.

### Can I Use These Locally?

The training commands work locally on **Linux with NVIDIA GPU**:

```bash
# Same commands work on Linux GPU
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.fp16=true
```

But notebooks are optimized for cloud platforms.
