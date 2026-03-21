# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Post-Training on Databricks
# MAGIC
# MAGIC This notebook provides a complete template for training language models using various post-training techniques on Databricks.
# MAGIC
# MAGIC **What's Included:**
# MAGIC - SFT (Supervised Fine-Tuning) for text and multimodal models
# MAGIC - DPO (Direct Preference Optimization) for preference learning
# MAGIC - PPO/RLHF for reinforcement learning from human feedback
# MAGIC - MLflow integration for experiment tracking
# MAGIC
# MAGIC **Cluster Requirements:**
# MAGIC - Runtime: ML Runtime 14.0+ (includes PyTorch, transformers)
# MAGIC - Worker type: GPU instances (g4dn.xlarge or better)
# MAGIC - Driver type: Same as worker
# MAGIC - Workers: 1-4 (depending on model size)
# MAGIC
# MAGIC **Storage:**
# MAGIC - Models saved to DBFS: `/dbfs/mnt/llm_outputs/`
# MAGIC - Experiment tracking: MLflow (automatic)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 📚 **Documentation:** See `docs/databricks_guide.md` in the repository

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC
# MAGIC 1. [Cluster Configuration](#cluster-config)
# MAGIC 2. [Setup](#setup)
# MAGIC 3. [GPU Check](#gpu-check)
# MAGIC 4. [Storage Setup](#storage)
# MAGIC 5. [Training Examples](#training)
# MAGIC    - [CLIP Image-Text Training](#clip)
# MAGIC    - [LLaVA Vision-Language Training](#llava)
# MAGIC    - [GPT-2 Text Training](#gpt2)
# MAGIC    - [DPO Preference Learning](#dpo)
# MAGIC 6. [MLflow Tracking](#mlflow)
# MAGIC 7. [Evaluation](#evaluation)
# MAGIC 8. [Model Registry](#registry)
# MAGIC 9. [Troubleshooting](#troubleshooting)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cluster Configuration
# MAGIC
# MAGIC **Recommended Cluster Settings:**
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "runtime_version": "14.3.x-gpu-ml-scala2.12",
# MAGIC   "node_type_id": "g4dn.xlarge",
# MAGIC   "driver_node_type_id": "g4dn.xlarge",
# MAGIC   "num_workers": 1,
# MAGIC   "spark_conf": {
# MAGIC     "spark.databricks.delta.preview.enabled": "true"
# MAGIC   },
# MAGIC   "custom_tags": {
# MAGIC     "project": "llm-post-training"
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **For Larger Models (LLaVA-7B, LLaMA-7B):**
# MAGIC - Use `g4dn.2xlarge` (1 GPU, 32GB RAM) or `g5.2xlarge` (1 A10G GPU)
# MAGIC - Increase worker count to 2-4 for distributed training
# MAGIC
# MAGIC **Cost Optimization:**
# MAGIC - Use spot instances for dev/test
# MAGIC - Enable auto-termination (120 minutes)
# MAGIC - Use smaller clusters for CLIP/GPT-2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup
# MAGIC
# MAGIC Clone repository and install dependencies.

# COMMAND ----------

# Clone repository
%sh
cd /dbfs/tmp
rm -rf llm-post-training  # Clean up if exists
git clone https://github.com/your-username/llm-post-training.git
echo "✓ Repository cloned to /dbfs/tmp/llm-post-training"

# COMMAND ----------

# Install dependencies
%pip install -q -r /dbfs/tmp/llm-post-training/requirements/base.txt
%pip install -q -r /dbfs/tmp/llm-post-training/requirements/multimodal.txt

# Restart Python to use new packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. GPU Check
# MAGIC
# MAGIC Verify GPU is available on the cluster.

# COMMAND ----------

# Check GPU with nvidia-smi
%sh
nvidia-smi

# COMMAND ----------

# Check PyTorch GPU access
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: No GPU detected!")
    print("Check cluster configuration: Edit cluster → GPU enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Storage Setup
# MAGIC
# MAGIC Configure DBFS storage for model outputs.

# COMMAND ----------

# Create output directories in DBFS
%sh
mkdir -p /dbfs/mnt/llm_outputs/clip
mkdir -p /dbfs/mnt/llm_outputs/llava
mkdir -p /dbfs/mnt/llm_outputs/gpt2
mkdir -p /dbfs/mnt/llm_outputs/dpo
echo "✓ DBFS directories created"

# COMMAND ----------

# Verify DBFS mount
%sh
ls -lh /dbfs/mnt/llm_outputs/

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training Examples
# MAGIC
# MAGIC Run training jobs with different models and techniques.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 CLIP Image-Text Training
# MAGIC
# MAGIC Train CLIP on image-caption pairs.
# MAGIC
# MAGIC **Memory:** ~3 GB VRAM
# MAGIC **Time:** ~2-3 minutes for 1000 samples on g4dn.xlarge
# MAGIC **LoRA:** Not needed (CLIP is small, and LoRA is broken for CLIP)

# COMMAND ----------

# Train CLIP on synthetic data
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=synthetic \
    data.max_train_samples=1000 \
    training.output_dir=/dbfs/mnt/llm_outputs/clip/synthetic \
    training.num_epochs=3 \
    training.per_device_train_batch_size=32 \
    training.fp16=true \
    training.logging_steps=10 \
    logging.use_wandb=false

# COMMAND ----------

# Train CLIP on COCO (real data)
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    data.dataset_name=coco \
    data.max_train_samples=5000 \
    training.output_dir=/dbfs/mnt/llm_outputs/clip/coco \
    training.num_epochs=3 \
    training.per_device_train_batch_size=32 \
    training.fp16=true

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 LLaVA Vision-Language Training
# MAGIC
# MAGIC Train LLaVA (7B) with LoRA and 4-bit quantization.
# MAGIC
# MAGIC **Memory:** ~6 GB VRAM (with 4-bit + LoRA)
# MAGIC **Time:** ~30-40 minutes for 1000 samples on g4dn.xlarge
# MAGIC **Requirements:** g4dn.xlarge minimum (16GB GPU memory)

# COMMAND ----------

# Train LLaVA-7B with LoRA
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_multimodal.py \
    experiment=llava_instruction \
    model.use_lora=true \
    model.use_4bit=true \
    data.max_train_samples=1000 \
    training.output_dir=/dbfs/mnt/llm_outputs/llava/instruction \
    training.num_epochs=1 \
    training.per_device_train_batch_size=4 \
    training.gradient_accumulation_steps=4 \
    training.fp16=true \
    training.logging_steps=10 \
    training.save_steps=100

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 GPT-2 Text Training
# MAGIC
# MAGIC Train GPT-2 on conversational data.
# MAGIC
# MAGIC **Memory:** ~2 GB VRAM (with LoRA)
# MAGIC **Time:** ~5-10 minutes for 5000 samples on g4dn.xlarge

# COMMAND ----------

# Train GPT-2 with LoRA
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_sft.py \
    experiment=gpt2_conversation \
    model.use_lora=true \
    data.max_train_samples=5000 \
    training.output_dir=/dbfs/mnt/llm_outputs/gpt2/conversation \
    training.num_epochs=3 \
    training.per_device_train_batch_size=16 \
    training.fp16=true

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 DPO Preference Learning
# MAGIC
# MAGIC Train using Direct Preference Optimization on preference pairs.
# MAGIC
# MAGIC **Memory:** ~3 GB VRAM (with LoRA)
# MAGIC **Time:** ~10-15 minutes for 2000 pairs on g4dn.xlarge

# COMMAND ----------

# Train GPT-2 with DPO
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_dpo.py \
    experiment=gpt2_dpo \
    model.use_lora=true \
    data.dataset_name=anthropic_hh \
    data.max_train_samples=2000 \
    training.output_dir=/dbfs/mnt/llm_outputs/dpo/gpt2 \
    training.num_epochs=1 \
    training.per_device_train_batch_size=8 \
    training.beta=0.1 \
    training.fp16=true

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow Tracking
# MAGIC
# MAGIC Track experiments with MLflow (automatic on Databricks).

# COMMAND ----------

import mlflow

# Get current experiment
experiment = mlflow.get_experiment_by_name("/Users/your-email@company.com/llm-post-training")
if experiment:
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")

    # List recent runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
    print("\nRecent runs:")
    print(runs[['run_id', 'start_time', 'status', 'metrics.train/loss']])

# COMMAND ----------

# View specific run details
run_id = "your-run-id-here"  # Get from above

with mlflow.start_run(run_id=run_id):
    # Log additional metrics
    mlflow.log_metric("custom_metric", 0.95)

    # Log artifacts
    mlflow.log_artifact("/dbfs/mnt/llm_outputs/clip/synthetic/config.yaml")

print(f"Run details: {mlflow.get_run(run_id).data}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluation
# MAGIC
# MAGIC Evaluate trained models on test data.

# COMMAND ----------

# Evaluate CLIP model
%sh
cd /dbfs/tmp/llm-post-training
python scripts/evaluate/evaluate_model.py \
    model_path=/dbfs/mnt/llm_outputs/clip/synthetic \
    model_type=clip \
    eval_dataset=coco \
    metrics=clip_score

# COMMAND ----------

# Evaluate text model
%sh
cd /dbfs/tmp/llm-post-training
python scripts/evaluate/evaluate_model.py \
    model_path=/dbfs/mnt/llm_outputs/gpt2/conversation \
    model_type=gpt2 \
    eval_dataset=conversation_test \
    metrics=perplexity,bleu,rouge

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Registry
# MAGIC
# MAGIC Register trained models in Databricks Model Registry.

# COMMAND ----------

import mlflow
import mlflow.pytorch

# Register model
model_path = "/dbfs/mnt/llm_outputs/clip/synthetic"
model_name = "clip-image-caption-finetuned"

# Create or get registered model
try:
    model_version = mlflow.register_model(
        model_uri=f"file://{model_path}",
        name=model_name
    )
    print(f"✓ Model registered: {model_name} version {model_version.version}")
except Exception as e:
    print(f"Error registering model: {e}")

# COMMAND ----------

# Transition model to staging
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

print(f"✓ Model transitioned to Staging")

# COMMAND ----------

# Load model from registry
import mlflow.pyfunc

model_uri = f"models:/{model_name}/Staging"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(f"✓ Model loaded from registry: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Troubleshooting
# MAGIC
# MAGIC ### Out of Memory (OOM)
# MAGIC
# MAGIC **Solutions:**
# MAGIC 1. Reduce batch size in training command
# MAGIC 2. Enable 4-bit quantization: `model.use_4bit=true`
# MAGIC 3. Use larger instance type (g4dn.2xlarge or g5.2xlarge)
# MAGIC 4. Enable gradient checkpointing: `model.gradient_checkpointing=true`
# MAGIC
# MAGIC ### Training Job Killed
# MAGIC
# MAGIC **Check cluster logs:**
# MAGIC - Clusters → Select your cluster → Driver Logs
# MAGIC - Look for OOM errors or GPU errors
# MAGIC
# MAGIC **Prevention:**
# MAGIC - Save frequent checkpoints: `training.save_steps=100`
# MAGIC - Use auto-scaling clusters
# MAGIC
# MAGIC ### DBFS Access Issues
# MAGIC
# MAGIC **Verify mount:**

# COMMAND ----------

# Check DBFS mount status
%sh
ls -lh /dbfs/mnt/

# If mount doesn't exist, create it
%python
dbutils.fs.mkdirs("/mnt/llm_outputs")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Package Version Conflicts
# MAGIC
# MAGIC **Solution: Restart Python**

# COMMAND ----------

# Restart Python kernel
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPU Not Detected
# MAGIC
# MAGIC **Check cluster configuration:**
# MAGIC 1. Edit cluster → Instance type (must be GPU instance)
# MAGIC 2. Runtime version (must be GPU ML runtime)
# MAGIC 3. Restart cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup
# MAGIC
# MAGIC Free up DBFS space when done.

# COMMAND ----------

# Clear local cache
%sh
rm -rf /dbfs/tmp/llm-post-training/.cache
echo "✓ Cache cleared"

# COMMAND ----------

# List DBFS outputs (don't delete unless sure)
%sh
du -sh /dbfs/mnt/llm_outputs/*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Resources
# MAGIC
# MAGIC - **Databricks Guide:** `docs/databricks_guide.md`
# MAGIC - **Full Documentation:** `docs/google_colab_guide.md` (similar concepts)
# MAGIC - **Known Issues:** `docs/known_issues.md`
# MAGIC - **Configuration Guide:** `docs/CONFIGURATION_GUIDE.md`
# MAGIC
# MAGIC ## Support
# MAGIC
# MAGIC - Issues: https://github.com/your-username/llm-post-training/issues
# MAGIC - Discussions: https://github.com/your-username/llm-post-training/discussions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced: Distributed Training
# MAGIC
# MAGIC For multi-GPU or multi-node training on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-GPU Training (Single Node)
# MAGIC
# MAGIC If your cluster has multiple GPUs per node:

# COMMAND ----------

# Train with multiple GPUs (automatically uses all available)
%sh
cd /dbfs/tmp/llm-post-training
python scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.output_dir=/dbfs/mnt/llm_outputs/clip/distributed \
    training.per_device_train_batch_size=16 \
    training.fp16=true

# PyTorch DDP (DistributedDataParallel) is used automatically

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-Node Training (Advanced)
# MAGIC
# MAGIC For very large models requiring multiple nodes:

# COMMAND ----------

# Configure for multi-node training
# Set environment variables
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '2'  # Number of nodes

# Run distributed training
%sh
cd /dbfs/tmp/llm-post-training
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    scripts/train/train_multimodal.py \
    experiment=clip_image_caption \
    training.output_dir=/dbfs/mnt/llm_outputs/clip/multi_node

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cost Monitoring
# MAGIC
# MAGIC Track training costs on Databricks.

# COMMAND ----------

# Get cluster cost (approximate)
from datetime import datetime, timedelta

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(f"Cluster ID: {cluster_id}")

# Estimate: g4dn.xlarge ~ $0.50/hour on-demand
# For accurate costs, check Databricks account console

# Calculate estimated cost for this session
# (This is a placeholder - actual cost tracking requires Databricks API)
session_hours = 2  # Replace with actual session time
instance_cost = 0.50  # USD per hour for g4dn.xlarge
estimated_cost = session_hours * instance_cost * 1  # 1 worker

print(f"Estimated session cost: ${estimated_cost:.2f}")
print("Note: Check Databricks account console for accurate billing")
