"""
Debug Configuration Loading

This script helps debug what config values are actually being loaded.
Run this to verify your experiment config is being applied correctly.

Usage:
    python scripts/debug_config.py experiment=reward_gpt2_hh_rlhf
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
CONFIGS_PATH = str(project_root / "configs")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

def main():
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_PATH, job_name="debug"):
        cfg = compose(config_name="config_reward", overrides=sys.argv[1:])

        print("=" * 80)
        print("Configuration Debug Info")
        print("=" * 80)

        print("\n[DATA CONFIGURATION]")
        print(f"  use_synthetic: {cfg.data.use_synthetic}")
        print(f"  dataset_name: {cfg.data.get('dataset_name', 'NOT SET')}")
        print(f"  format: {cfg.data.get('format', 'NOT SET')}")
        print(f"  num_train_examples: {cfg.data.num_train_examples}")
        print(f"  num_eval_examples: {cfg.data.num_eval_examples}")

        print("\n[TRAINING CONFIGURATION]")
        print(f"  output_dir: {cfg.training.output_dir}")
        print(f"  learning_rate: {cfg.training.learning_rate}")
        print(f"  per_device_train_batch_size: {cfg.training.per_device_train_batch_size}")

        print("\n[MODEL CONFIGURATION]")
        print(f"  name: {cfg.model.name}")
        print(f"  use_lora: {cfg.model.use_lora}")

        print("\n[FULL CONFIG (YAML)]")
        print("-" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("-" * 80)

        # Check if we're using real or synthetic data
        print("\n[EXPECTED BEHAVIOR]")
        if cfg.data.use_synthetic:
            print("  ⚠️  Will use SYNTHETIC data (for testing)")
        else:
            print(f"  ✅ Will use REAL data from: {cfg.data.dataset_name}")
            print(f"  📊 Format parser: {cfg.data.format}")

        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
