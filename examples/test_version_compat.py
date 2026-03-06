"""
Test script to verify version compatibility utilities.

This script checks that the version detection and compatibility
helpers work correctly across different library versions.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.compat import (
    get_version_info,
    print_version_info,
    get_training_args_kwargs,
    should_pass_tokenizer_to_trainer,
    training_step_accepts_num_items,
    TRANSFORMERS_VERSION,
    TRANSFORMERS_4_36,
)


def main():
    print("=" * 70)
    print("COMPATIBILITY UTILITIES TEST")
    print("=" * 70)

    # Print version information
    print_version_info()

    # Test training args kwargs
    print("\n" + "=" * 70)
    print("Testing TrainingArguments compatibility...")
    print("=" * 70)

    training_kwargs = get_training_args_kwargs(
        output_dir="./test_output",
        eval_enabled=True,
        logging_dir="./test_logs",
        learning_rate=5e-5,
        num_train_epochs=3,
    )

    print("\nGenerated TrainingArguments kwargs:")
    for key, value in training_kwargs.items():
        print(f"  {key}: {value}")

    # Check for version-specific keys
    if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
        print("\n✅ Using transformers 4.36+ API")
        assert 'eval_strategy' in training_kwargs, "Missing eval_strategy"
        assert 'evaluation_strategy' not in training_kwargs, "Should not have evaluation_strategy"
        assert 'logging_dir' not in training_kwargs, "Should not have logging_dir"
        print("  - eval_strategy: ✓")
        print("  - logging_dir removed: ✓")
    else:
        print("\n✅ Using transformers <4.36 API")
        assert 'evaluation_strategy' in training_kwargs, "Missing evaluation_strategy"
        assert 'eval_strategy' not in training_kwargs, "Should not have eval_strategy"
        assert 'logging_dir' in training_kwargs, "Missing logging_dir"
        print("  - evaluation_strategy: ✓")
        print("  - logging_dir: ✓")

    # Test tokenizer handling
    print("\n" + "=" * 70)
    print("Testing Trainer tokenizer compatibility...")
    print("=" * 70)

    should_pass = should_pass_tokenizer_to_trainer()
    print(f"\nShould pass tokenizer to Trainer.__init__(): {should_pass}")

    if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
        assert not should_pass, "Should NOT pass tokenizer in 4.36+"
        print("  ✅ Correctly returns False for transformers 4.36+")
    else:
        assert should_pass, "Should pass tokenizer in <4.36"
        print("  ✅ Correctly returns True for transformers <4.36")

    # Test training_step signature
    print("\n" + "=" * 70)
    print("Testing training_step signature compatibility...")
    print("=" * 70)

    accepts_num_items = training_step_accepts_num_items(None)
    print(f"\ntraining_step accepts num_items_in_batch: {accepts_num_items}")

    if TRANSFORMERS_VERSION >= TRANSFORMERS_4_36:
        assert accepts_num_items, "Should accept num_items_in_batch in 4.36+"
        print("  ✅ Correctly returns True for transformers 4.36+")
    else:
        assert not accepts_num_items, "Should NOT accept num_items_in_batch in <4.36"
        print("  ✅ Correctly returns False for transformers <4.36")

    print("\n" + "=" * 70)
    print("ALL COMPATIBILITY TESTS PASSED! ✅")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
