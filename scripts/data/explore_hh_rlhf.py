"""
Explore Anthropic HH-RLHF Dataset

This script loads and displays examples from the Anthropic HH-RLHF dataset
to help understand the data format and content.

Usage:
    python scripts/data/explore_hh_rlhf.py
    python scripts/data/explore_hh_rlhf.py --num_examples 5
    python scripts/data/explore_hh_rlhf.py --split test
"""

import sys
from pathlib import Path
import argparse
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset


def parse_conversation(text):
    """Parse conversational format into turns."""
    turns = []
    current_speaker = None
    current_text = []

    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('Human:'):
            if current_speaker:
                turns.append((current_speaker, '\n'.join(current_text)))
            current_speaker = 'Human'
            current_text = [line[6:].strip()]
        elif line.startswith('Assistant:'):
            if current_speaker:
                turns.append((current_speaker, '\n'.join(current_text)))
            current_speaker = 'Assistant'
            current_text = [line[10:].strip()]
        elif line:
            current_text.append(line)

    if current_speaker:
        turns.append((current_speaker, '\n'.join(current_text)))

    return turns


def display_example(example, idx, detailed=False):
    """Display a preference pair example."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE {idx}")
    print(f"{'='*80}")

    if detailed:
        print("\n🔍 Detailed View (Turn-by-Turn):")
        print("\n✅ CHOSEN (Preferred Response):")
        print("-" * 80)
        chosen_turns = parse_conversation(example['chosen'])
        for i, (speaker, text) in enumerate(chosen_turns):
            print(f"\n[Turn {i+1}] {speaker}:")
            print(text)

        print(f"\n\n❌ REJECTED (Not Preferred):")
        print("-" * 80)
        rejected_turns = parse_conversation(example['rejected'])
        for i, (speaker, text) in enumerate(rejected_turns):
            print(f"\n[Turn {i+1}] {speaker}:")
            print(text)

        # Highlight the difference
        print(f"\n\n💡 Key Difference:")
        print("-" * 80)
        if len(chosen_turns) > 0 and len(rejected_turns) > 0:
            chosen_last = chosen_turns[-1][1] if chosen_turns[-1][0] == 'Assistant' else "N/A"
            rejected_last = rejected_turns[-1][1] if rejected_turns[-1][0] == 'Assistant' else "N/A"

            print("Last Assistant response in CHOSEN:")
            print(chosen_last[:200] + "..." if len(chosen_last) > 200 else chosen_last)
            print("\nLast Assistant response in REJECTED:")
            print(rejected_last[:200] + "..." if len(rejected_last) > 200 else rejected_last)
    else:
        print("\n✅ CHOSEN:")
        print("-" * 80)
        print(example['chosen'][:500] + "..." if len(example['chosen']) > 500 else example['chosen'])

        print(f"\n❌ REJECTED:")
        print("-" * 80)
        print(example['rejected'][:500] + "..." if len(example['rejected']) > 500 else example['rejected'])


def analyze_dataset(dataset_split):
    """Analyze dataset statistics."""
    print(f"\n📊 Dataset Analysis:")
    print(f"  Total examples: {len(dataset_split):,}")

    # Length statistics
    chosen_lengths = [len(ex['chosen']) for ex in dataset_split.select(range(min(1000, len(dataset_split))))]
    rejected_lengths = [len(ex['rejected']) for ex in dataset_split.select(range(min(1000, len(dataset_split))))]

    print(f"\n  Character lengths (sample of 1000):")
    print(f"    Chosen avg: {sum(chosen_lengths)/len(chosen_lengths):.0f} chars")
    print(f"    Rejected avg: {sum(rejected_lengths)/len(rejected_lengths):.0f} chars")
    print(f"    Chosen min/max: {min(chosen_lengths)} / {max(chosen_lengths)}")
    print(f"    Rejected min/max: {min(rejected_lengths)} / {max(rejected_lengths)}")

    # Count turns
    example = dataset_split[0]
    chosen_turns = len(parse_conversation(example['chosen']))
    rejected_turns = len(parse_conversation(example['rejected']))

    print(f"\n  Conversational turns (first example):")
    print(f"    Chosen: {chosen_turns} turns")
    print(f"    Rejected: {rejected_turns} turns")

    print(f"\n  Format: Conversational (Human:/Assistant: alternating turns)")


def main():
    parser = argparse.ArgumentParser(description="Explore Anthropic HH-RLHF dataset")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of examples to display")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test"],
                       help="Dataset split to explore")
    parser.add_argument("--random", action="store_true",
                       help="Show random examples instead of first N")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed turn-by-turn view")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only show statistics, no examples")

    args = parser.parse_args()

    print("Loading Anthropic HH-RLHF dataset...")
    print("(This may take a minute on first run)")
    dataset = load_dataset("Anthropic/hh-rlhf")

    split = dataset[args.split]

    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Split: {args.split}")
    print(f"   Examples: {len(split):,}")

    # Analyze dataset
    analyze_dataset(split)

    if args.analyze_only:
        return

    # Show examples
    print(f"\n\n{'='*80}")
    print(f"SHOWING {args.num_examples} EXAMPLES FROM {args.split.upper()} SET")
    print(f"{'='*80}")

    if args.random:
        indices = random.sample(range(len(split)), args.num_examples)
    else:
        indices = range(min(args.num_examples, len(split)))

    for i, idx in enumerate(indices, 1):
        example = split[idx]
        display_example(example, i, detailed=args.detailed)

    print(f"\n\n{'='*80}")
    print("Exploration complete!")
    print(f"{'='*80}")

    print(f"\n💡 Tips:")
    print(f"  - Use --detailed to see turn-by-turn conversation breakdown")
    print(f"  - Use --random to see different examples each time")
    print(f"  - Use --num_examples 10 to see more examples")
    print(f"  - Use --split test to explore the test set")

    print(f"\n📚 Dataset Info:")
    print(f"  - Source: Anthropic HH-RLHF (Helpful & Harmless)")
    print(f"  - Purpose: Training reward models for RLHF")
    print(f"  - Format: Preference pairs (chosen vs rejected)")
    print(f"  - Human annotations: Real human preferences")


if __name__ == "__main__":
    main()
