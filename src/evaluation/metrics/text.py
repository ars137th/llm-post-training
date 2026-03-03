"""
Text Evaluation Metrics

Implements various metrics for evaluating text generation quality:
- BLEU: N-gram overlap with reference
- ROUGE: Recall-oriented n-gram matching
- Perplexity: Model confidence
- Diversity: Lexical diversity metrics
"""

from typing import List, Dict, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import re


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU scores (1-gram through max_n-gram).

    BLEU measures n-gram overlap between prediction and reference.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        max_n: Maximum n-gram order (default 4 for BLEU-4)

    Returns:
        Dictionary with BLEU-1 through BLEU-max_n scores

    Note:
        This is a simplified implementation. For production, use
        evaluate.load("bleu") from HuggingFace.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    scores = {f"bleu_{n}": [] for n in range(1, max_n + 1)}

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)

        for n in range(1, max_n + 1):
            score = _compute_ngram_overlap(pred_tokens, ref_tokens, n)
            scores[f"bleu_{n}"].append(score)

    # Average across examples
    return {k: np.mean(v) for k, v in scores.items()}


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute ROUGE scores.

    ROUGE measures recall-oriented n-gram overlap.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        rouge_types: Types of ROUGE to compute ("rouge1", "rouge2", "rougeL")

    Returns:
        Dictionary with ROUGE scores

    Note:
        This is a simplified implementation. For production, use
        evaluate.load("rouge") from HuggingFace.
    """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    scores = {rt: [] for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)

        if "rouge1" in rouge_types:
            score = _compute_rouge_n(pred_tokens, ref_tokens, 1)
            scores["rouge1"].append(score)

        if "rouge2" in rouge_types:
            score = _compute_rouge_n(pred_tokens, ref_tokens, 2)
            scores["rouge2"].append(score)

        if "rougeL" in rouge_types:
            score = _compute_rouge_l(pred_tokens, ref_tokens)
            scores["rougeL"].append(score)

    # Average across examples
    return {k: np.mean(v) for k, v in scores.items()}


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from model logits.

    Perplexity measures how well the model predicts the text.
    Lower is better (model is more confident).

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore (typically -100 for padding)

    Returns:
        Perplexity value
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross-entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean',
    )

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss)
    return perplexity.item()


def compute_diversity(
    texts: List[str],
    n: int = 2,
) -> Dict[str, float]:
    """
    Compute lexical diversity metrics.

    Measures how diverse the generated texts are:
    - distinct-n: Ratio of unique n-grams to total n-grams
    - entropy: Shannon entropy of n-gram distribution

    Args:
        texts: List of generated texts
        n: N-gram order (default 2)

    Returns:
        Dictionary with diversity metrics
    """
    all_ngrams = []

    for text in texts:
        tokens = _tokenize(text)
        ngrams = _get_ngrams(tokens, n)
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return {"distinct_n": 0.0, "entropy": 0.0}

    # Distinct-n: unique / total
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    distinct_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    # Entropy
    ngram_counts = Counter(all_ngrams)
    total = sum(ngram_counts.values())
    probs = [count / total for count in ngram_counts.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)

    return {
        f"distinct_{n}": distinct_n,
        f"entropy_{n}": entropy,
    }


def compute_repetition(texts: List[str]) -> Dict[str, float]:
    """
    Compute repetition metrics.

    Measures how repetitive the generated text is.

    Args:
        texts: List of generated texts

    Returns:
        Dictionary with repetition metrics
    """
    repetitions = []

    for text in texts:
        tokens = _tokenize(text)
        if len(tokens) < 2:
            repetitions.append(0.0)
            continue

        # Count repeated adjacent tokens
        repeated = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
        repetition_rate = repeated / (len(tokens) - 1)
        repetitions.append(repetition_rate)

    return {
        "repetition_rate": np.mean(repetitions),
    }


# Helper functions

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    # Lowercase and split on whitespace
    text = text.lower()
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _compute_ngram_overlap(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Compute n-gram precision (BLEU-style)."""
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    pred_ngrams = Counter(_get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_get_ngrams(ref_tokens, n))

    # Count matches
    matches = sum((pred_ngrams & ref_ngrams).values())
    total = sum(pred_ngrams.values())

    return matches / total if total > 0 else 0.0


def _compute_rouge_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Compute ROUGE-n (recall-based)."""
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    pred_ngrams = Counter(_get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_get_ngrams(ref_tokens, n))

    # Count matches (recall-based)
    matches = sum((pred_ngrams & ref_ngrams).values())
    total_ref = sum(ref_ngrams.values())

    return matches / total_ref if total_ref > 0 else 0.0


def _compute_rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """
    Compute ROUGE-L (longest common subsequence).

    Measures the longest common subsequence between prediction and reference.
    """
    lcs_length = _lcs_length(pred_tokens, ref_tokens)

    # F1 score of LCS
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


class TextMetrics:
    """
    Unified interface for computing text generation metrics.

    Example:
        >>> metrics = TextMetrics()
        >>> results = metrics.compute(
        ...     predictions=["Hello world", "How are you"],
        ...     references=["Hello there", "How are you doing"],
        ... )
        >>> print(results)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize metrics computer.

        Args:
            metrics: List of metric names to compute
                    ("bleu", "rouge", "diversity", "repetition")
                    If None, computes all metrics.
        """
        if metrics is None:
            self.metrics = ["bleu", "rouge", "diversity", "repetition"]
        else:
            self.metrics = metrics

    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute all requested metrics.

        Args:
            predictions: List of predicted texts
            references: List of reference texts (required for BLEU/ROUGE)

        Returns:
            Dictionary of computed metrics
        """
        results = {}

        if "bleu" in self.metrics:
            if references is None:
                raise ValueError("references required for BLEU")
            bleu_scores = compute_bleu(predictions, references)
            results.update(bleu_scores)

        if "rouge" in self.metrics:
            if references is None:
                raise ValueError("references required for ROUGE")
            rouge_scores = compute_rouge(predictions, references)
            results.update(rouge_scores)

        if "diversity" in self.metrics:
            diversity_scores = compute_diversity(predictions, n=2)
            results.update(diversity_scores)

        if "repetition" in self.metrics:
            repetition_scores = compute_repetition(predictions)
            results.update(repetition_scores)

        return results
