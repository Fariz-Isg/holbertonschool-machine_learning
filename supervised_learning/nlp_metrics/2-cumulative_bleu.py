#!/usr/bin/env python3
"""Cumulative n-gram BLEU score without external dependencies.

Computes the standard BLEU metric as defined in Papineni et al. (2002):
the brevity penalty multiplied by the geometric mean of clipped n-gram
precisions for orders 1 through n, with uniform weights.
"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, order):
    """Generate contiguous n-grams from a tokenized sentence.

    Args:
        sentence (list[str]): Tokenized sentence.
        order (int): N-gram order (n >= 1).

    Returns:
        list[str]: List of n-grams joined by single spaces.
    """
    ngrams = []
    for i in range(len(sentence) - order + 1):
        ngram = sentence[i: i + order]
        ngrams.append(' '.join(ngram))

    return ngrams


def modified_precision(references, sentence, order):
    """Calculate clipped n-gram precision for a given order.

    Args:
        references (list[list[str]]): Reference translations; tokenized.
        sentence (list[str]): Candidate translation; tokenized.
        order (int): N-gram order.

    Returns:
        float: Clipped precision in [0, 1]. Returns 0 if no n-grams exist.
    """
    sentence_ngrams = Counter(generate_ngram(sentence, order))

    if not sentence_ngrams:
        return 0

    max_counts = {}
    for reference in references:
        ref_ngrams = Counter(generate_ngram(reference, order))
        for ngram in sentence_ngrams:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    ref_ngrams[ngram])

    clipped_counts = {
        ngram: min(count, max_counts.get(ngram, 0))
        for ngram, count in sentence_ngrams.items()
    }

    numerator = sum(clipped_counts.values())
    denominator = max(1, len(sentence) - order + 1)

    return numerator / denominator


def cumulative_bleu(references, sentence, n):
    """Compute cumulative n-gram BLEU score for a candidate sentence.

    The cumulative BLEU score is the brevity penalty multiplied by the
    geometric mean of clipped n-gram precisions for orders 1 through n,
    with each order weighted equally (1/n).

    Args:
        references (list[list[str]]): Reference translations; each is a
            list of tokens.
        sentence (list[str]): Candidate translation as a list of tokens.
        n (int): Largest n-gram order to use for evaluation.

    Returns:
        float: Cumulative n-gram BLEU score.
    """
    len_sentence = len(sentence)

    # Brevity penalty
    closest_ref_len = min((abs(len(ref) - len_sentence), len(ref))
                          for ref in references)[1]
    if len_sentence > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / len_sentence)

    # Collect clipped precisions for each order 1..n
    precisions = []
    for order in range(1, n + 1):
        precisions.append(modified_precision(references, sentence, order))

    # Geometric mean with uniform weights (log-space to avoid underflow)
    precisions = np.array(precisions)
    if np.any(precisions == 0):
        return 0

    log_avg = np.sum((1 / n) * np.log(precisions))
    bleu_score = BP * np.exp(log_avg)

    return bleu_score
