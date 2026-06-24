from typing import List, Set
import re


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def groundedness_score(answer: str, chunks: List[str]) -> float:
    """
    Heuristic groundedness: fraction of answer sentences that contain
    at least one n-gram overlap with the retrieved context.
    """
    if not answer or not chunks:
        return 0.0

    context = " ".join(chunks).lower()
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0

    grounded = 0
    for sent in sentences:
        words = set(sent.lower().split())
        # 3-gram overlap check
        context_words = set(context.split())
        overlap = len(words & context_words) / max(len(words), 1)
        if overlap >= 0.3:
            grounded += 1

    return grounded / len(sentences)


def citation_accuracy(
    cited_sources: List[str],
    ground_truth_sources: List[str],
) -> float:
    """Fraction of cited sources that are in the ground truth source list."""
    if not cited_sources:
        return 0.0
    if not ground_truth_sources:
        return 1.0  # No ground truth to check against

    gt_set = set(s.lower() for s in ground_truth_sources)
    correct = sum(1 for s in cited_sources if s.lower() in gt_set)
    return correct / len(cited_sources)
