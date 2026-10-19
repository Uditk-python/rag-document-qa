"""
Metrics for the evaluation harness, split into two families:

1. Retrieval quality (standard IR metrics) -- do we find the right chunks?
   These only need the labeled "relevant chunk ids" per question, not any
   LLM call, so they're fast, free, and deterministic.

2. Answer quality -- given the retrieved chunks, is the generated answer
   actually correct and faithful? Correctness uses a cheap keyword-overlap
   F1 by default, with an optional LLM-judge for a richer signal when an
   API key is available. Faithfulness reuses src/citations.py so "did the
   demo's citation checker work" and "how faithful is this system overall"
   are measured the same way.
"""

import re
from typing import List

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


# --- Retrieval metrics -----------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for r in relevant_ids if r in retrieved_set)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for r in retrieved_ids if r in relevant_set)
    return hits / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Binary-relevance nDCG: relevant chunks are worth 1, others 0."""
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids):
        rel = 1.0 if rid in relevant_set else 0.0
        dcg += rel / _log2(i + 2)

    ideal_hits = min(len(relevant_set), len(retrieved_ids))
    idcg = sum(1.0 / _log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _log2(x: float) -> float:
    import math
    return math.log2(x)


# --- Answer-quality metrics -------------------------------------------------------

def keyword_overlap_f1(generated_answer: str, reference_answer: str) -> float:
    """A cheap, dependency-free stand-in for answer correctness: token-level
    F1 between the generated answer and a reference answer, à la SQuAD's
    metric. Not a substitute for human judgement, but a fast, reproducible
    signal that doesn't need an API key -- good for CI/regression checks."""
    gen_tokens = _tokenize(generated_answer)
    ref_tokens = _tokenize(reference_answer)
    if not ref_tokens:
        return 1.0
    if not gen_tokens:
        return 0.0

    overlap = gen_tokens & ref_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(gen_tokens)
    recall = len(overlap) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def llm_judge_score(question: str, generated_answer: str, reference_answer: str) -> float:
    """Optional richer correctness signal: ask the LLM to rate the generated
    answer against the reference on a 1-5 scale. Requires an API key; the
    caller should catch exceptions and skip this metric if none is set."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        prompt = (
            f"Question: {question}\n\n"
            f"Reference (correct) answer: {reference_answer}\n\n"
            f"Candidate answer to grade: {generated_answer}\n\n"
            "On a scale of 1-5, how factually consistent is the candidate answer "
            "with the reference answer? 5 = fully consistent, 1 = contradicts it. "
            "Respond with ONLY the number."
        )
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        match = re.search(r"[1-5]", text)
        return float(match.group()) if match else 3.0

    raise RuntimeError("No API key available for LLM-judge scoring.")
