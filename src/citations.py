"""
Citations: parse the [S1], [S2]... tags the LLM is instructed to emit, and
independently verify each cited sentence against its claimed source chunk.

Why verify at all? Because an LLM citing a source is not the same as an
LLM being *right* to cite it -- models attach citations confidently
regardless of whether the source actually backs the claim. Treating the
citation as a claim to be checked (rather than ground truth) is what turns
"looks trustworthy" into "is trustworthy".

The check here is deliberately simple (token/lexical overlap) rather than
another LLM call, so it's fast, free, deterministic, and has no risk of
the verifier sharing the generator's blind spots. It's a heuristic, not a
proof -- documented as such -- but it reliably catches the common failure
mode of a model citing the wrong chunk or asserting something the chunk
doesn't say.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

import config
from src.retriever import RetrievedChunk

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class CitedClaim:
    sentence: str
    citation_ids: List[int]   # e.g. [1, 2] for a sentence citing [S1][S2]
    supported: bool
    best_overlap: float       # lexical overlap score with the best-matching cited source


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _lexical_overlap(claim: str, source: str) -> float:
    """Fraction of the claim's content words that also appear in the source.
    Cheap stand-in for semantic entailment -- good enough to catch a model
    citing the wrong chunk or inventing specifics the chunk never states."""
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return 1.0
    source_tokens = _tokenize(source)
    overlap = claim_tokens & source_tokens
    return len(overlap) / len(claim_tokens)


def parse_and_verify(
    answer_text: str,
    retrieved: List[RetrievedChunk],
    overlap_threshold: float = config.FAITHFULNESS_OVERLAP_THRESHOLD,
) -> List[CitedClaim]:
    """Split the answer into sentences, find [S#] tags in each, and check
    each cited sentence's lexical overlap against its claimed source(s)."""
    source_by_id: Dict[int, str] = {
        i + 1: rc.chunk.text for i, rc in enumerate(retrieved)
    }

    # Split on sentence-final punctuation while keeping citation tags attached.
    raw_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())

    claims: List[CitedClaim] = []
    for raw in raw_sentences:
        if not raw.strip():
            continue
        ids = [int(m) for m in _CITATION_RE.findall(raw)]
        clean_sentence = _CITATION_RE.sub("", raw).strip()
        if not ids:
            # Uncited sentence (e.g. a transition phrase) -- not a claim to verify.
            continue

        best_overlap = 0.0
        for cid in ids:
            source_text = source_by_id.get(cid)
            if source_text is None:
                continue
            best_overlap = max(best_overlap, _lexical_overlap(clean_sentence, source_text))

        claims.append(
            CitedClaim(
                sentence=clean_sentence,
                citation_ids=ids,
                supported=best_overlap >= overlap_threshold,
                best_overlap=round(best_overlap, 3),
            )
        )

    return claims


def faithfulness_score(claims: List[CitedClaim]) -> float:
    """Fraction of cited claims that passed verification. 1.0 if there
    were no citations to check (avoids penalizing e.g. 'I don't know' answers)."""
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if c.supported)
    return supported / len(claims)
