"""
Retriever: sits between the raw vector store and the LLM. This is where
"search" turns into "the right context to hand the model", via three
deliberate steps beyond plain top-k:

  1. Pull CANDIDATE_K (more than we need) from FAISS.
  2. Drop candidates below a similarity floor -- if nothing in the corpus
     is actually relevant, we'd rather say "not found" than force-feed the
     LLM the least-bad-of-a-bad-bunch and let it hallucinate an answer.
  3. Re-rank the survivors with Maximal Marginal Relevance (MMR) so the
     final top-k isn't five near-duplicate chunks from the same paragraph.
     MMR balances "is this relevant to the query" against "is this
     different from what I've already picked".
"""

from dataclasses import dataclass
from typing import List

import numpy as np

import config
from src.chunking import Chunk
from src.embeddings import embed_query, embed_texts
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity to the query, 0-1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are already L2-normalized


def mmr_rerank(
    query_embedding: np.ndarray,
    candidates: List[RetrievedChunk],
    candidate_embeddings: np.ndarray,
    k: int,
    lambda_param: float = config.MMR_LAMBDA,
) -> List[RetrievedChunk]:
    """Greedy MMR selection. lambda_param=1 -> pure relevance ranking,
    lambda_param=0 -> pure diversity (ignores relevance after the first pick)."""
    if len(candidates) <= k:
        return candidates

    selected_idx: List[int] = []
    remaining_idx = list(range(len(candidates)))

    while len(selected_idx) < k and remaining_idx:
        best_score = -1e9
        best_i = None
        for i in remaining_idx:
            relevance = _cosine(query_embedding, candidate_embeddings[i])
            if selected_idx:
                diversity_penalty = max(
                    _cosine(candidate_embeddings[i], candidate_embeddings[j])
                    for j in selected_idx
                )
            else:
                diversity_penalty = 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

    return [candidates[i] for i in selected_idx]


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = config.TOP_K,
    candidate_k: int = config.CANDIDATE_K,
    similarity_floor: float = config.SIMILARITY_FLOOR,
    use_mmr: bool = config.USE_MMR,
) -> List[RetrievedChunk]:
    query_embedding = embed_query(query)
    raw_results = store.search(query_embedding, k=candidate_k)

    candidates = [
        RetrievedChunk(chunk=chunk, score=score)
        for chunk, score in raw_results
        if score >= similarity_floor
    ]

    if not candidates:
        return []

    if not use_mmr or len(candidates) <= top_k:
        return candidates[:top_k]

    # Re-embed candidate texts for diversity comparison. In practice you'd
    # cache these alongside the FAISS index rather than recomputing, but
    # re-embedding a handful of short chunks is cheap and keeps this
    # function self-contained.
    candidate_embeddings = embed_texts([rc.chunk.text for rc in candidates])
    return mmr_rerank(query_embedding, candidates, candidate_embeddings, k=top_k)
