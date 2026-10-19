"""
Embeddings: thin wrapper around sentence-transformers so the rest of the
codebase depends on a stable interface (`embed_texts`, `embed_query`)
rather than the library directly. Swapping models or providers (e.g. to an
API-based embedding model) means changing only this file.
"""

from functools import lru_cache
from typing import List

import numpy as np

import config


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily: this import is slow (~seconds) and pulls in torch,
    # so we don't pay that cost until embeddings are actually needed
    # (e.g. `python -m eval.run_eval --help` shouldn't import torch).
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of chunk texts. Returns an (N, dim) float32 array,
    L2-normalized so inner product == cosine similarity in FAISS."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a (dim,) float32 array."""
    return embed_texts([query])[0]
