"""
Vector store: a thin FAISS wrapper exposing exactly the surface area the
rest of the app needs (add / search / save / load). Because the interface
is this small, swapping FAISS for Chroma, pgvector, or Pinecone later means
reimplementing this one class -- nothing else in the codebase changes.

We use IndexFlatIP (inner product) over L2-normalized vectors, which is
mathematically equivalent to cosine similarity search but faster than
computing cosine directly. Flat (exact, brute-force) search is used
deliberately: for a document-Q&A tool with thousands-not-millions of
chunks, exact search is fast enough and removes an entire class of
recall-loss bugs that approximate indexes (IVF/HNSW) can introduce.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

import config
from src.chunking import Chunk


class VectorStore:
    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []  # parallel array: index i <-> chunks[i]

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")
        if len(embeddings) == 0:
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        """Returns up to k (chunk, similarity_score) pairs, best first."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_path: Path = config.INDEX_PATH, metadata_path: Path = config.METADATA_PATH) -> None:
        faiss.write_index(self.index, str(index_path))
        metadata = [asdict(c) for c in self.chunks]
        # encoding="utf-8" is required explicitly -- without it, Path.write_text
        # falls back to the OS's default locale encoding, which on Windows is
        # typically cp1252 ("charmap"). Since chunks can contain any Unicode
        # character a source PDF uses (Greek letters like \u03bb/lambda in
        # scientific notation, curly quotes, em dashes, etc.) and we write
        # with ensure_ascii=False to keep them human-readable in the JSON
        # file, cp1252 can't represent most of them and raises
        # UnicodeEncodeError mid-write.
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path = config.INDEX_PATH, metadata_path: Path = config.METADATA_PATH) -> "VectorStore":
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "No saved index found. Build one first (see app.py 'Build Index' "
                "or eval/run_eval.py)."
            )
        index = faiss.read_index(str(index_path))
        raw_chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw_chunks]
        store = cls(dim=index.d)
        store.index = index
        store.chunks = chunks
        return store

    def __len__(self) -> int:
        return len(self.chunks)
