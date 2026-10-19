"""
RagPipeline: the single object the UI (app.py) and the eval harness both
use. Wiring everything through one class keeps app.py and run_eval.py
from duplicating pipeline logic (and drifting out of sync with each other).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import config
from src.chunking import chunk_pages
from src.citations import CitedClaim, faithfulness_score, parse_and_verify
from src.embeddings import embed_texts
from src.ingestion import load_documents
from src.llm import GeneratedAnswer, generate_answer
from src.retriever import RetrievedChunk, retrieve
from src.vector_store import VectorStore


@dataclass
class QAResult:
    question: str
    answer: GeneratedAnswer
    retrieved: List[RetrievedChunk]
    cited_claims: List[CitedClaim]
    faithfulness: float  # fraction of cited claims verified as supported


class RagPipeline:
    def __init__(self):
        self.store: VectorStore = VectorStore()

    # --- Indexing ----------------------------------------------------------

    def build_index(self, file_paths: List[Path]) -> int:
        """Ingest, chunk, embed, and index the given documents.
        Returns the number of chunks indexed."""
        pages = load_documents(file_paths)
        if not pages:
            raise ValueError(
                "No extractable text found in the provided files. "
                "Scanned/image-only PDFs need OCR before they can be indexed."
            )
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("Chunking produced no chunks -- documents may be empty.")

        embeddings = embed_texts([c.text for c in chunks])
        self.store = VectorStore()
        self.store.add(embeddings, chunks)
        return len(chunks)

    def save(self) -> None:
        self.store.save()

    def load(self) -> None:
        self.store = VectorStore.load()

    # --- Querying ------------------------------------------------------------

    def answer_question(self, question: str) -> QAResult:
        retrieved = retrieve(question, self.store)
        generated = generate_answer(question, retrieved)
        claims = parse_and_verify(generated.text, retrieved)
        score = faithfulness_score(claims)
        return QAResult(
            question=question,
            answer=generated,
            retrieved=retrieved,
            cited_claims=claims,
            faithfulness=score,
        )

    def __len__(self) -> int:
        return len(self.store)
