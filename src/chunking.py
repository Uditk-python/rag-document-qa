"""
Chunking: split page text into overlapping, sentence-aware windows.

Naive chunkers just cut every N characters, which routinely slices a
sentence (and the fact/number/date it contains) in half. That's especially
bad for a Q&A + citation system: a citation pointing at half a sentence
is worse than useless. Instead we:

  1. Split into sentences with a lightweight regex (no heavy NLP dependency
     needed for this).
  2. Greedily pack sentences into a chunk until adding the next one would
     exceed CHUNK_SIZE.
  3. Start the next chunk by re-including the trailing sentences of the
     previous chunk up to CHUNK_OVERLAP characters, so information near a
     chunk boundary is still retrievable from both sides.
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.ingestion import Page
import config


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page_number: int
    chunk_index: int  # position of this chunk within its source page


def split_into_sentences(text: str) -> List[str]:
    # Collapse excessive whitespace/newlines first -- PDFs are messy.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(
    page: Page,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    min_chunk_size: int = config.MIN_CHUNK_SIZE,
) -> List[Chunk]:
    sentences = split_into_sentences(page.text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_len = 0
    chunk_idx = 0

    def flush():
        nonlocal current_sentences, current_len, chunk_idx
        if not current_sentences:
            return
        text = " ".join(current_sentences)
        if len(text) >= min_chunk_size or not chunks:
            chunks.append(
                Chunk(
                    id=f"{page.source}::p{page.page_number}::c{chunk_idx}",
                    text=text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current_len + sentence_len > chunk_size and current_sentences:
            flush()
            # Build overlap: keep trailing sentences up to `overlap` chars
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len

    flush()
    return chunks


def chunk_pages(pages: List[Page]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks
