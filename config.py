"""
Central place for every tunable knob in the pipeline. Keeping these here
(instead of scattered magic numbers) is what makes the retrieval pipeline
something you can actually tune and evaluate rather than a black box.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

INDEX_PATH = STORAGE_DIR / "faiss.index"
METADATA_PATH = STORAGE_DIR / "chunks_metadata.json"

# --- Chunking ------------------------------------------------------------
# Chunk size / overlap are measured in characters (simple, dependency-free,
# and language-agnostic). Sentence boundaries are still respected -- see
# src/chunking.py -- so this is a target, not a hard cut.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100  # drop trailing scraps smaller than this

# --- Embeddings ----------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match the model above

# --- Retrieval -------------------------------------------------------------
TOP_K = 4                     # chunks finally passed to the LLM
CANDIDATE_K = 12              # chunks pulled from FAISS before re-ranking
SIMILARITY_FLOOR = 0.15       # drop candidates below this cosine similarity
USE_MMR = True                # diversify results instead of raw top-k
MMR_LAMBDA = 0.6              # 1.0 = pure relevance, 0.0 = pure diversity

# --- LLM generation --------------------------------------------------------
# "auto" picks anthropic -> openai -> extractive based on which API key
# is present in the environment.
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "auto")
ANTHROPIC_MODEL = "claude-sonnet-4-5"
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"  # gemini-2.0-flash was shut down June 2026
OLLAMA_MODEL = "llama3.2"  # free, local, no API key -- see README for setup
MAX_ANSWER_TOKENS = 800
GENERATION_TEMPERATURE = 0.1  # low temperature: this is Q&A, not creative writing

# --- Evaluation ------------------------------------------------------------
EVAL_TOP_K = TOP_K
FAITHFULNESS_OVERLAP_THRESHOLD = 0.35  # min lexical overlap to count as "supported"
