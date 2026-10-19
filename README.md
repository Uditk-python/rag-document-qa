# RAG-Based Document Q&A Assistant

An assistant that answers questions from your own PDFs/text files using
Retrieval-Augmented Generation — with **inline source citations**, a
**retrieval + answer evaluation harness**, and a **Streamlit** front end.

This isn't a "load a PDF, chunk it, throw it at an LLM" demo. The parts that
usually get skipped are the point of this project:

- **Citations that are verified, not just generated.** The LLM is forced to
  tag every claim with `[S1]`, `[S2]`... referencing the exact chunks it
  used. A separate faithfulness checker then verifies that cited claims
  actually appear in the cited chunk, and flags unsupported ("hallucinated")
  sentences instead of trusting the model's word for it.
- **A retrieval pipeline with more than one knob.** Sentence-aware chunking
  with configurable overlap, cosine similarity search over FAISS, a
  similarity-score floor so junk chunks aren't forced into context, and
  optional Maximal Marginal Relevance (MMR) re-ranking so the top-k isn't
  five near-duplicate chunks from the same paragraph.
- **An evaluation harness, not a demo.** `eval/run_eval.py` runs a labeled
  question set through the pipeline and reports standard IR metrics
  (Recall@k, Precision@k, MRR, nDCG) for retrieval, plus keyword-overlap and
  LLM-judge scores for answer correctness/faithfulness. This is what you'd
  show in an interview to prove the system was *measured*, not just built.

## Architecture

```
                 ┌──────────────┐
   PDFs/.txt --> │  ingestion   │  extract text + per-page metadata
                 └──────┬───────┘
                        v
                 ┌──────────────┐
                 │  chunking    │  sentence-aware, overlapping windows
                 └──────┬───────┘
                        v
                 ┌──────────────┐
                 │  embeddings  │  sentence-transformers (MiniLM)
                 └──────┬───────┘
                        v
                 ┌──────────────┐
                 │ vector_store │  FAISS index (cosine via normalized IP)
                 └──────┬───────┘
                        v
   query -->    ┌──────────────┐
                 │  retriever   │  top-k search + score floor + MMR
                 └──────┬───────┘
                        v
                 ┌──────────────┐
                 │     llm      │  answer + inline [S#] citations
                 └──────┬───────┘
                        v
                 ┌──────────────┐
                 │  evaluation  │  retrieval IR metrics + answer scoring
                 └──────────────┘
```

All of this is wired together in `src/rag_pipeline.py` and exposed through
`app.py` (Streamlit) and `eval/run_eval.py` (CLI evaluation).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### LLM backend

The project works with **either** a hosted LLM API or a fully local/offline
fallback so you can run it with zero API keys for a demo:

```bash
# Option A: Anthropic (recommended)
export ANTHROPIC_API_KEY=sk-ant-...

# Option B: OpenAI
export OPENAI_API_KEY=sk-...

# Option C: nothing set -> falls back to an extractive answerer
# (no generation, just returns the most relevant sentences verbatim
#  with citations — useful for testing the retrieval pipeline alone)
```

Set the backend explicitly in `config.py` (`LLM_BACKEND = "anthropic" | "openai" | "extractive"`)
or leave it on `"auto"` to pick based on which key is present.

## Running the app

```bash
streamlit run app.py
```

1. Upload one or more PDFs/.txt files in the sidebar.
2. Click **Build Index** (chunks + embeds + writes the FAISS index to `./storage`).
3. Ask a question. The answer streams back with `[S1]`, `[S2]`... citations;
   expand "Show retrieved sources" to see the exact chunks and page numbers,
   and which cited claims passed/failed the faithfulness check.

## Running the evaluation harness

```bash
python -m eval.run_eval --docs data/sample_docs --qa eval/eval_dataset.json
```

This builds a fresh index over `--docs`, runs every question in the labeled
set through retrieval + generation, and prints a report like:

```
=== Retrieval quality (k=4) ===
Recall@4:     0.92
Precision@4:  0.41
MRR:          0.87
nDCG@4:       0.89

=== Answer quality ===
Keyword-overlap F1 (avg):     0.78
Citation faithfulness (avg):  0.94   (fraction of cited claims supported by their source chunk)
LLM-judge correctness (avg):  4.3 / 5   (skipped if no API key set)
```

`eval/eval_dataset.json` ships with a small labeled set built against the
sample docs in `data/sample_docs/` so the harness runs out of the box —
swap in your own docs + questions to evaluate on your real use case.

## Project layout

```
rag-document-qa/
├── app.py                   Streamlit UI
├── config.py                All tunables in one place
├── requirements.txt
├── src/
│   ├── ingestion.py         PDF/text -> raw text + page metadata
│   ├── chunking.py          Sentence-aware overlapping chunker
│   ├── embeddings.py        sentence-transformers wrapper
│   ├── vector_store.py      FAISS index build/save/load/search
│   ├── retriever.py         top-k + score floor + MMR re-ranking
│   ├── llm.py                Anthropic/OpenAI/extractive answer generation
│   ├── citations.py         Citation parsing + faithfulness verification
│   └── rag_pipeline.py      Wires it all together
├── eval/
│   ├── eval_dataset.json    Labeled question -> relevant chunk(s) set
│   ├── metrics.py           Recall/Precision/MRR/nDCG + answer scoring
│   └── run_eval.py          CLI entry point
└── data/sample_docs/        Small sample corpus for demo/eval
```

## Design notes worth mentioning in an interview

- **Why FAISS over a managed vector DB:** zero infra, runs locally, and the
  abstraction in `vector_store.py` is thin enough to swap for Chroma/Pinecone
  by implementing the same 3 methods (`add`, `search`, `save`/`load`).
- **Why sentence-aware chunking instead of fixed-char windows:** fixed-width
  chunking regularly splits a sentence (and its citation-worthy fact) in
  half. Chunking on sentence boundaries with overlap keeps facts intact
  while still giving the retriever fine-grained units.
- **Why MMR re-ranking:** plain top-k similarity search tends to return
  near-duplicate chunks (e.g., five chunks from the same repetitive section).
  MMR trades a little relevance for diversity so the LLM sees more distinct
  information within its context budget.
- **Why verify citations instead of trusting the model:** LLMs cite
  confidently even when wrong. `citations.py` re-checks each `[S#]`-tagged
  sentence against the actual source chunk using lexical overlap, so the UI
  can visibly flag unsupported claims rather than silently trusting them.
