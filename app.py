"""
Streamlit UI for the RAG document Q&A assistant.

Two tabs:
  - "Chat": upload docs, build the index, ask questions, inspect sources
    and per-sentence citation faithfulness.
  - "Evaluation": run the labeled eval set from the UI and see retrieval +
    answer-quality metrics without touching the command line.
"""

import tempfile
from pathlib import Path

import streamlit as st

import config
from src.llm import _resolve_backend
from src.rag_pipeline import RagPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")


# --- Session state -----------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline()
if "history" not in st.session_state:
    st.session_state.history = []  # list of QAResult
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


def build_index(uploaded_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for f in uploaded_files:
            p = Path(tmpdir) / f.name
            p.write_bytes(f.getbuffer())
            paths.append(p)

        with st.spinner(f"Ingesting, chunking, and embedding {len(paths)} file(s)..."):
            n_chunks = st.session_state.pipeline.build_index(paths)
            st.session_state.pipeline.save()
            st.session_state.indexed_files = [f.name for f in uploaded_files]
            st.session_state.history = []

    st.success(f"Indexed {n_chunks} chunks from {len(paths)} document(s).")


# --- Sidebar: ingestion --------------------------------------------------------

with st.sidebar:
    st.header("📄 Documents")
    backend = _resolve_backend()
    backend_labels = {
        "anthropic": "🟢 Anthropic (Claude)",
        "openai": "🟢 OpenAI",
        "gemini": "🟢 Google Gemini",
        "ollama": "🟢 Ollama (local, free)",
        "extractive": "🟡 No API key — extractive fallback",
    }
    st.caption(f"LLM backend: {backend_labels.get(backend, backend)}")

    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build Index", type="primary", use_container_width=True, disabled=not uploaded_files):
            build_index(uploaded_files)
    with col2:
        if st.button("Load Saved", use_container_width=True):
            try:
                st.session_state.pipeline.load()
                st.success(f"Loaded index with {len(st.session_state.pipeline)} chunks.")
            except FileNotFoundError as e:
                st.error(str(e))

    if st.session_state.indexed_files:
        st.caption("Indexed: " + ", ".join(st.session_state.indexed_files))
    if len(st.session_state.pipeline) > 0:
        st.metric("Chunks in index", len(st.session_state.pipeline))

    st.divider()
    st.subheader("Retrieval settings")
    config.TOP_K = st.slider("Top-K chunks retrieved", 1, 10, config.TOP_K)
    config.USE_MMR = st.checkbox("Use MMR re-ranking (diversify results)", value=config.USE_MMR)


# --- Main: tabs ------------------------------------------------------------

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab_chat:
    st.title("RAG Document Q&A Assistant")
    st.caption(
        "Ask a question about the uploaded documents. Answers cite their sources "
        "as [S1], [S2]... — expand a result to verify each cited claim against "
        "the actual source text."
    )

    if len(st.session_state.pipeline) == 0:
        st.info("Upload documents and click **Build Index** in the sidebar to get started, "
                 "or click **Load Saved** to reuse a previously built index.")
    else:
        question = st.text_input("Your question", placeholder="e.g. How many vacation days do employees get?")
        ask = st.button("Ask", type="primary")

        if ask and question.strip():
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                result = st.session_state.pipeline.answer_question(question.strip())
            st.session_state.history.insert(0, result)

        for result in st.session_state.history:
            st.markdown(f"#### Q: {result.question}")
            st.markdown(result.answer.text)

            badge_color = "green" if result.faithfulness >= 0.8 else (
                "orange" if result.faithfulness >= 0.5 else "red"
            )
            st.markdown(
                f"**Faithfulness score:** :{badge_color}[{result.faithfulness:.0%}] "
                f"of cited claims verified against their source · "
                f"backend: `{result.answer.backend_used}`"
            )

            with st.expander(f"Show {len(result.retrieved)} retrieved source(s) and claim verification"):
                st.markdown("**Retrieved chunks:**")
                for i, rc in enumerate(result.retrieved, start=1):
                    st.markdown(
                        f"**[S{i}]** `{rc.chunk.source}`, page {rc.chunk.page_number} "
                        f"— similarity `{rc.score:.3f}`"
                    )
                    st.text(rc.chunk.text)

                if result.cited_claims:
                    st.markdown("**Citation faithfulness check:**")
                    for claim in result.cited_claims:
                        icon = "✅" if claim.supported else "⚠️"
                        tags = "".join(f"[S{i}]" for i in claim.citation_ids)
                        st.markdown(
                            f"{icon} {claim.sentence} {tags} "
                            f"_(lexical overlap with source: {claim.best_overlap:.2f})_"
                        )
                else:
                    st.caption("No citation tags found in the generated answer.")

            st.divider()

with tab_eval:
    st.title("Evaluation")
    st.caption(
        "Run the labeled question set in eval/eval_dataset.json against the "
        "currently built index and report retrieval + answer-quality metrics."
    )

    eval_path = st.text_input("Path to labeled QA JSON", value="eval/eval_dataset.json")
    run_eval_btn = st.button("Run Evaluation")

    if run_eval_btn:
        if len(st.session_state.pipeline) == 0:
            st.error("Build or load an index first (see the Chat tab).")
        else:
            import json
            import statistics

            from eval.metrics import (
                keyword_overlap_f1,
                ndcg_at_k,
                precision_at_k,
                reciprocal_rank,
                recall_at_k,
            )
            from src.citations import faithfulness_score, parse_and_verify
            from src.llm import generate_answer
            from src.retriever import retrieve

            qa_set = json.loads(Path(eval_path).read_text())
            rows = []
            with st.spinner(f"Running {len(qa_set)} evaluation question(s)..."):
                for item in qa_set:
                    retrieved = retrieve(item["question"], st.session_state.pipeline.store)
                    retrieved_ids = [rc.chunk.id for rc in retrieved]
                    relevant_ids = item["relevant_chunk_ids"]

                    generated = generate_answer(item["question"], retrieved)
                    claims = parse_and_verify(generated.text, retrieved)

                    rows.append({
                        "question": item["question"],
                        "recall": recall_at_k(retrieved_ids, relevant_ids),
                        "precision": precision_at_k(retrieved_ids, relevant_ids),
                        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                        "ndcg": ndcg_at_k(retrieved_ids, relevant_ids),
                        "keyword_f1": keyword_overlap_f1(generated.text, item.get("reference_answer", "")),
                        "faithfulness": faithfulness_score(claims),
                    })

            st.subheader("Aggregate metrics")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Recall@k", f"{statistics.mean(r['recall'] for r in rows):.2f}")
            c2.metric("Precision@k", f"{statistics.mean(r['precision'] for r in rows):.2f}")
            c3.metric("MRR", f"{statistics.mean(r['mrr'] for r in rows):.2f}")
            c4.metric("nDCG@k", f"{statistics.mean(r['ndcg'] for r in rows):.2f}")
            c5.metric("Keyword F1", f"{statistics.mean(r['keyword_f1'] for r in rows):.2f}")
            c6.metric("Faithfulness", f"{statistics.mean(r['faithfulness'] for r in rows):.2f}")

            st.subheader("Per-question breakdown")
            st.dataframe(rows, use_container_width=True)
