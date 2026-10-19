"""
CLI evaluation harness.

Usage:
    python -m eval.run_eval --docs data/sample_docs --qa eval/eval_dataset.json
    python -m eval.run_eval --docs data/sample_docs --qa eval/eval_dataset.json --skip-generation

--skip-generation runs retrieval-only evaluation (no LLM calls needed at
all) -- useful for CI or for tuning chunk size / top_k / MMR without
burning API credits.
"""

import argparse
import json
import statistics
from pathlib import Path

from eval.metrics import (
    keyword_overlap_f1,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from src.citations import faithfulness_score, parse_and_verify
from src.llm import generate_answer
from src.rag_pipeline import RagPipeline
from src.retriever import retrieve


def load_qa_set(path: Path):
    return json.loads(path.read_text())


def run(docs_dir: Path, qa_path: Path, skip_generation: bool, use_llm_judge: bool):
    file_paths = sorted(
        [p for p in docs_dir.iterdir() if p.suffix.lower() in (".pdf", ".txt", ".md")]
    )
    if not file_paths:
        raise SystemExit(f"No .pdf/.txt/.md files found in {docs_dir}")

    print(f"Building index over {len(file_paths)} document(s): "
          f"{', '.join(p.name for p in file_paths)}")
    pipeline = RagPipeline()
    n_chunks = pipeline.build_index(file_paths)
    print(f"Indexed {n_chunks} chunks.\n")

    qa_set = load_qa_set(qa_path)

    recalls, precisions, mrrs, ndcgs = [], [], [], []
    keyword_f1s, faithfulness_scores, judge_scores = [], [], []

    for item in qa_set:
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]
        reference_answer = item.get("reference_answer", "")

        retrieved = retrieve(question, pipeline.store)
        retrieved_ids = [rc.chunk.id for rc in retrieved]

        recalls.append(recall_at_k(retrieved_ids, relevant_ids))
        precisions.append(precision_at_k(retrieved_ids, relevant_ids))
        mrrs.append(reciprocal_rank(retrieved_ids, relevant_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids))

        if not skip_generation:
            generated = generate_answer(question, retrieved)
            claims = parse_and_verify(generated.text, retrieved)
            faithfulness_scores.append(faithfulness_score(claims))

            if reference_answer:
                keyword_f1s.append(keyword_overlap_f1(generated.text, reference_answer))

                if use_llm_judge:
                    try:
                        from eval.metrics import llm_judge_score
                        judge_scores.append(
                            llm_judge_score(question, generated.text, reference_answer)
                        )
                    except Exception as e:
                        print(f"  [llm-judge skipped for '{question[:40]}...': {e}]")

    def avg(xs):
        return round(statistics.mean(xs), 3) if xs else float("nan")

    from src import retriever as retriever_module
    print(f"=== Retrieval quality (k={retriever_module.config.TOP_K}) ===")
    print(f"Recall@k:     {avg(recalls)}")
    print(f"Precision@k:  {avg(precisions)}")
    print(f"MRR:          {avg(mrrs)}")
    print(f"nDCG@k:       {avg(ndcgs)}")

    if not skip_generation:
        print("\n=== Answer quality ===")
        print(f"Keyword-overlap F1 (avg):     {avg(keyword_f1s)}")
        print(f"Citation faithfulness (avg):  {avg(faithfulness_scores)}"
              "   (fraction of cited claims supported by their source chunk)")
        if use_llm_judge:
            if judge_scores:
                print(f"LLM-judge correctness (avg):  {avg(judge_scores)} / 5")
            else:
                print("LLM-judge correctness (avg):  skipped (no API key or all calls failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a labeled QA set.")
    parser.add_argument("--docs", type=Path, required=True, help="Directory of source documents to index")
    parser.add_argument("--qa", type=Path, required=True, help="Path to labeled QA JSON file")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Only evaluate retrieval; skip LLM answer generation entirely")
    parser.add_argument("--llm-judge", action="store_true",
                         help="Also score answer correctness with an LLM judge (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    run(args.docs, args.qa, args.skip_generation, args.llm_judge)
