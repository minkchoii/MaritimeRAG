"""
Compare dense-only vs dense+BM25+RRF for Rule lookup V06/V07.

  python scripts/34_rule_hybrid_benchmark.py
  python scripts/34_rule_hybrid_benchmark.py --unified full_corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import (
    RetrievedChunk,
    load_questions,
    load_unified_collection,
    retrieve_for_question,
)
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED
from rule_lookup_retrieval_log import save_rule_lookup_run_log
from rule_lookup_structured_answer import build_rule_lookup_structured_answer


def _top_file_names(chunks: list[RetrievedChunk], k: int = 5) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        fn = str(c.file_name or "")
        if fn and fn not in seen:
            seen.append(fn)
        if len(seen) >= k:
            break
    return seen


def _has_target(files: list[str], needles: list[str]) -> bool:
    blob = " ".join(files).lower()
    return any(n.lower() in blob for n in needles)


def _catalog_as_confirmed(answer: str) -> bool:
    if "확정 여부: 확정" not in answer:
        return False
    return "catalog" in answer.lower() and "Candidate" not in answer


def run_variant(
    row: dict,
    collection,
    embed_model: str,
    *,
    unified_id: str,
    index_dir: Path,
    use_hybrid: bool,
    chunks_dir: Path,
    fetch_k: int = 56,
    top_k: int = 8,
) -> dict:
    r = dict(row)
    r["retrieval_variant"] = "hybrid_rrf" if use_hybrid else "dense_only"
    r["_use_hybrid_bm25"] = use_hybrid
    r.pop("_hybrid_retrieval_log", None)

    pool = retrieve_for_question(
        collection,
        embed_model,
        r,
        top_k=fetch_k,
        fetch_k=fetch_k,
        chunks_dir=chunks_dir,
        unified_id=unified_id,
        index_dir=index_dir,
    )
    retrieved = pool[:top_k]
    warnings = list(r.get("warning_flags") or [])
    if r.get("_hybrid_retrieval_log"):
        warnings.extend(r["_hybrid_retrieval_log"].get("warning_flags") or [])

    answer, ans_warnings = build_rule_lookup_structured_answer(
        retrieved, question=str(row["question"]), warning_flags=warnings
    )
    warnings.extend(ans_warnings)

    log = r.get("_hybrid_retrieval_log") or {}
    save_rule_lookup_run_log(
        question=str(row["question"]),
        row=r,
        category=str(row.get("category") or "rule_lookup"),
        dense_results=log.get("dense_results") or [],
        bm25_results=log.get("bm25_results") or [],
        fused_results=log.get("fused_results") or [],
        retrieved=retrieved,
        answer=answer,
        warning_flags=warnings,
        tag="hybrid" if use_hybrid else "dense_only",
    )

    top5_files = _top_file_names(pool, 5)
    return {
        "question_id": row.get("question_id"),
        "variant": r["retrieval_variant"],
        "top5_file_names": top5_files,
        "warning_flags": list(dict.fromkeys(warnings)),
        "answer_preview": answer[:600],
        "doc_name_mismatch_count": sum(1 for w in warnings if "doc_name_mismatch" in w),
        "source_filter_fallback": "source_filter_fallback" in warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/eval/pilot_validation_questions.jsonl"))
    parser.add_argument("--unified", type=str, default=DEFAULT_UNIFIED)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/rule_lookup_hybrid/benchmark_summary.json"))
    args = parser.parse_args()

    rows = [q for q in load_questions(args.questions) if q.get("question_id") in {"V06", "V07"}]
    if not rows:
        raise SystemExit("V06/V07 not found in questions file")

    collection, embed_model, _ = load_unified_collection(args.unified, args.index_dir)
    results: list[dict] = []
    for row in rows:
        dense = run_variant(
            row, collection, embed_model,
            unified_id=args.unified, index_dir=args.index_dir,
            use_hybrid=False, chunks_dir=args.chunks_dir,
        )
        hybrid = run_variant(
            row, collection, embed_model,
            unified_id=args.unified, index_dir=args.index_dir,
            use_hybrid=True, chunks_dir=args.chunks_dir,
        )
        qid = row["question_id"]
        needles = ["DNV-CG-0264"] if qid == "V06" else ["notice", "lr", "low-flashpoint", "section 15"]
        comparison = {
            "question_id": qid,
            "question": row["question"],
            "dense_only": dense,
            "hybrid_rrf": hybrid,
            "checks": {
                "target_in_top5_dense": _has_target(dense["top5_file_names"], needles),
                "target_in_top5_hybrid": _has_target(hybrid["top5_file_names"], needles),
                "catalog_as_confirmed_dense": _catalog_as_confirmed(dense["answer_preview"]),
                "catalog_as_confirmed_hybrid": _catalog_as_confirmed(hybrid["answer_preview"]),
                "doc_name_mismatch_dense": dense["doc_name_mismatch_count"],
                "doc_name_mismatch_hybrid": hybrid["doc_name_mismatch_count"],
            },
        }
        results.append(comparison)
        print(json.dumps(comparison, ensure_ascii=True, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unified_id": args.unified,
        "comparisons": results,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
