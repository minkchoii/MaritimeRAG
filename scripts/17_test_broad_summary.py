"""Test broad multi-document summary pipeline (search + optional LLM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import load_unified_collection, run_retrieval_only, run_rag_pipeline

TESTS = [
    {
        "id": "T1",
        "question_id": "T1",
        "question": "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.",
        "retrieval_sources": ["MEPC"],
        "category": "trend_summary",
    },
    {
        "id": "T2",
        "question_id": "T2",
        "question": "최신 MEPC 회의에서 선박 운항 및 규제 보고에 직접 영향을 주는 사항을 정리해줘.",
        "retrieval_sources": ["MEPC"],
        "category": "env_regulation",
    },
    {
        "id": "T3",
        "question_id": "T3",
        "question": "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.",
        "retrieval_sources": ["MSC"],
        "category": "autonomous",
    },
]


def _bullet_stats(answer: str) -> dict:
    bullets = [ln for ln in answer.splitlines() if ln.strip().startswith("- ")]
    sec1 = answer.split("## 2)")[0] if "## 2)" in answer else answer
    sec1_bullets = [ln for ln in sec1.splitlines() if ln.strip().startswith("- ")]
    short = 0
    for b in bullets:
        sents = [x for x in b.replace("- ", "").split(".") if x.strip()]
        if len(sents) < 2 and len(b) < 120:
            short += 1
    return {
        "total_bullets": len(bullets),
        "sec1_bullets": len(sec1_bullets),
        "short_bullets": short,
        "chars": len(answer),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Run full multi-doc LLM (slow)")
    parser.add_argument("--test-id", default="", help="T1|T2|T3 only")
    args = parser.parse_args()

    collection, embed_model, _ = load_unified_collection("full_corpus", _ROOT / "data/processed/index")
    out_dir = _ROOT / "data/processed/logs/pilot_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    tests = TESTS
    if args.test_id:
        tests = [t for t in TESTS if t["id"] == args.test_id]

    for row in tests:
        print(f"\n=== {row['id']}: {row['question'][:50]}...", flush=True)
        if args.llm:
            r = run_rag_pipeline(
                row,
                collection,
                embed_model,
                chunks_dir=_ROOT / "data/processed/chunks",
                top_k=10,
                fetch_k=120,
                max_chunks_per_doc=3,
                max_docs=15,
                gold_doc_filter=False,
            )
            stats = _bullet_stats(r.get("answer", ""))
            path = out_dir / f"broad_test_{row['id']}_answer.md"
            path.write_text(r.get("answer", ""), encoding="utf-8")
            print(f"  mode={r.get('retrieval_config', {}).get('answer_mode')} docs={stats} -> {path}")
        else:
            r = run_retrieval_only(
                row,
                collection,
                embed_model,
                chunks_dir=_ROOT / "data/processed/chunks",
                top_k=10,
                fetch_k=120,
                max_chunks_per_doc=3,
                max_docs=15,
            )
            print(
                f"  mode={r.get('answer_mode')} category={r.get('question_category_label')} "
                f"docs={len(r.get('doc_groups', []))} chunks={len(r['retrieved'])}"
            )
        report.append(
            {
                "id": row["id"],
                "answer_mode": r.get("answer_mode") or r.get("retrieval_config", {}).get("answer_mode"),
                "question_category": r.get("question_category"),
                "doc_count": len(r.get("doc_groups", [])),
                "chunk_count": len(r.get("retrieved", [])),
                "warnings": r.get("pipeline_warnings", []),
            }
        )

    (out_dir / "broad_test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nReport:", out_dir / "broad_test_report.json")


if __name__ == "__main__":
    main()
