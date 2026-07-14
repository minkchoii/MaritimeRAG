"""Batch LLM validation for pilot 7 questions."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import load_questions, load_unified_collection, run_rag_pipeline


def main() -> None:
    questions = load_questions(_ROOT / "data/eval/pilot_validation_questions.jsonl")
    collection, embed_model, _ = load_unified_collection("full_corpus", _ROOT / "data/processed/index")
    out_dir = _ROOT / "data/processed/logs/pilot_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Pilot LLM Validation Answers",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "Model: llama3.1:8b | rerank=balanced | top_k=8",
        "",
    ]

    for row in questions:
        qid = row["question_id"]
        print(f"Running {qid}...", flush=True)
        r = run_rag_pipeline(
            row,
            collection,
            embed_model,
            chunks_dir=_ROOT / "data/processed/chunks",
            model="llama3.1:8b",
        )
        m = r["retrieval_metrics"]
        lines.extend(
            [
                f"## {qid} [{row['category']}]",
                "",
                f"**질문:** {row['question']}",
                "",
                f"- gold_doc@5: {'YES' if m.get('gold_doc_hit_at_5') else 'NO'}",
                f"- gold_page@5: {'YES' if m.get('gold_page_set_hit_at_5') else 'NO'}",
                f"- keyword_coverage: {m.get('keyword_coverage', 0):.0%}",
                f"- provider: {r.get('llm_provider')}",
                "",
                r.get("answer") or "*(no answer)*",
                "",
                "---",
                "",
            ]
        )

    path = out_dir / "pilot_validation_answers_llm.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
