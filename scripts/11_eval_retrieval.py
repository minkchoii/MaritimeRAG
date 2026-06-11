"""Evaluate retrieval Recall@k against gold questions (JSONL)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_eval_lib import run_retrieval_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval Recall@k against gold questions (JSONL)."
    )
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Gold questions JSONL (default: data/eval/<doc_id>_questions.jsonl)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embedding-preset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/logs"))
    args = parser.parse_args()

    questions_path = args.questions or Path("data/eval") / f"{args.doc_id}_questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    summary, results, json_path, txt_path = run_retrieval_eval(
        doc_id=args.doc_id,
        questions_path=questions_path,
        top_k=args.top_k,
        index_dir=args.index_dir,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        embedding_preset=args.embedding_preset,
    )

    print(f"Evaluated {summary['num_questions']} questions (top_k={args.top_k})")
    print(
        f"Recall@{args.top_k}: {summary['recall_at_k']:.1%} "
        f"({summary['hits_at_k']}/{summary['num_questions']})"
    )
    print(f"Page hit@{args.top_k}: {summary['page_recall_at_k']:.1%}")
    print(f"Gold pages: p{summary['gold_page_min']} – p{summary['gold_page_max']}")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")


if __name__ == "__main__":
    main()
