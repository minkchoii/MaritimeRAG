"""Evaluate retrieval Recall@k against gold questions (JSONL)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_eval_lib import run_retrieval_eval, run_unified_retrieval_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval Recall@k against gold questions (JSONL)."
    )
    parser.add_argument("--doc-id", type=str, default=None, help="Single-document index")
    parser.add_argument(
        "--unified",
        type=str,
        default=None,
        help="Unified collection id (e.g. full_corpus, pilot_100)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Gold questions JSONL",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embedding-preset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/logs"))
    args = parser.parse_args()

    if bool(args.doc_id) == bool(args.unified):
        parser.error("Provide exactly one of: --doc-id OR --unified")

    if args.questions:
        questions_path = args.questions
    elif args.unified == "full_corpus":
        questions_path = Path("data/eval/full_corpus_questions.jsonl")
    elif args.unified:
        questions_path = Path("data/eval") / f"unified_{args.unified}_questions.jsonl"
    else:
        questions_path = Path("data/eval") / f"{args.doc_id}_questions.jsonl"

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    if args.unified:
        summary, results, json_path, txt_path = run_unified_retrieval_eval(
            unified_id=args.unified,
            questions_path=questions_path,
            top_k=args.top_k,
            index_dir=args.index_dir,
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            embedding_preset=args.embedding_preset,
        )
    else:
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
    if summary.get("by_category"):
        print("By category:")
        for cat, stats in summary["by_category"].items():
            print(f"  {cat}: {stats['recall_at_k']:.1%} ({stats['hits']}/{stats['questions']})")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")


if __name__ == "__main__":
    main()
