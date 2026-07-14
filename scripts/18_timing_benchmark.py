"""Run end-to-end latency benchmark with cold/warm cache separation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from rag_resource_cache import clear_process_caches, warm_all_resources
from retrieval_timing import DEFAULT_TIMING_LOG, append_timing_log, finalize_ui_timing

PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
BENCHMARK_QIDS = ("V01", "V05", "V06")

# Subprocess cold-load baseline (pre-caching, from prior benchmark)
BASELINE_BEFORE = {
    "V01": {
        "query_embedding_time": 19.77,
        "vector_search_time": 0.79,
        "retrieval_total_time": 21.57,
        "llm_ttft": 71.78,
        "llm_generation_time": 334.76,
        "end_to_end_ttft": 93.48,
        "end_to_end_total": 359.99,
    },
    "V05": {
        "query_embedding_time": 20.08,
        "vector_search_time": 0.81,
        "retrieval_total_time": 21.96,
        "llm_ttft": 68.33,
        "llm_generation_time": 232.78,
        "end_to_end_ttft": 90.44,
        "end_to_end_total": 258.40,
    },
    "V06": {
        "query_embedding_time": 19.85,
        "vector_search_time": 0.87,
        "retrieval_total_time": 21.75,
        "llm_ttft": 49.18,
        "llm_generation_time": 156.15,
        "end_to_end_ttft": 71.09,
        "end_to_end_total": 181.50,
    },
}

COMPARE_METRICS = (
    "query_embedding_time",
    "vector_search_time",
    "retrieval_total_time",
    "llm_ttft",
    "llm_generation_time",
    "end_to_end_ttft",
    "end_to_end_total",
)


def _load_questions(path: Path) -> dict[str, dict]:
    from rag_eval_lib import load_questions

    return {r["question_id"]: r for r in load_questions(path)}


def run_question(
    row: dict,
    *,
    start_type: str,
    run_index: int,
    top_k: int,
    fetch_k: int,
    max_doc: int,
    max_docs: int,
    use_rerank: bool,
    llm_model: str,
    ollama_base: str,
    multi_doc_strategy: str,
    max_llm_docs: int,
    skip_llm: bool,
    collection=None,
    embed_model: str | None = None,
    manifest: dict | None = None,
    log: bool,
) -> dict:
    user_submit_ts = time.time()
    out = run_full_inprocess(
        row,
        top_k=top_k,
        fetch_k=fetch_k,
        max_doc=max_doc,
        max_docs=max_docs,
        use_rerank=use_rerank,
        llm_model=llm_model,
        ollama_base=ollama_base,
        temperature=0.15,
        multi_doc_strategy=multi_doc_strategy,
        max_llm_docs=max_llm_docs,
        user_submit_ts=user_submit_ts,
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        start_type=start_type,
        run_index=run_index,
        skip_llm=skip_llm,
    )
    final = finalize_ui_timing(out.get("timing_log"), user_submit_ts)
    if final:
        final["start_type"] = start_type
        final["run_index"] = run_index
        if log:
            append_timing_log(final, DEFAULT_TIMING_LOG)
    metrics = (final or out).get("timing_metrics") or {}
    log_row = final or out.get("timing_log") or {}
    search_out = out.get("search_out") or {}
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "answer_mode": out.get("answer_mode"),
        "start_type": start_type,
        "run_index": run_index,
        "timing_metrics": metrics,
        "timing_log": log_row,
        "_embed_model": search_out.get("embed_model"),
    }


def _fmt(val: float | None) -> str:
    return "—" if val is None else f"{val:.2f}s"


def print_result(result: dict) -> None:
    m = result["timing_metrics"]
    print(f"\n{'=' * 72}")
    print(
        f"[{result['question_id']}] {result.get('start_type')} run#{result.get('run_index')} — "
        f"{result['question']}"
    )
    print(f"  mode: {result.get('answer_mode', '—')}")
    for key in COMPARE_METRICS:
        print(f"  {key:22s} {_fmt(m.get(key))}")
    cache = result.get("timing_log", {}).get("cache_status") or {}
    if cache:
        print("  --- cache_status ---")
        for k, v in cache.items():
            print(f"  {k}: {v}")


def print_comparison(warm_results: list[dict]) -> None:
    by_qid: dict[str, dict] = {}
    for r in warm_results:
        by_qid[r["question_id"]] = r["timing_metrics"]

    print(f"\n{'=' * 72}")
    print("Before vs After (warm) comparison")
    print(f"| {'QID':<4} | {'metric':<22} | {'before':>8} | {'after_warm':>10} | {'improvement':>11} |")
    print(f"|{'-' * 6}|{'-' * 24}|{'-' * 10}|{'-' * 12}|{'-' * 13}|")
    for qid in BENCHMARK_QIDS:
        before = BASELINE_BEFORE.get(qid, {})
        after = by_qid.get(qid, {})
        for metric in COMPARE_METRICS:
            b = before.get(metric)
            a = after.get(metric)
            if b is None or a is None:
                imp = "—"
            else:
                imp = f"{a - b:+.2f}s"
            print(
                f"| {qid:<4} | {metric:<22} | {_fmt(b):>8} | {_fmt(a):>10} | {imp:>11} |"
            )


def run_suite(
    rows: dict[str, dict],
    qids: list[str],
    *,
    start_type: str,
    run_index_start: int,
    args: argparse.Namespace,
    collection=None,
    embed_model: str | None = None,
    manifest: dict | None = None,
) -> tuple[list[dict], object, str | None, dict | None]:
    results = []
    coll, emb, man = collection, embed_model, manifest
    idx = run_index_start
    for qid in qids:
        if start_type == "cold":
            clear_process_caches()
            coll, emb, man = None, None, None
        row = rows[qid]
        result = run_question(
            row,
            start_type=start_type,
            run_index=idx,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            max_doc=args.max_doc,
            max_docs=args.max_docs,
            use_rerank=not args.no_rerank,
            llm_model=args.llm_model,
            ollama_base=args.ollama_base,
            multi_doc_strategy=args.multi_doc_strategy,
            max_llm_docs=args.max_llm_docs,
            skip_llm=args.retrieval_only,
            collection=coll,
            embed_model=emb,
            manifest=man,
            log=not args.no_log,
        )
        print_result(result)
        results.append(result)
        if start_type == "warm" and result.get("_embed_model"):
            emb = result["_embed_model"]
        idx += 1
    return results, coll, emb, man


def main() -> int:
    parser = argparse.ArgumentParser(description="MaritimeRAG cold/warm timing benchmark")
    parser.add_argument("--questions", default=str(PILOT_QUESTIONS))
    parser.add_argument("--qids", default=",".join(BENCHMARK_QIDS))
    parser.add_argument("--mode", choices=["cold", "warm", "both"], default="both")
    parser.add_argument("--repeat", type=int, default=3, help="Warm repeat count (full suite)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fetch-k", type=int, default=120)
    parser.add_argument("--max-doc", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--multi-doc-strategy", default="single_pass")
    parser.add_argument("--max-llm-docs", type=int, default=6)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM (retrieval latency only)")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    rows = _load_questions(Path(args.questions))
    missing = [q for q in qids if q not in rows]
    if missing:
        print(f"Missing question ids: {missing}", file=sys.stderr)
        return 1

    print(f"Benchmark mode={args.mode} qids={qids} retrieval_only={args.retrieval_only}")
    print(f"Log: {DEFAULT_TIMING_LOG}")

    cold_results: list[dict] = []
    warm_results: list[dict] = []
    collection = embed_model = manifest = None

    if args.mode in ("cold", "both"):
        print("\n### COLD START (clear_process_caches before each question)")
        cold_results, collection, embed_model, manifest = run_suite(
            rows,
            qids,
            start_type="cold",
            run_index_start=1,
            args=args,
        )

    if args.mode in ("warm", "both"):
        if args.mode == "warm":
            clear_process_caches()
            print("\n### WARM-UP (load vector DB + embedding model)")
            collection, embed_model, manifest = warm_all_resources(
                DEFAULT_UNIFIED, DEFAULT_INDEX_DIR
            )
        elif collection is None:
            collection, embed_model, manifest = warm_all_resources(
                DEFAULT_UNIFIED, DEFAULT_INDEX_DIR
            )

        repeats = max(1, args.repeat)
        for rep in range(repeats):
            print(f"\n### WARM START suite repeat {rep + 1}/{repeats}")
            batch, collection, embed_model, manifest = run_suite(
                rows,
                qids,
                start_type="warm",
                run_index_start=rep * len(qids) + 1,
                args=args,
                collection=collection,
                embed_model=embed_model,
                manifest=manifest,
            )
            if rep == repeats - 1:
                warm_results = batch
            else:
                warm_results.extend(batch)

    if warm_results:
        print_comparison(warm_results[-len(qids):] if len(warm_results) > len(qids) else warm_results)

    targets = {
        "query_embedding_time": 1.0,
        "retrieval_total_time": 2.0,
        "vector_search_time": 1.5,
    }
    print(f"\n{'=' * 72}")
    print("Warm-run target check (last warm suite)")
    last_warm = warm_results[-len(qids):] if warm_results else []
    for r in last_warm:
        m = r["timing_metrics"]
        qid = r["question_id"]
        for metric, limit in targets.items():
            val = m.get(metric)
            ok = val is not None and val <= limit
            mark = "OK" if ok else "MISS"
            print(f"  [{qid}] {metric}: {_fmt(val)} (target <= {limit:.1f}s) {mark}")

    out_path = _ROOT / "data/processed/logs/timing_benchmark_cold_warm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"cold": cold_results, "warm": warm_results, "baseline_before": BASELINE_BEFORE},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
