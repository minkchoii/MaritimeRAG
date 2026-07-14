"""Accurate mode initial response + streaming benchmark (initial_ack <= 3s)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ollama_env_probe import snapshot_ollama_env
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from retrieval_timing import finalize_ui_timing

PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
QIDS = ("V01", "V05", "V06")
DEFAULT_MODEL = "llama3.1:8b"
REPEATS = 3

ACCURATE_DEFAULTS = {
    "top_k": 10,
    "fetch_k": 120,
    "max_doc": 3,
    "max_docs": 10,
    "use_rerank": True,
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(path: Path) -> dict[str, dict]:
    from rag_eval_lib import load_questions

    return {r["question_id"]: r for r in load_questions(path)}


def _pass_initial(lat: float | None) -> str:
    if lat is None:
        return "—"
    return "PASS" if lat <= 3.0 else "FAIL"


def run_accurate_once(
    row: dict,
    model: str,
    ollama_base: str,
    collection,
    embed_model,
    manifest,
    *,
    run_index: int,
) -> dict[str, Any]:
    user_submit_ts = time.time()
    streamed: list[str] = []

    def on_token(tok: str) -> None:
        streamed.append(tok)

    out = run_full_inprocess(
        row,
        user_submit_ts=user_submit_ts,
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        llm_model=model,
        ollama_base=ollama_base,
        latency_mode="accurate",
        start_type="warm",
        auto_llm_warm=False,
        mark_initial_ack=True,
        on_token=on_token,
        **ACCURATE_DEFAULTS,
    )
    final = finalize_ui_timing(out.get("timing_log"), user_submit_ts) or out.get("timing_log") or {}
    m = final.get("timing_metrics") or {}
    env = snapshot_ollama_env(model, ollama_base)
    return {
        "timestamp": _utc_iso(),
        "run_index": run_index,
        "qid": row["question_id"],
        "model_name": model,
        "mode": "accurate",
        "gpu_detected": env.get("gpu_detected"),
        "processor": env.get("ollama_processor"),
        "initial_ack_latency": m.get("initial_ack_latency"),
        "retrieval_time": m.get("retrieval_time") or m.get("retrieval_total_time"),
        "accurate_llm_ttft": m.get("accurate_llm_ttft") or m.get("llm_ttft"),
        "first_llm_visible_latency": m.get("accurate_first_visible_llm_latency"),
        "full_report_total": m.get("accurate_full_report_total_time"),
        "accurate_total_time": m.get("accurate_total_time") or m.get("end_to_end_total"),
        "initial_3s_pass": m.get("accurate_initial_response_3s_pass"),
        "first_llm_visible_3s_pass": m.get("accurate_first_llm_visible_3s_pass"),
        "judgement_initial_3s": _pass_initial(m.get("initial_ack_latency")),
        "judgement_first_llm_3s": _pass_initial(m.get("accurate_first_visible_llm_latency")),
        "timing_metrics": m,
        "timing_log": final,
        "streamed_chars": len("".join(streamed)),
        "pass_3s": m.get("accurate_initial_response_3s_pass"),
    }


def summarize(rows: list[dict]) -> list[dict]:
    by_qid: dict[str, list[dict]] = {}
    for r in rows:
        by_qid.setdefault(r["qid"], []).append(r)
    out = []
    for qid, rs in sorted(by_qid.items()):
        init_lats = [x["initial_ack_latency"] for x in rs if x.get("initial_ack_latency") is not None]
        out.append(
            {
                "qid": qid,
                "runs": len(rs),
                "avg_initial_ack": round(statistics.mean(init_lats), 4) if init_lats else None,
                "pass_count": sum(1 for x in rs if x.get("initial_3s_pass")),
                "fail_count": sum(1 for x in rs if not x.get("initial_3s_pass")),
            }
        )
    return out


def print_table(rows: list[dict]) -> None:
    print(
        "\n| qid | run | initial_ack_latency | retrieval_time | accurate_llm_ttft | "
        "first_llm_visible_latency | full_report_total | initial_3s_pass |"
    )
    print("| --- | --: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        print(
            f"| {r['qid']} | {r['run_index']} | {r.get('initial_ack_latency')} | "
            f"{r.get('retrieval_time')} | {r.get('accurate_llm_ttft')} | "
            f"{r.get('first_llm_visible_latency')} | {r.get('full_report_total')} | "
            f"{r.get('judgement_initial_3s')} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Accurate streaming initial response benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--qids", default=",".join(QIDS))
    args = parser.parse_args()

    from ollama_warmup import bootstrap_all_resources

    qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    rows_map = _load_rows(PILOT_QUESTIONS)
    print(f"\n========== Bootstrap {args.model} ==========")
    boot = bootstrap_all_resources(
        args.model, args.ollama_base, DEFAULT_UNIFIED, DEFAULT_INDEX_DIR
    )
    collection = boot["collection"]
    embed_model = boot["embed_model"]
    manifest = boot["manifest"]
    env = snapshot_ollama_env(args.model, args.ollama_base)
    print(json.dumps({k: env[k] for k in ("ollama_processor", "gpu_detected", "gpu_name")}, indent=2))

    all_runs: list[dict] = []
    for qid in qids:
        row = rows_map[qid]
        for i in range(1, args.repeats + 1):
            print(f"  [{args.model}] {qid} run={i} …")
            result = run_accurate_once(
                row,
                args.model,
                args.ollama_base,
                collection,
                embed_model,
                manifest,
                run_index=i,
            )
            all_runs.append(result)
            print(
                f"    initial_ack={result.get('initial_ack_latency')}s "
                f"llm_ttft={result.get('accurate_llm_ttft')}s "
                f"first_visible={result.get('first_llm_visible_latency')}s "
                f"full={result.get('full_report_total')}s "
                f"{result.get('judgement_initial_3s')}"
            )

    print_table(all_runs)
    summary = summarize(all_runs)
    out_path = _ROOT / "data/processed/logs/accurate_streaming_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_type": "accurate_streaming_response",
        "model": args.model,
        "repeats": args.repeats,
        "qids": qids,
        "runs": all_runs,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
