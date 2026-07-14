"""Compare UI Fast path vs benchmark Fast path (V06, llama3.1:8b)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import DEFAULT_OLLAMA_BASE
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from ollama_warmup import (
    bootstrap_all_resources,
    ensure_fast_warm,
    ensure_fast_warm_checked,
    get_warm_preflight_snapshot,
)
from retrieval_timing import TimingTrace, finalize_ui_timing

MODEL = "llama3.1:8b"
BASE = DEFAULT_OLLAMA_BASE
QID = "V06"


def _load_row(qid: str) -> dict:
    path = Path("data/eval/pilot_validation_questions.jsonl")
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("question_id") == qid:
            return row
    raise KeyError(qid)


def run_ui_path(row, coll, emb, manifest) -> tuple[dict, dict, dict]:
    preflight = get_warm_preflight_snapshot(MODEL)
    if preflight.get("rewarm_needed_before_click"):
        ensure_fast_warm(MODEL, BASE)
    else:
        ensure_fast_warm_checked(MODEL, BASE, allow_rewarm=False)
    ts = time.time()
    timing = TimingTrace()
    timing.run_id = "ui_sim"
    timing.set_user_click(ts)
    timing.set_cache("llm_server_ready", True)
    timing.meta["latency_mode"] = "fast"
    out = run_full_inprocess(
        row,
        user_submit_ts=ts,
        collection=coll,
        embed_model=emb,
        manifest=manifest,
        llm_model=MODEL,
        ollama_base=BASE,
        latency_mode="fast",
        start_type="warm",
        use_rerank=False,
        top_k=3,
        fetch_k=10,
        max_doc=1,
        max_docs=2,
        auto_llm_warm=False,
        skip_ollama_probe=True,
        timing=timing,
    )
    final = finalize_ui_timing(out.get("timing_log"), ts) or out.get("timing_log") or {}
    return preflight, final.get("timing_metrics") or {}, final


def run_bench_path(row, coll, emb, manifest) -> dict:
    ensure_fast_warm(MODEL, BASE)
    ts = time.time()
    out = run_full_inprocess(
        row,
        user_submit_ts=ts,
        collection=coll,
        embed_model=emb,
        manifest=manifest,
        llm_model=MODEL,
        ollama_base=BASE,
        latency_mode="fast",
        start_type="warm",
        use_rerank=False,
        top_k=3,
        fetch_k=10,
        max_doc=1,
        max_docs=2,
        auto_llm_warm=True,
    )
    final = finalize_ui_timing(out.get("timing_log"), ts) or out.get("timing_log") or {}
    return final.get("timing_metrics") or {}


def main() -> None:
    row = _load_row(QID)
    boot = bootstrap_all_resources(
        MODEL, BASE, DEFAULT_UNIFIED, DEFAULT_INDEX_DIR, force_llm_warm=True
    )
    coll, emb, manifest = boot["collection"], boot["embed_model"], boot["manifest"]

    print(f"=== {QID} {MODEL} — 3x UI path repeat ===")
    for i in range(3):
        preflight, ui_m, ui_final = run_ui_path(row, coll, emb, manifest)
        print(
            f"  run{i + 1}: ret={ui_m.get('retrieval_total_time')} "
            f"pre_llm={ui_m.get('pre_llm_latency')} "
            f"llm_ttft={ui_m.get('llm_ttft')} "
            f"e2e={ui_m.get('end_to_end_ttft')} "
            f"load_ms={ui_m.get('ollama_load_duration_ms')}"
        )

    preflight, ui_m, ui_final = run_ui_path(row, coll, emb, manifest)
    bench_m = run_bench_path(row, coll, emb, manifest)

    keys = [
        "retrieval_total_time",
        "pre_llm_latency",
        "ensure_fast_warm_check_time",
        "rewarm_time",
        "ollama_probe_time",
        "prompt_build_time",
        "stream_init_time",
        "llm_ttft",
        "answer_first_visible_latency",
        "end_to_end_ttft",
    ]
    print(f"=== {QID} {MODEL} path comparison ===")
    print("UI path (auto_llm_warm=False, skip_ollama_probe=True):")
    for k in keys:
        print(f"  {k}: {ui_m.get(k)}")
    print(f"  rewarm_triggered: {ui_final.get('rewarm_triggered')}")
    print(f"  rewarm_reason: {ui_final.get('rewarm_reason')}")
    print(f"  preflight valid: {preflight.get('warm_state_valid_before_click')}")
    print(f"  ollama_load_ms: {ui_m.get('ollama_load_duration_ms')}")
    print(f"  pure_inference_ttft: {ui_m.get('pure_inference_ttft')}")
    print("Benchmark path (auto_llm_warm=True):")
    for k in keys:
        print(f"  {k}: {bench_m.get(k)}")


if __name__ == "__main__":
    main()
