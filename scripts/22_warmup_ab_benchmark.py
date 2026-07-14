"""A/B benchmark: legacy vs matched Fast LLM warm-up (V01/V05/V06)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ollama_warmup import get_warm_status_display, legacy_warmup_generate, warmup_fast_chat
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from rag_resource_cache import warm_all_resources
from retrieval_timing import finalize_ui_timing

PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
QIDS = ("V01", "V05", "V06")
MODELS = ("llama3.2:latest", "llama3.1:8b")


def _load_rows(path: Path) -> dict[str, dict]:
    from rag_eval_lib import load_questions

    return {r["question_id"]: r for r in load_questions(path)}


def _judge_3s(e2e: float | None) -> str:
    if e2e is None:
        return "—"
    if e2e <= 3.0:
        return "PASS"
    if e2e <= 5.0:
        return "NEAR"
    return "FAIL"


def run_fast(row: dict, model: str, ollama_base: str, collection, embed_model, manifest, *, auto_llm_warm: bool) -> dict:
    import time

    user_submit_ts = time.time()
    out = run_full_inprocess(
        row,
        user_submit_ts=user_submit_ts,
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        llm_model=model,
        ollama_base=ollama_base,
        latency_mode="fast",
        start_type="warm",
        use_rerank=False,
        top_k=3,
        fetch_k=10,
        max_doc=1,
        max_docs=2,
        auto_llm_warm=auto_llm_warm,
    )
    final = finalize_ui_timing(out.get("timing_log"), user_submit_ts) or out.get("timing_log") or {}
    m = final.get("timing_metrics") or {}
    warm = get_warm_status_display()
    return {
        "qid": row["question_id"],
        "model": model,
        "warmup_matched": m.get("warmup_matched_request"),
        "context_changed": m.get("context_changed"),
        "retrieval_total_time": m.get("retrieval_total_time"),
        "llm_ttft": m.get("llm_ttft"),
        "llm_reload_or_warmup_time": m.get("llm_reload_or_warmup_time"),
        "pure_inference_ttft": m.get("pure_inference_ttft"),
        "end_to_end_ttft": m.get("end_to_end_ttft"),
        "end_to_end_total": m.get("end_to_end_total"),
        "ollama_load_duration_ms": m.get("ollama_load_duration_ms"),
        "judgement_3s": _judge_3s(m.get("end_to_end_ttft")),
        "llm_fast_warm": warm.get("llm_fast_warm"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data/processed/logs/warmup_ab_benchmark.json"),
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = _load_rows(PILOT_QUESTIONS)
    collection, embed_model, manifest = warm_all_resources(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)

    all_results: list[dict] = []
    for model in models:
        print(f"\n========== {model} LEGACY warmup ==========")
        legacy_warmup_generate(model, args.ollama_base)
        for qid in QIDS:
            r = run_fast(
                rows[qid], model, args.ollama_base, collection, embed_model, manifest,
                auto_llm_warm=False,
            )
            r["warmup_mode"] = "legacy"
            all_results.append(r)
            print(f"  {qid} e2e_ttft={r['end_to_end_ttft']} pure={r['pure_inference_ttft']} {r['judgement_3s']}")

        print(f"\n========== {model} MATCHED warmup ==========")
        warmup_fast_chat(model, args.ollama_base, force=True)
        for qid in QIDS:
            r = run_fast(
                rows[qid], model, args.ollama_base, collection, embed_model, manifest,
                auto_llm_warm=True,
            )
            r["warmup_mode"] = "matched"
            all_results.append(r)
            print(f"  {qid} e2e_ttft={r['end_to_end_ttft']} pure={r['pure_inference_ttft']} {r['judgement_3s']}")

    print(
        "\n| model | QID | warmup | matched | ctx_chg | retrieval | llm_ttft | "
        "pure_inf | e2e_ttft | e2e_total | 3s |"
    )
    print("| --- | --- | --- | --- | --- | ---:| ---:| ---:| ---:| ---:| --- |")
    for r in all_results:
        print(
            f"| {r['model']} | {r['qid']} | {r['warmup_mode']} | {r.get('warmup_matched')} | "
            f"{r.get('context_changed')} | {r.get('retrieval_total_time')} | {r.get('llm_ttft')} | "
            f"{r.get('pure_inference_ttft')} | {r.get('end_to_end_ttft')} | "
            f"{r.get('end_to_end_total')} | {r.get('judgement_3s')} |"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
