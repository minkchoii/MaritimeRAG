"""Fast mode 3s TTFT stability: 10x repeat, Accurate→Fast switch, keep_alive expiry."""
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
from ollama_warmup import (
    WARMUP_TRACE_LOG,
    bootstrap_all_resources,
    get_warm_status_display,
    invalidate_fast_warm_for_test,
    warmup_fast_chat,
)
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from retrieval_timing import finalize_ui_timing

PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
QIDS = ("V01", "V05", "V06")
MODELS = ("llama3.2:latest", "llama3.1:8b")
REPEATS = 10

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


def _judge_3s(e2e: float | None) -> str:
    if e2e is None:
        return "—"
    if e2e <= 3.0:
        return "PASS"
    if e2e <= 5.0:
        return "NEAR"
    return "FAIL"


def _pass_bool(e2e: float | None) -> bool:
    return e2e is not None and e2e <= 3.0


def run_fast(
    row: dict,
    model: str,
    ollama_base: str,
    collection,
    embed_model,
    manifest,
    *,
    run_index: int,
    phase: str = "stability",
    auto_llm_warm: bool = True,
) -> dict[str, Any]:
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
    env = snapshot_ollama_env(model, ollama_base)
    warm = get_warm_status_display()
    e2e = m.get("end_to_end_ttft")
    return {
        "timestamp": _utc_iso(),
        "phase": phase,
        "run_index": run_index,
        "qid": row["question_id"],
        "model_name": model,
        "mode": "fast",
        "gpu_detected": env.get("gpu_detected"),
        "processor": env.get("ollama_processor"),
        "warmup_state": warm,
        "retrieval_total_time": m.get("retrieval_total_time"),
        "llm_ttft": m.get("llm_ttft"),
        "pure_inference_ttft": m.get("pure_inference_ttft"),
        "end_to_end_ttft": e2e,
        "end_to_end_total": m.get("end_to_end_total"),
        "context_changed": m.get("context_changed"),
        "warmup_matched_request": m.get("warmup_matched_request"),
        "rewarm_triggered": m.get("rewarm_triggered"),
        "rewarm_reason": m.get("rewarm_reason"),
        "input_token_estimate": final.get("input_token_estimate"),
        "final_prompt_chars": final.get("final_prompt_chars"),
        "selected_chunk_count": final.get("selected_chunk_count"),
        "timing_metrics": m,
        "pass_3s": _pass_bool(e2e),
        "judgement_3s": _judge_3s(e2e),
    }


def run_accurate(
    row: dict,
    model: str,
    ollama_base: str,
    collection,
    embed_model,
    manifest,
    *,
    run_index: int,
    phase: str,
) -> dict[str, Any]:
    user_submit_ts = time.time()
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
        **ACCURATE_DEFAULTS,
    )
    final = finalize_ui_timing(out.get("timing_log"), user_submit_ts) or out.get("timing_log") or {}
    m = final.get("timing_metrics") or {}
    e2e = m.get("end_to_end_ttft")
    return {
        "timestamp": _utc_iso(),
        "phase": phase,
        "run_index": run_index,
        "qid": row["question_id"],
        "model_name": model,
        "mode": "accurate",
        "end_to_end_ttft": e2e,
        "end_to_end_total": m.get("end_to_end_total"),
        "pass_3s": _pass_bool(e2e),
        "judgement_3s": _judge_3s(e2e),
        "timing_metrics": m,
    }


def summarize_stability(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.get("phase") != "stability_repeat":
            continue
        key = (r["model_name"], r["qid"])
        groups.setdefault(key, []).append(float(r["end_to_end_ttft"] or 0))
    out = []
    for (model, qid), vals in sorted(groups.items()):
        passes = sum(1 for v in vals if v <= 3.0)
        out.append(
            {
                "model": model,
                "qid": qid,
                "avg_e2e_ttft": round(statistics.mean(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "pass_count": passes,
                "fail_count": len(vals) - passes,
                "all_pass": passes == len(vals),
            }
        )
    return out


def run_stability_suite(
    model: str,
    ollama_base: str,
    rows: dict[str, dict],
    collection,
    embed_model,
    manifest,
    *,
    repeats: int = REPEATS,
) -> list[dict]:
    results: list[dict] = []
    warmup_fast_chat(model, ollama_base, force=True)
    for qid in QIDS:
        for i in range(1, repeats + 1):
            r = run_fast(
                rows[qid],
                model,
                ollama_base,
                collection,
                embed_model,
                manifest,
                run_index=i,
                phase="stability_repeat",
            )
            results.append(r)
            print(
                f"  [{model}] {qid} run={i} e2e={r['end_to_end_ttft']}s "
                f"rewarm={r.get('rewarm_triggered')} {r['judgement_3s']}"
            )
    return results


def run_accurate_switch_suite(
    model: str,
    ollama_base: str,
    rows: dict[str, dict],
    collection,
    embed_model,
    manifest,
) -> list[dict]:
    warmup_fast_chat(model, ollama_base, force=True)
    scenario = [
        ("fast", "V01", "fast_before"),
        ("accurate", "V01", "accurate"),
        ("fast", "V01", "fast_after_accurate"),
        ("fast", "V05", "fast_before"),
        ("accurate", "V05", "accurate"),
        ("fast", "V05", "fast_after_accurate"),
        ("fast", "V06", "fast_before"),
        ("accurate", "V06", "accurate"),
        ("fast", "V06", "fast_after_accurate"),
    ]
    results: list[dict] = []
    for idx, (mode, qid, phase) in enumerate(scenario, start=1):
        if mode == "fast":
            r = run_fast(
                rows[qid],
                model,
                ollama_base,
                collection,
                embed_model,
                manifest,
                run_index=idx,
                phase=phase,
            )
        else:
            r = run_accurate(
                rows[qid],
                model,
                ollama_base,
                collection,
                embed_model,
                manifest,
                run_index=idx,
                phase=phase,
            )
        results.append(r)
        if mode == "fast":
            print(
                f"  [{model}] {phase} {qid} e2e={r.get('end_to_end_ttft')}s "
                f"rewarm={r.get('rewarm_triggered')} reason={r.get('rewarm_reason')} "
                f"{r.get('judgement_3s')}"
            )
        else:
            print(f"  [{model}] {phase} {qid} accurate e2e={r.get('end_to_end_ttft')}s")
    return results


def run_keep_alive_suite(
    model: str,
    ollama_base: str,
    rows: dict[str, dict],
    collection,
    embed_model,
    manifest,
    *,
    short_keep_alive: str | None = "10s",
) -> list[dict]:
    results: list[dict] = []
    # Normal keep_alive warm + run
    warmup_fast_chat(model, ollama_base, force=True, keep_alive="30m")
    r1 = run_fast(
        rows["V01"],
        model,
        ollama_base,
        collection,
        embed_model,
        manifest,
        run_index=1,
        phase="keep_alive_warm_run1",
    )
    results.append(r1)
    print(f"  [{model}] keep_alive run1 e2e={r1['end_to_end_ttft']}s {r1['judgement_3s']}")

    # Immediate second run — should stay warm
    r2 = run_fast(
        rows["V01"],
        model,
        ollama_base,
        collection,
        embed_model,
        manifest,
        run_index=2,
        phase="keep_alive_immediate_run2",
    )
    results.append(r2)
    print(
        f"  [{model}] keep_alive immediate run2 e2e={r2['end_to_end_ttft']}s "
        f"rewarm={r2.get('rewarm_triggered')} {r2['judgement_3s']}"
    )

    if short_keep_alive:
        warmup_fast_chat(model, ollama_base, force=True, keep_alive=short_keep_alive)
        r3 = run_fast(
            rows["V01"],
            model,
            ollama_base,
            collection,
            embed_model,
            manifest,
            run_index=3,
            phase="keep_alive_short_warm_run",
        )
        results.append(r3)
        wait_s = 12
        print(f"  [{model}] waiting {wait_s}s for keep_alive={short_keep_alive} expiry…")
        time.sleep(wait_s)
        invalidate_fast_warm_for_test("keep_alive_expiry_simulated")
        r4 = run_fast(
            rows["V01"],
            model,
            ollama_base,
            collection,
            embed_model,
            manifest,
            run_index=4,
            phase="keep_alive_after_expiry",
        )
        results.append(r4)
        print(
            f"  [{model}] after expiry e2e={r4['end_to_end_ttft']}s "
            f"rewarm={r4.get('rewarm_triggered')} reason={r4.get('rewarm_reason')} "
            f"{r4['judgement_3s']}"
        )
    return results


def print_stability_table(summary: list[dict]) -> None:
    print(
        "\n| model | qid | avg_e2e_ttft | min | max | std | pass_count | fail_count |"
    )
    print("| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|")
    for s in summary:
        print(
            f"| {s['model']} | {s['qid']} | {s['avg_e2e_ttft']} | {s['min']} | "
            f"{s['max']} | {s['std']} | {s['pass_count']} | {s['fail_count']} |"
        )


def print_switch_table(switch_rows: list[dict]) -> None:
    print(
        "\n| model | qid | phase | context_changed | rewarm_triggered | "
        "rewarm_reason | e2e_ttft | 3s |"
    )
    print("| --- | --- | --- | --- | --- | --- | ---:| --- |")
    for r in switch_rows:
        if r.get("mode") != "fast":
            continue
        print(
            f"| {r['model_name']} | {r['qid']} | {r['phase']} | "
            f"{r.get('context_changed')} | {r.get('rewarm_triggered')} | "
            f"{r.get('rewarm_reason')} | {r.get('end_to_end_ttft')} | {r.get('judgement_3s')} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast mode stability benchmark")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--skip-keep-alive", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--switch-only", action="store_true")
    args = parser.parse_args()
    repeats = args.repeats

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = _load_rows(PILOT_QUESTIONS)

    stability_all: list[dict] = []
    switch_all: list[dict] = []
    keep_alive_all: list[dict] = []

    for model in models:
        print(f"\n========== Bootstrap {model} ==========")
        boot = bootstrap_all_resources(model, args.ollama_base, DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)
        collection = boot["collection"]
        embed_model = boot["embed_model"]
        manifest = boot["manifest"]
        env = snapshot_ollama_env(model, args.ollama_base)
        print(json.dumps({k: env[k] for k in ("ollama_processor", "gpu_detected", "gpu_name")}, indent=2))

        print(f"\n--- Stability {repeats}x ---")
        if not args.skip_stability and not args.switch_only:
            stability_all.extend(
                run_stability_suite(
                    model, args.ollama_base, rows, collection, embed_model, manifest, repeats=repeats
                )
            )

        print(f"\n--- Accurate→Fast switch ---")
        switch_all.extend(
            run_accurate_switch_suite(model, args.ollama_base, rows, collection, embed_model, manifest)
        )

        if not args.skip_keep_alive and not args.switch_only:
            print(f"\n--- keep_alive test ---")
            keep_alive_all.extend(
                run_keep_alive_suite(
                    model, args.ollama_base, rows, collection, embed_model, manifest
                )
            )

    summary = summarize_stability(stability_all)
    print_stability_table(summary)
    print_switch_table(switch_all)

    out_stability = _ROOT / "data/processed/logs/fast_stability_benchmark.json"
    out_log = _ROOT / "data/processed/logs/fast_stability_benchmark.log"
    out_switch = _ROOT / "data/processed/logs/fast_after_accurate_switch_benchmark.json"
    payload = {
        "benchmark_type": "fast_stability",
        "repeats": repeats,
        "models": models,
        "stability_runs": stability_all,
        "stability_summary": summary,
        "accurate_switch_runs": switch_all,
        "keep_alive_runs": keep_alive_all,
        "warmup_trace_log": str(WARMUP_TRACE_LOG),
    }
    out_stability.parent.mkdir(parents=True, exist_ok=True)
    out_stability.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_lines = [
        f"Fast stability benchmark @ {_utc_iso()}",
        f"models={models} repeats={repeats}",
        "",
        "=== Stability summary ===",
    ]
    for s in summary:
        log_lines.append(
            f"{s['model']} {s['qid']}: avg={s['avg_e2e_ttft']}s min={s['min']} max={s['max']} "
            f"pass={s['pass_count']}/{s['pass_count'] + s['fail_count']}"
        )
    log_lines.extend(["", "=== Accurate→Fast switch ==="])
    for r in switch_all:
        if r.get("phase") in ("fast_before", "fast_after_accurate"):
            log_lines.append(
                f"{r.get('model_name')} {r.get('qid')} {r.get('phase')}: "
                f"e2e={r.get('end_to_end_ttft')}s {r.get('judgement_3s')}"
            )
    log_lines.extend(["", "=== keep_alive ==="])
    for r in keep_alive_all:
        log_lines.append(
            f"{r.get('model_name')} {r.get('phase')}: e2e={r.get('end_to_end_ttft')}s "
            f"rewarm={r.get('rewarm_triggered')} {r.get('judgement_3s')}"
        )
    out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    out_switch.write_text(
        json.dumps(
            {"benchmark_type": "accurate_fast_switch", "runs": switch_all},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_stability}")
    print(f"Saved: {out_log}")
    print(f"Saved: {out_switch}")
    print(f"Warmup trace: {WARMUP_TRACE_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
