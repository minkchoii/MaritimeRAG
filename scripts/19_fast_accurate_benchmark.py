"""Compare Fast vs Accurate latency on GPU (with CPU baseline comparison)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ollama_env_probe import snapshot_ollama_env
from ollama_warmup import get_warm_status_display, legacy_warmup_generate, warmup_fast_chat
from rag_answer_lib import DEFAULT_OLLAMA_MODEL
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from rag_resource_cache import warm_all_resources
from retrieval_timing import DEFAULT_TIMING_LOG, append_timing_log, finalize_ui_timing

PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
QIDS = ("V01", "V05", "V06")

ACCURATE_DEFAULTS = {
    "top_k": 10,
    "fetch_k": 120,
    "max_doc": 3,
    "max_docs": 10,
    "use_rerank": True,
}

# CPU-only Ollama baseline (llama3.1:8b, processor=100% CPU)
CPU_BASELINE = {
    ("V01", "fast"): {"e2e_ttft": 12.0, "e2e_total": 60.0, "llm_ttft": 11.0, "llm_total": 59.0},
    ("V05", "fast"): {"e2e_ttft": 12.0, "e2e_total": 47.0, "llm_ttft": 11.0, "llm_total": 47.0},
    ("V06", "fast"): {"e2e_ttft": 32.0, "e2e_total": 72.0, "llm_ttft": 32.0, "llm_total": 71.0},
    ("V01", "accurate"): {"e2e_ttft": 333.0, "e2e_total": 526.0, "llm_ttft": 332.0, "llm_total": 525.0},
    ("V05", "accurate"): {"e2e_ttft": 77.0, "e2e_total": 244.0, "llm_ttft": 77.0, "llm_total": 243.0},
    ("V06", "accurate"): {"e2e_ttft": 83.0, "e2e_total": 195.0, "llm_ttft": 82.0, "llm_total": 194.0},
}


def _load_rows(path: Path) -> dict[str, dict]:
    from rag_eval_lib import load_questions

    return {r["question_id"]: r for r in load_questions(path)}


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 10:
        return f"{v:.0f}s"
    return f"{v:.2f}s"


def _tokens_per_second(log_row: dict, metrics: dict) -> float | None:
    out_tok = log_row.get("output_token_estimate")
    gen = metrics.get("llm_generation_time")
    if out_tok and gen and gen > 0:
        return round(out_tok / gen, 2)
    return None


def _judge_3s(e2e_ttft: float | None) -> str:
    if e2e_ttft is None:
        return "—"
    if e2e_ttft <= 3.0:
        return "PASS"
    if e2e_ttft <= 5.0:
        return "NEAR"
    return "FAIL"


def run_one(
    row: dict,
    *,
    latency_mode: str,
    collection,
    embed_model,
    manifest,
    llm_model: str,
    ollama_base: str,
    llm_run_phase: str,
    log: bool,
) -> dict[str, Any]:
    user_submit_ts = time.time()
    kwargs: dict[str, Any] = {
        "user_submit_ts": user_submit_ts,
        "collection": collection,
        "embed_model": embed_model,
        "manifest": manifest,
        "llm_model": llm_model,
        "ollama_base": ollama_base,
        "start_type": "warm" if llm_run_phase == "warm" else "cold",
        "latency_mode": latency_mode,
        "multi_doc_strategy": "single_pass",
        "max_llm_docs": 6,
    }
    if latency_mode == "accurate":
        kwargs.update(ACCURATE_DEFAULTS)
    else:
        kwargs.update(top_k=3, fetch_k=10, max_doc=1, max_docs=2, use_rerank=False)

    out = run_full_inprocess(row, **kwargs)
    final = finalize_ui_timing(out.get("timing_log"), user_submit_ts)
    log_row = final or out.get("timing_log") or {}
    m = log_row.get("timing_metrics") or out.get("timing_metrics") or {}
    answer_out = out.get("answer_out") or {}
    prompt_meta = answer_out.get("prompt_meta") or {}

    answer_text = out.get("answer") or answer_out.get("answer") or ""

    gpu_env = snapshot_ollama_env(llm_model, ollama_base)
    tps = _tokens_per_second(log_row, m)

    record = {
        "qid": row["question_id"],
        "question_id": row["question_id"],
        "mode": latency_mode,
        "latency_mode": latency_mode,
        "llm_run_phase": llm_run_phase,
        "model_name": llm_model,
        "processor": gpu_env.get("ollama_processor"),
        "chunks": log_row.get("selected_chunk_count"),
        "selected_doc_count": log_row.get("selected_doc_count"),
        "selected_chunk_count": log_row.get("selected_chunk_count"),
        "final_prompt_chars": log_row.get("final_prompt_chars"),
        "input_token_estimate": log_row.get("input_token_estimate"),
        "num_ctx": log_row.get("num_ctx"),
        "max_new_tokens": log_row.get("max_new_tokens"),
        "timing_metrics": m,
        "query_embedding_time": m.get("query_embedding_time"),
        "vector_search_time": m.get("vector_search_time"),
        "retrieval_total_time": m.get("retrieval_total_time"),
        "llm_ttft": m.get("llm_ttft"),
        "llm_reload_or_warmup_time": m.get("llm_reload_or_warmup_time"),
        "pure_inference_ttft": m.get("pure_inference_ttft"),
        "llm_generation_time": m.get("llm_generation_time"),
        "end_to_end_ttft": m.get("end_to_end_ttft"),
        "end_to_end_total": m.get("end_to_end_total"),
        "ollama_load_duration_ms": m.get("ollama_load_duration_ms"),
        "ollama_prompt_eval_duration_ms": m.get("ollama_prompt_eval_duration_ms"),
        "context_changed": m.get("context_changed"),
        "warmup_matched_request": m.get("warmup_matched_request"),
        "tokens_per_second": tps,
        "ollama_processor": gpu_env.get("ollama_processor"),
        "gpu_detected": gpu_env.get("gpu_detected"),
        "gpu_name": gpu_env.get("gpu_name"),
        "gpu_vram_used_mb": gpu_env.get("gpu_vram_used_mb"),
        "ollama_context": gpu_env.get("ollama_context"),
        "answer_length_chars": len(answer_text),
        "answer_preview": answer_text[:300],
        "answer": answer_text,
        "gpu_env": gpu_env,
        "timing_log": log_row,
        "prompt_meta": prompt_meta,
    }
    if final and log:
        final.update(
            {
                "latency_mode": latency_mode,
                "llm_run_phase": llm_run_phase,
                "model_name": llm_model,
                "ollama_processor": record["ollama_processor"],
                "gpu_detected": record["gpu_detected"],
                "gpu_name": record["gpu_name"],
                "gpu_vram_used_mb": record["gpu_vram_used_mb"],
                "ollama_context": record["ollama_context"],
                "tokens_per_second": tps,
            }
        )
        append_timing_log(final, DEFAULT_TIMING_LOG)
    return record


def print_table(results: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(
        f"| {'QID':<4} | {'mode':>8} | {'phase':>5} | {'proc':>8} | {'chunks':>6} | "
        f"{'tokens':>6} | {'retrieval':>8} | {'LLM TTFT':>8} | {'E2E TTFT':>8} | {'E2E total':>9} | {'tok/s':>6} |"
    )
    print("|" + "-" * 110 + "|")
    for r in results:
        m = r["timing_metrics"]
        print(
            f"| {r['question_id']:<4} | {r['latency_mode']:>8} | {r.get('llm_run_phase','?'):>5} | "
            f"{str(r.get('ollama_processor',''))[:8]:>8} | {r.get('chunks') or '—':>6} | "
            f"{r.get('input_token_estimate') or '—':>6} | "
            f"{_fmt(m.get('retrieval_total_time')):>8} | {_fmt(m.get('llm_ttft')):>8} | "
            f"{_fmt(m.get('end_to_end_ttft')):>8} | {_fmt(m.get('end_to_end_total')):>9} | "
            f"{r.get('tokens_per_second') or '—':>6} |"
        )


def print_cpu_gpu_comparison(gpu_results: list[dict], model: str) -> None:
    print("\n" + "=" * 100)
    print("CPU vs GPU comparison (warm GPU runs vs historical CPU baseline)")
    print(
        f"| {'QID':<4} | {'mode':<8} | {'model':<14} | {'processor':<10} | "
        f"{'input_tok':>8} | {'CPU E2E TTFT':>12} | {'GPU E2E TTFT':>12} | "
        f"{'CPU total':>9} | {'GPU total':>9} | {'개선율':>8} |"
    )
    print("|" + "-" * 115 + "|")
    warm = [r for r in gpu_results if r.get("llm_run_phase") == "warm"]
    for r in warm:
        key = (r["question_id"], r["latency_mode"])
        cpu = CPU_BASELINE.get(key, {})
        cpu_ttft = cpu.get("e2e_ttft")
        gpu_ttft = r.get("end_to_end_ttft")
        cpu_total = cpu.get("e2e_total")
        gpu_total = r.get("end_to_end_total")
        imp = "—"
        if cpu_ttft and gpu_ttft and cpu_ttft > 0:
            imp = f"{(1 - gpu_ttft / cpu_ttft) * 100:.0f}%"
        print(
            f"| {r['question_id']:<4} | {r['latency_mode']:<8} | {model:<14} | "
            f"{str(r.get('ollama_processor',''))[:10]:<10} | "
            f"{r.get('input_token_estimate') or '—':>8} | "
            f"{_fmt(cpu_ttft):>12} | {_fmt(gpu_ttft):>12} | "
            f"{_fmt(cpu_total):>9} | {_fmt(gpu_total):>9} | {imp:>8} |"
        )


def print_3s_judgement(fast_warm: list[dict]) -> None:
    print("\n3-second TTFT judgement (Fast mode, warm LLM):")
    print(f"| {'QID':<4} | {'Fast E2E TTFT':>14} | {'3초 조건':<12} |")
    print(f"|{'-' * 6}|{'-' * 16}|{'-' * 14}|")
    for r in fast_warm:
        ttft = r.get("end_to_end_ttft")
        print(f"| {r['question_id']:<4} | {_fmt(ttft):>14} | {_judge_3s(ttft):<12} |")


def diagnose_3s_failures(fast_warm: list[dict]) -> None:
    print("\nGap analysis (Fast warm, if 3s not met):")
    for r in fast_warm:
        if _judge_3s(r.get("end_to_end_ttft")) == "PASS":
            continue
        m = r["timing_metrics"]
        print(f"\n  [{r['question_id']}] E2E TTFT={_fmt(r.get('end_to_end_ttft'))}")
        print(f"    retrieval={_fmt(m.get('retrieval_total_time'))} "
              f"(embed={_fmt(m.get('query_embedding_time'))}, vector={_fmt(m.get('vector_search_time'))})")
        print(f"    llm_ttft={_fmt(m.get('llm_ttft'))} | tokens={r.get('input_token_estimate')} | "
              f"num_ctx={r.get('num_ctx')} | ollama_ctx={r.get('ollama_context')}")
        print(f"    processor={r.get('ollama_processor')} | vram={r.get('gpu_vram_used_mb')}MB")
        ret = m.get("retrieval_total_time") or 0
        llm = m.get("llm_ttft") or 0
        if ret > 2:
            print("    -> retrieval > 2s")
        if llm > 3:
            print("    -> LLM TTFT > 3s (model size / num_ctx / warm-up)")
        if r.get("num_ctx", 0) and r.get("num_ctx", 0) > 8192:
            print("    -> num_ctx may add overhead")


def _default_output_path(model: str) -> Path:
    if model == "llama3.1:8b":
        return _ROOT / "data/processed/logs/fast_accurate_benchmark_gpu_llama31_8b.json"
    if model == "llama3.2:latest":
        return _ROOT / "data/processed/logs/fast_accurate_benchmark_gpu.json"
    slug = model.replace(":", "_").replace(".", "")
    return _ROOT / f"data/processed/logs/fast_accurate_benchmark_gpu_{slug}.json"


def print_metrics_table(results: list[dict], title: str) -> None:
    print(f"\n{title}")
    cols = (
        "qid", "mode", "model_name", "processor", "input_token_estimate",
        "retrieval_total_time", "llm_ttft", "end_to_end_ttft", "end_to_end_total",
        "answer_length_chars",
    )
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in results:
        print(
            f"| {r.get('qid')} | {r.get('mode')} | {r.get('model_name')} | "
            f"{r.get('processor')} | {r.get('input_token_estimate')} | "
            f"{r.get('retrieval_total_time')} | {r.get('llm_ttft')} | "
            f"{r.get('end_to_end_ttft')} | {r.get('end_to_end_total')} | "
            f"{r.get('answer_length_chars')} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast vs Accurate GPU benchmark with CPU comparison")
    parser.add_argument("--questions", default=str(PILOT_QUESTIONS))
    parser.add_argument("--qids", default=",".join(QIDS))
    parser.add_argument("--llm-model", default=DEFAULT_OLLAMA_MODEL, help=f"default: {DEFAULT_OLLAMA_MODEL}")
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--skip-cold-run", action="store_true")
    parser.add_argument(
        "--warmup-mode",
        choices=("legacy", "matched"),
        default="matched",
        help="legacy=/api/generate num_ctx=512; matched=/api/chat num_ctx=4096",
    )
    parser.add_argument("--skip-llm-warmup", action="store_true")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path (default: model-specific under data/processed/logs/)",
    )
    args = parser.parse_args()

    qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    rows = _load_rows(Path(args.questions))
    model = args.llm_model

    print("=" * 80)
    print(f"Model: {model} (UI default: {DEFAULT_OLLAMA_MODEL})")
    print(f"Fast & Accurate use the SAME model: {model}")
    print("=" * 80)

    env_before = snapshot_ollama_env(model, args.ollama_base)
    print("\n[GPU env BEFORE benchmark]")
    print(json.dumps({k: env_before[k] for k in (
        "model_name", "ollama_processor", "gpu_detected", "gpu_name",
        "gpu_vram_used_mb", "llama_server_on_gpu", "ollama_context",
    ) if k in env_before}, indent=2, ensure_ascii=False))
    print(env_before.get("ollama_ps_cli", "")[:800])

    print("\nWarming RAG resources (vector DB + embedding)...")
    collection, embed_model, manifest = warm_all_resources(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)

    results: list[dict] = []
    cold_results: list[dict] = []

    if not args.skip_cold_run:
        print(f"\n### COLD LLM run (no warm-up): V06 / fast")
        cold = run_one(
            rows["V06"],
            latency_mode="fast",
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            llm_model=model,
            ollama_base=args.ollama_base,
            llm_run_phase="cold",
            log=not args.no_log,
        )
        cold_results.append(cold)
        results.append(cold)
        print(f"  cold E2E TTFT={_fmt(cold.get('end_to_end_ttft'))} processor={cold.get('ollama_processor')}")

    warmup_info: dict[str, Any] = {}
    if not args.skip_llm_warmup:
        print(f"\n### LLM WARM-UP ({args.warmup_mode}): {model}")
        if args.warmup_mode == "legacy":
            warmup_info = legacy_warmup_generate(model, args.ollama_base)
            print(f"  legacy warmup elapsed={warmup_info.get('warmup_elapsed_s')}s api=generate num_ctx=512")
        else:
            warmup_info = warmup_fast_chat(model, args.ollama_base, force=True)
            print(
                f"  matched warmup ttft={warmup_info.get('warmup_ttft')}s "
                f"total={warmup_info.get('warmup_total_time')}s api=chat num_ctx=4096"
            )
        print(json.dumps(get_warm_status_display(), indent=2, ensure_ascii=False)[:600])

    for qid in qids:
        row = rows[qid]
        for mode in ("fast", "accurate"):
            print(f"\nRunning {qid} / {mode} (warm LLM)...")
            try:
                r = run_one(
                    row,
                    latency_mode=mode,
                    collection=collection,
                    embed_model=embed_model,
                    manifest=manifest,
                    llm_model=model,
                    ollama_base=args.ollama_base,
                    llm_run_phase="warm",
                    log=not args.no_log,
                )
                results.append(r)
                print(
                    f"  E2E TTFT={_fmt(r.get('end_to_end_ttft'))} "
                    f"LLM TTFT={_fmt(r.get('llm_ttft'))} "
                    f"proc={r.get('ollama_processor')} vram={r.get('gpu_vram_used_mb')}MB"
                )
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr)

    env_after = snapshot_ollama_env(model, args.ollama_base)
    print("\n[GPU env AFTER benchmark]")
    print(json.dumps({k: env_after[k] for k in (
        "model_name", "ollama_processor", "gpu_detected", "gpu_name",
        "gpu_vram_used_mb", "llama_server_on_gpu", "ollama_context",
    ) if k in env_after}, indent=2, ensure_ascii=False))
    print(env_after.get("ollama_ps_cli", "")[:800])

    warm_results = [r for r in results if r.get("llm_run_phase") == "warm"]
    print_table(results, "All runs (cold + warm)")
    print_metrics_table(warm_results, "Warm runs — key metrics")
    print_cpu_gpu_comparison(warm_results, model)
    fast_warm = [r for r in warm_results if r["latency_mode"] == "fast"]
    print_3s_judgement(fast_warm)
    diagnose_3s_failures(fast_warm)

    out_path = Path(args.output_json) if args.output_json else _default_output_path(model)
    payload = {
        "benchmark_type": "gpu",
        "warmup_mode": args.warmup_mode,
        "model_name": model,
        "ui_default_model": DEFAULT_OLLAMA_MODEL,
        "env_before": env_before,
        "env_after": env_after,
        "warmup_info": warmup_info,
        "cold_results": cold_results,
        "warm_results": warm_results,
        "all_results": results,
        "cpu_baseline": {f"{k[0]}_{k[1]}": v for k, v in CPU_BASELINE.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
