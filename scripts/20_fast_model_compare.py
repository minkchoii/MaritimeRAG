"""Fast-mode only comparison across small Ollama models (GPU TTFT sweep)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ollama_warmup import warmup_fast_chat
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from rag_resource_cache import warm_all_resources

PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
DEFAULT_MODELS = (
    "llama3.2:latest",
    "qwen2.5:1.5b",
    "llama3.1:8b",
)
QIDS = ("V06",)  # quick probe; use --qids V01,V05,V06 for full


def _load_rows(path: Path) -> dict[str, dict]:
    from rag_eval_lib import load_questions

    return {r["question_id"]: r for r in load_questions(path)}


def run_fast_probe(row: dict, model: str, ollama_base: str, collection, embed_model, manifest) -> dict:
    import time

    from retrieval_timing import finalize_ui_timing

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
    )
    final = finalize_ui_timing(out.get("timing_log"), user_submit_ts) or out.get("timing_log") or {}
    m = final.get("timing_metrics") or {}
    env = snapshot_ollama_env(model, ollama_base)
    return {
        "question_id": row["question_id"],
        "model_name": model,
        "ollama_processor": env.get("ollama_processor"),
        "gpu_vram_used_mb": env.get("gpu_vram_used_mb"),
        "input_token_estimate": final.get("input_token_estimate"),
        "llm_ttft": m.get("llm_ttft"),
        "end_to_end_ttft": m.get("end_to_end_ttft"),
        "llm_generation_time": m.get("llm_generation_time"),
        "end_to_end_total": m.get("end_to_end_total"),
        "answer_preview": (out.get("answer") or "")[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast mode model comparison (GPU)")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--qids", default="V06")
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--questions", default=str(PILOT_QUESTIONS))
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    rows = _load_rows(Path(args.questions))
    collection, embed_model, manifest = warm_all_resources(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)

    results = []
    for model in models:
        print(f"\n=== Model: {model} ===")
        warmup_fast_chat(model, args.ollama_base, force=True)
        for qid in qids:
            r = run_fast_probe(rows[qid], model, args.ollama_base, collection, embed_model, manifest)
            results.append(r)
            print(
                f"  {qid} proc={r['ollama_processor']} "
                f"E2E TTFT={r.get('end_to_end_ttft')}s LLM TTFT={r.get('llm_ttft')}s"
            )

    out = _ROOT / "data/processed/logs/fast_model_compare_gpu.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("\n| model | processor | LLM TTFT | E2E TTFT | gen |")
    print("|-------|-----------|----------|----------|-----|")
    for r in results:
        print(
            f"| {r['model_name']} | {r.get('ollama_processor')} | "
            f"{r.get('llm_ttft')}s | {r.get('end_to_end_ttft')}s | {r.get('llm_generation_time')}s |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
