"""Verify Rule/Guidance Fast vs Accurate paths (A/B/C)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from rag_query_router import enrich_row_for_routing, resolve_pipeline_route
from rag_inprocess import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_UNIFIED,
    run_full_inprocess,
)
from rag_resource_cache import warm_all_resources
from ollama_warmup import warmup_fast_chat

TESTS = {
    "A_fast_lr": {
        "latency_mode": "fast",
        "question": "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.",
        "expect_llm": False,
        "expect_source": "structured_template",
    },
    "B_accurate_lr": {
        "latency_mode": "accurate",
        "question": "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.",
        "expect_llm": True,
        "expect_source": "llm_grounded_summary",
    },
    "C_no_evidence": {
        "latency_mode": "accurate",
        "question": "LR에서 존재하지 않는 XYZ-9999 연료 규정 Rule/Guidance를 찾아줘.",
        "expect_llm": False,
        "expect_source": "fallback_no_evidence",
    },
}


def run_case(name: str, spec: dict, col, emb, man, model: str) -> dict:
    t0 = time.perf_counter()
    row = enrich_row_for_routing(
        {"question_id": name, "question": spec["question"]},
        latency_mode=spec["latency_mode"],
    )
    route = resolve_pipeline_route(spec["question"], row, latency_mode=spec["latency_mode"])
    out = run_full_inprocess(
        row,
        collection=col,
        embed_model=emb,
        manifest=man,
        latency_mode=spec["latency_mode"],
        llm_model=model,
        auto_llm_warm=False,
        skip_ollama_probe=True,
        unified_id=DEFAULT_UNIFIED,
        index_dir=DEFAULT_INDEX_DIR,
        chunks_dir=DEFAULT_CHUNKS_DIR,
    )
    elapsed = time.perf_counter() - t0
    so = out.get("search_out") or {}
    ao = out.get("answer_out") or {}
    tm = out.get("timing_metrics") or (ao.get("timing_metrics") or {})
    tl = out.get("timing_log") or {}
    trace = (tl.get("meta") or {}).get("rag_debug_trace") or tl.get("rag_debug_trace") or {}
    ag = row.get("_answer_generation") or trace.get("ANSWER_GENERATION") or {}
    final = trace.get("FINAL_CONTEXT_TO_LLM") or {}
    societies = {c.get("society") for c in (final.get("chunks") or [])}
    ret = so.get("retrieved") or []
    ret_soc = {getattr(c, "source", "") for c in ret}
    result = {
        "case": name,
        "latency_mode": spec["latency_mode"],
        "selected_answer_mode": route.get("selected_answer_mode"),
        "detected_society": route.get("detected_society"),
        "retrieved_societies": sorted(ret_soc),
        "final_context_societies": sorted(s for s in societies if s),
        "llm_used": ag.get("llm_used"),
        "answer_source": ag.get("answer_source"),
        "llm_call_function": ag.get("llm_call_function"),
        "llm_ttft": tm.get("llm_ttft"),
        "e2e_ttft": tm.get("end_to_end_ttft"),
        "accurate_llm_ttft": tm.get("accurate_llm_ttft"),
        "accurate_first_visible_llm_latency": tm.get("accurate_first_visible_llm_latency"),
        "rule_guidance_first_token_latency": (ag or {}).get("rule_guidance_first_token_latency"),
        "rule_guidance_first_token_3s_pass": (ag or {}).get("rule_guidance_first_token_3s_pass"),
        "total_ms": (trace.get("LATENCY_BREAKDOWN") or {}).get("total_ms"),
        "pass_3s": (trace.get("LATENCY_BREAKDOWN") or {}).get("pass_3s"),
        "wall_s": round(elapsed, 2),
        "answer_preview": (out.get("answer") or "")[:200],
        "expect_llm": spec["expect_llm"],
        "expect_source": spec["expect_source"],
        "pass": (
            ag.get("llm_used") == spec["expect_llm"]
            and ag.get("answer_source") == spec["expect_source"]
            and ret_soc <= {"LR"} if route.get("detected_society") == "LR" else True
            and (not societies or societies <= {"LR"})
        ),
    }
    return result


def main() -> None:
    import io
    import sys as _sys

    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model = "llama3.1:8b"
    print("Warming resources...")
    col, emb, man = warm_all_resources(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)
    warmup_fast_chat(model, "http://127.0.0.1:11434", force=True)
    results = []
    for name, spec in TESTS.items():
        print(f"\n--- {name} ---")
        r = run_case(name, spec, col, emb, man, model)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    out_path = _ROOT / "data/processed/logs/rule_guidance_mode_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("SUMMARY:", "PASS" if all(r["pass"] for r in results) else "FAIL", results)


if __name__ == "__main__":
    main()
