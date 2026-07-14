"""Compare llama3.2:latest vs llama3.1:8b GPU benchmark results."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
LOGS = _ROOT / "data/processed/logs"


def _judge_3s(e2e_ttft: float | None) -> str:
    if e2e_ttft is None:
        return "—"
    if e2e_ttft <= 3.0:
        return "PASS"
    if e2e_ttft <= 5.0:
        return "NEAR"
    return "FAIL"


def _quality_note(qid: str, mode: str, model: str, answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "답변 없음"
    has_citation = bool(re.search(r"\.pdf|p\.\d+|MSC|MEPC|DNV|page", text, re.I))
    notes: list[str] = []
    if qid == "V01":
        if re.search(r"MEPC|mepc|MARPOL|환경|ECA|Annex", text, re.I):
            notes.append("MEPC/환경규제 관련 내용 포함")
        else:
            notes.append("MEPC 핵심 내용 부족")
    elif qid == "V05":
        if re.search(r"MASS|MSC\s*111|autonomous", text, re.I):
            notes.append("MASS Code/MSC 111 관련")
        else:
            notes.append("MASS 핵심 부족")
    elif qid == "V06":
        if re.search(r"DNV|Smart|CG-0264|autonomous", text, re.I):
            notes.append("DNV/Smart Vessel 관련")
        else:
            notes.append("DNV 관련 부족")
    if has_citation:
        notes.append("문서명/페이지 인용 있음")
    else:
        notes.append("인용 약함")
    if len(text) < 120 and mode == "accurate":
        notes.append("Accurate 대비 짧음")
    if model == "llama3.1:8b" and len(text) > 400:
        notes.append("8B — 상세도 양호")
    if model == "llama3.2:latest" and mode == "fast" and len(text) < 200:
        notes.append("3.2B Fast — 간결")
    return "; ".join(notes)


def _row_from_record(r: dict) -> dict:
    return {
        "qid": r.get("qid") or r.get("question_id"),
        "mode": r.get("mode") or r.get("latency_mode"),
        "model": r.get("model_name"),
        "processor": r.get("processor") or r.get("ollama_processor"),
        "gpu_name": r.get("gpu_name"),
        "gpu_vram_used_mb": r.get("gpu_vram_used_mb"),
        "input_tokens": r.get("input_token_estimate"),
        "selected_doc_count": r.get("selected_doc_count"),
        "selected_chunk_count": r.get("selected_chunk_count"),
        "final_prompt_chars": r.get("final_prompt_chars"),
        "num_ctx": r.get("num_ctx"),
        "max_new_tokens": r.get("max_new_tokens"),
        "retrieval_total_time": r.get("retrieval_total_time"),
        "query_embedding_time": r.get("query_embedding_time"),
        "vector_search_time": r.get("vector_search_time"),
        "llm_ttft": r.get("llm_ttft"),
        "llm_generation_time": r.get("llm_generation_time"),
        "end_to_end_ttft": r.get("end_to_end_ttft"),
        "end_to_end_total": r.get("end_to_end_total"),
        "tokens_per_second": r.get("tokens_per_second"),
        "answer_length_chars": r.get("answer_length_chars"),
        "answer_preview": r.get("answer_preview") or (r.get("answer") or "")[:300],
        "judgement_3s": _judge_3s(r.get("end_to_end_ttft"))
        if (r.get("mode") or r.get("latency_mode")) == "fast"
        else "—",
        "quality_note": _quality_note(
            r.get("qid") or r.get("question_id") or "",
            r.get("mode") or r.get("latency_mode") or "",
            r.get("model_name") or "",
            r.get("answer") or r.get("answer_preview") or "",
        ),
    }


def load_warm(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    warm = data.get("warm_results") or []
    return [_row_from_record(r) for r in warm]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama32",
        default=str(LOGS / "fast_accurate_benchmark_gpu.json"),
    )
    parser.add_argument(
        "--llama31",
        default=str(LOGS / "fast_accurate_benchmark_gpu_llama31_8b.json"),
    )
    parser.add_argument(
        "--output",
        default=str(LOGS / "model_comparison_llama32_vs_llama31_8b.json"),
    )
    args = parser.parse_args()

    p32 = Path(args.llama32)
    p31 = Path(args.llama31)
    if not p32.is_file():
        print(f"Missing: {p32}", file=sys.stderr)
        return 1
    if not p31.is_file():
        print(f"Missing: {p31}", file=sys.stderr)
        return 1

    rows32 = {((r["qid"], r["mode"])): r for r in load_warm(p32)}
    rows31 = {((r["qid"], r["mode"])): r for r in load_warm(p31)}

    comparison: list[dict] = []
    for key in sorted(set(rows32) | set(rows31)):
        qid, mode = key
        r32 = rows32.get(key)
        r31 = rows31.get(key)
        comparison.append({"qid": qid, "mode": mode, "llama3.2:latest": r32, "llama3.1:8b": r31})

    fast_judgement = []
    for qid in ("V01", "V05", "V06"):
        r32 = rows32.get((qid, "fast"))
        r31 = rows31.get((qid, "fast"))
        fast_judgement.append(
            {
                "qid": qid,
                "llama3.2:latest": {
                    "e2e_ttft": r32.get("end_to_end_ttft") if r32 else None,
                    "judgement": r32.get("judgement_3s") if r32 else "—",
                },
                "llama3.1:8b": {
                    "e2e_ttft": r31.get("end_to_end_ttft") if r31 else None,
                    "judgement": r31.get("judgement_3s") if r31 else "—",
                },
            }
        )

    out = {
        "comparison_type": "llama3.2:latest vs llama3.1:8b (GPU warm)",
        "sources": {"llama3.2:latest": str(p32), "llama3.1:8b": str(p31)},
        "rows": comparison,
        "flat_comparison": [rows32.get(k) for k in sorted(rows32)] + [rows31.get(k) for k in sorted(rows31) if k not in rows32],
        "fast_3s_judgement": fast_judgement,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| QID | mode | model | processor | input_tokens | LLM TTFT | E2E TTFT | E2E total | 3초 조건 | quality note |")
    print("| --- | ---- | ----- | --------- | -----------: | -------: | -------: | --------: | ----- | ------------ |")
    for model_label, rows in (("llama3.2:latest", rows32), ("llama3.1:8b", rows31)):
        for key in sorted(rows):
            r = rows[key]
            print(
                f"| {r['qid']} | {r['mode']} | {model_label} | {r.get('processor')} | "
                f"{r.get('input_tokens')} | {r.get('llm_ttft')} | {r.get('end_to_end_ttft')} | "
                f"{r.get('end_to_end_total')} | {r.get('judgement_3s')} | {r.get('quality_note')} |"
            )

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
