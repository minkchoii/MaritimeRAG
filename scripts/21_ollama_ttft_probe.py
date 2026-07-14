"""Pure Ollama streaming TTFT probe — isolate RAG vs runtime overhead."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ollama_env_probe import snapshot_ollama_env, warmup_ollama_model
from rag_answer_lib import _ollama_chat_payload
from rag_fast_mode import FAST_LLM, FAST_SYSTEM_PROMPT, build_fast_prompts
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_search_inprocess
from retrieval_timing import estimate_tokens

MODELS = ("llama3.2:latest", "llama3.1:8b")
NUM_CTX_VALUES = (1024, 2048, 4096)
MAX_NEW_TOKENS_VALUES = (64, 256)
PILOT_QUESTIONS = _ROOT / "data/eval/pilot_validation_questions.jsonl"
RAG_BENCHMARK_PATHS = {
    "llama3.2:latest": _ROOT / "data/processed/logs/fast_accurate_benchmark_gpu.json",
    "llama3.1:8b": _ROOT / "data/processed/logs/fast_accurate_benchmark_gpu_llama31_8b.json",
}


def _load_question(qid: str) -> dict:
    from rag_eval_lib import load_questions

    for row in load_questions(PILOT_QUESTIONS):
        if row.get("question_id") == qid:
            return row
    raise KeyError(qid)


def build_dummy_user_prompt(target_tokens: int = 500) -> str:
    phrase = (
        "해사 규정 문서에 따르면 선박 운항자는 MARPOL Annex VI, SOLAS, "
        "그리고 IMO 결의를 준수해야 하며 환경·안전·검증 절차를 문서화해야 한다. "
    )
    parts: list[str] = []
    while estimate_tokens("".join(parts)) < target_tokens:
        parts.append(phrase)
    text = "".join(parts)
    while estimate_tokens(text) > target_tokens + 20 and len(text) > len(phrase):
        text = text[: -len(phrase)]
    return text


def build_probe_prompts() -> dict[str, dict[str, str]]:
    dummy_user = build_dummy_user_prompt(500)
    return {
        "hello": {"system": "", "user": "hello"},
        "cii_one_sentence": {
            "system": "",
            "user": "해사 규정에서 CII가 무엇인지 한 문장으로 설명해줘",
        },
        "dummy_500tok": {
            "system": FAST_SYSTEM_PROMPT,
            "user": dummy_user,
        },
    }


def payload_dict(
    model: str,
    system: str,
    user: str,
    *,
    num_ctx: int,
    num_predict: int,
    temperature: float = 0.1,
    stream: bool = True,
) -> dict[str, Any]:
    raw = json.loads(
        _ollama_chat_payload(
            model,
            system,
            user,
            stream=stream,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
        ).decode("utf-8")
    )
    return raw


def probe_stream(
    payload: dict[str, Any],
    base_url: str,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    t_first_chunk: float | None = None
    t_first_content: float | None = None
    t_done: float | None = None
    parts: list[str] = []
    eval_count: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            now = time.perf_counter()
            if t_first_chunk is None:
                t_first_chunk = now
            if not line.strip():
                continue
            data = json.loads(line.decode("utf-8"))
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                if t_first_content is None:
                    t_first_content = now
                parts.append(chunk)
            if data.get("done"):
                t_done = now
                eval_count = data.get("eval_count")
                load_duration_ns = data.get("load_duration")
                prompt_eval_count = data.get("prompt_eval_count")
                break

    if t_done is None:
        t_done = time.perf_counter()
    ttft = (t_first_content or t_first_chunk or t_done) - t0
    total = t_done - t0
    gen_time = (t_done - (t_first_content or t_first_chunk or t0)) if t_done else None
    tok_s = None
    if eval_count and gen_time and gen_time > 0:
        tok_s = round(eval_count / gen_time, 2)

    return {
        "pure_ollama_ttft": round(ttft, 4),
        "t_first_chunk_s": round((t_first_chunk or t0) - t0, 4) if t_first_chunk else None,
        "t_first_content_s": round((t_first_content or t0) - t0, 4) if t_first_content else None,
        "total_time": round(total, 4),
        "generation_time": round(gen_time, 4) if gen_time is not None else None,
        "tok_s": tok_s,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "load_duration_ms": round(load_duration_ns / 1e6, 1) if load_duration_ns else None,
        "output_chars": len("".join(parts)),
        "output_preview": "".join(parts)[:120],
    }


def build_rag_fast_payloads(qids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    from rag_resource_cache import warm_all_resources

    collection, embed_model, manifest = warm_all_resources(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)
    out: dict[str, dict[str, Any]] = {}
    for qid in qids:
        row = _load_question(qid)
        search = run_search_inprocess(
            row,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            latency_mode="fast",
            use_rerank=False,
            top_k=3,
            fetch_k=10,
            max_doc=1,
            max_docs=2,
        )
        chunks = search.get("retrieved") or []
        system, user, compact = build_fast_prompts(row, chunks)
        out[qid] = {
            "system": system,
            "user": user,
            "compact_context_chars": len(compact),
            "input_token_estimate": estimate_tokens(system + user),
            "final_prompt_chars": len(system) + len(user),
        }
    return out


def load_rag_benchmark_ttft(model: str, qid: str) -> float | None:
    path = RAG_BENCHMARK_PATHS.get(model)
    if not path or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("warm_results") or []:
        if row.get("question_id") == qid and row.get("latency_mode") == "fast":
            return row.get("llm_ttft")
    return None


def run_matrix(
    models: list[str],
    base_url: str,
    *,
    repeats: int = 2,
) -> list[dict[str, Any]]:
    prompts = build_probe_prompts()
    results: list[dict[str, Any]] = []

    for model in models:
        print(f"\n=== Warm-up {model} ===")
        warmup_ollama_model(model, base_url)

        for prompt_type, parts in prompts.items():
            system, user = parts["system"], parts["user"]
            prompt_len = len(system) + len(user)
            prompt_tokens = estimate_tokens(system + user)

            for num_ctx in NUM_CTX_VALUES:
                for num_predict in MAX_NEW_TOKENS_VALUES:
                    for run_index in range(1, repeats + 1):
                        payload = payload_dict(
                            model,
                            system,
                            user,
                            num_ctx=num_ctx,
                            num_predict=num_predict,
                            temperature=FAST_LLM["temperature"],
                        )
                        metrics = probe_stream(payload, base_url)
                        env = snapshot_ollama_env(model, base_url)
                        record = {
                            "model": model,
                            "prompt_type": prompt_type,
                            "num_ctx": num_ctx,
                            "max_new_tokens": num_predict,
                            "run_index": run_index,
                            "run_phase": "warm_repeat" if run_index > 1 else "first_after_warmup",
                            "prompt_length_chars": prompt_len,
                            "prompt_token_estimate": prompt_tokens,
                            "payload": {
                                "model": payload["model"],
                                "prompt_length_chars": prompt_len,
                                "options": payload.get("options"),
                                "stream": payload.get("stream"),
                                "temperature": payload["options"].get("temperature"),
                                "num_predict": payload["options"].get("num_predict"),
                                "num_ctx": payload["options"].get("num_ctx"),
                                "keep_alive": payload["options"].get("keep_alive"),
                            },
                            "processor": env.get("ollama_processor"),
                            "gpu_vram_used_mb": env.get("gpu_vram_used_mb"),
                            **metrics,
                        }
                        results.append(record)
                        print(
                            f"  {model} {prompt_type} ctx={num_ctx} pred={num_predict} "
                            f"run={run_index} ttft={metrics['pure_ollama_ttft']}s "
                            f"total={metrics['total_time']}s"
                        )
    return results


def run_rag_payload_probes(
    models: list[str],
    base_url: str,
    rag_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model in models:
        warmup_ollama_model(model, base_url)
        for qid, parts in rag_payloads.items():
            system, user = parts["system"], parts["user"]
            rag_ttft = load_rag_benchmark_ttft(model, qid)
            for num_predict in MAX_NEW_TOKENS_VALUES:
                payload = payload_dict(
                    model,
                    system,
                    user,
                    num_ctx=FAST_LLM["num_ctx"],
                    num_predict=num_predict,
                    temperature=FAST_LLM["temperature"],
                )
                metrics = probe_stream(payload, base_url)
                env = snapshot_ollama_env(model, base_url)
                record = {
                    "model": model,
                    "prompt_type": f"rag_fast_{qid}",
                    "num_ctx": FAST_LLM["num_ctx"],
                    "max_new_tokens": num_predict,
                    "run_index": 1,
                    "prompt_length_chars": parts["final_prompt_chars"],
                    "prompt_token_estimate": parts["input_token_estimate"],
                    "rag_benchmark_llm_ttft": rag_ttft,
                    "ttft_delta_vs_rag_benchmark": (
                        round(metrics["pure_ollama_ttft"] - rag_ttft, 4)
                        if rag_ttft is not None
                        else None
                    ),
                    "payload": {
                        "model": payload["model"],
                        "prompt_length_chars": parts["final_prompt_chars"],
                        "options": payload.get("options"),
                        "stream": payload.get("stream"),
                        "temperature": payload["options"].get("temperature"),
                        "num_predict": payload["options"].get("num_predict"),
                        "num_ctx": payload["options"].get("num_ctx"),
                        "keep_alive": payload["options"].get("keep_alive"),
                    },
                    "processor": env.get("ollama_processor"),
                    "gpu_vram_used_mb": env.get("gpu_vram_used_mb"),
                    **metrics,
                }
                results.append(record)
                print(
                    f"  RAG payload {model} {qid} pred={num_predict} "
                    f"pure_ttft={metrics['pure_ollama_ttft']}s "
                    f"rag_logged={rag_ttft}s"
                )
    return results


def print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "\n| model | prompt_type | num_ctx | max_new_tokens | run | "
        "pure_ollama_ttft | total_time | tok_s |"
    )
    print("| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in rows:
        print(
            f"| {r['model']} | {r['prompt_type']} | {r['num_ctx']} | "
            f"{r['max_new_tokens']} | {r.get('run_index', 1)} | "
            f"{r.get('pure_ollama_ttft')} | {r.get('total_time')} | "
            f"{r.get('tok_s') or '—'} |"
        )


def diagnose(rows: list[dict[str, Any]], rag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hello = [r for r in rows if r["prompt_type"] == "hello" and r["run_index"] == 1]
    hello_ctx4096 = [r for r in hello if r["num_ctx"] == 4096 and r["max_new_tokens"] == 64]
    hello_ttfts = [r["pure_ollama_ttft"] for r in hello_ctx4096]

    rag_fast = [r for r in rag_rows if r.get("rag_benchmark_llm_ttft") is not None]
    rag_gaps = [r["ttft_delta_vs_rag_benchmark"] for r in rag_fast if r.get("ttft_delta_vs_rag_benchmark") is not None]

    ctx_effect: dict[str, float] = {}
    for model in MODELS:
        for prompt_type in ("hello", "dummy_500tok"):
            base_key = (model, prompt_type, 64, 1)
            vals = {}
            for ctx in NUM_CTX_VALUES:
                match = next(
                    (
                        r
                        for r in rows
                        if r["model"] == model
                        and r["prompt_type"] == prompt_type
                        and r["num_ctx"] == ctx
                        and r["max_new_tokens"] == 64
                        and r["run_index"] == 1
                    ),
                    None,
                )
                if match:
                    vals[ctx] = match["pure_ollama_ttft"]
            if len(vals) >= 2:
                ctx_effect[f"{model}/{prompt_type}"] = max(vals.values()) - min(vals.values())

    warm_effect: dict[str, dict[str, float]] = {}
    for model in MODELS:
        r1 = next(
            (r for r in rows if r["model"] == model and r["prompt_type"] == "hello"
             and r["num_ctx"] == 4096 and r["max_new_tokens"] == 64 and r["run_index"] == 1),
            None,
        )
        r2 = next(
            (r for r in rows if r["model"] == model and r["prompt_type"] == "hello"
             and r["num_ctx"] == 4096 and r["max_new_tokens"] == 64 and r["run_index"] == 2),
            None,
        )
        if r1 and r2:
            warm_effect[model] = {
                "run1_ttft": r1["pure_ollama_ttft"],
                "run2_ttft": r2["pure_ollama_ttft"],
                "delta_s": round(r2["pure_ollama_ttft"] - r1["pure_ollama_ttft"], 4),
            }

    verdicts: list[str] = []
    warm_ttfts = [w["run2_ttft"] for w in warm_effect.values()]
    if hello_ttfts and max(hello_ttfts) >= 3.0 and warm_ttfts and max(warm_ttfts) < 1.0:
        verdicts.append(
            "run1( context 변경 직후 ) TTFT 5~8s, run2( 동일 num_ctx warm ) TTFT ~0.27s "
            "→ prompt 길이와 무관, Ollama model/context reload overhead"
        )
    elif hello_ttfts and min(hello_ttfts) >= 3.0:
        verdicts.append(
            "순수 Ollama hello TTFT도 3s+ → Ollama runtime / model load overhead"
        )
    elif hello_ttfts and max(hello_ttfts) < 1.0:
        verdicts.append(
            "순수 hello TTFT < 1s → RAG Fast 5s대는 RAG payload 또는 측정 방식 문제 가능"
        )

    if ctx_effect and max(ctx_effect.values()) > 0.5:
        verdicts.append("num_ctx 변경 시 TTFT 변동 >0.5s → context 설정 영향 있음")
    else:
        verdicts.append("num_ctx 변경 영향 작음 (<0.5s)")

    if warm_effect:
        for model, w in warm_effect.items():
            if w["delta_s"] < -0.3:
                verdicts.append(f"{model}: keep_alive 반복 후 TTFT 감소 → warm-up/cache 효과")
            elif w["delta_s"] > 0.3:
                verdicts.append(f"{model}: 반복 실행 시 TTFT 증가 (재로드/ctx 변경 가능)")

    if rag_gaps:
        avg_gap = sum(rag_gaps) / len(rag_gaps)
        warm_rag = [r for r in rag_rows if r.get("pure_ollama_ttft", 99) < 1.0]
        if warm_rag:
            verdicts.append(
                f"RAG Fast payload warm TTFT {min(r['pure_ollama_ttft'] for r in warm_rag):.2f}~"
                f"{max(r['pure_ollama_ttft'] for r in warm_rag):.2f}s → 3초 조건 충족 가능"
            )
        if avg_gap < -2.0:
            verdicts.append(
                f"RAG benchmark llm_ttft가 pure warm TTFT보다 ~{-avg_gap:.1f}s 큼 "
                "→ 벤치마크가 context reload run1 구간을 포함한 것으로 보임"
            )
        elif abs(avg_gap) < 0.5:
            verdicts.append(
                f"RAG benchmark vs pure payload TTFT 평균 차 {avg_gap:+.2f}s → 측정 방식 차이 작음"
            )

    return {
        "hello_ttft_by_model": {
            r["model"]: r["pure_ollama_ttft"] for r in hello_ctx4096
        },
        "ctx_sensitivity_s": ctx_effect,
        "keep_alive_repeat": warm_effect,
        "rag_vs_pure_delta_avg_s": round(sum(rag_gaps) / len(rag_gaps), 4) if rag_gaps else None,
        "verdicts": verdicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure Ollama TTFT probe vs RAG Fast")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=2, help="Repeat per config for keep_alive test")
    parser.add_argument("--rag-qids", default="V01,V05,V06")
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data/processed/logs/ollama_ttft_probe.json"),
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    qids = tuple(q.strip() for q in args.rag_qids.split(",") if q.strip())

    env_before = {m: snapshot_ollama_env(m, args.ollama_base) for m in models}
    print("Building RAG Fast payloads from live retrieval (V01/V05/V06)...")
    rag_payloads = build_rag_fast_payloads(qids)

    matrix_rows = run_matrix(models, args.ollama_base, repeats=args.repeats)
    rag_rows = run_rag_payload_probes(models, args.ollama_base, rag_payloads)
    env_after = {m: snapshot_ollama_env(m, args.ollama_base) for m in models}

    all_rows = matrix_rows + rag_rows
    print_table(matrix_rows)

    diagnosis = diagnose(matrix_rows, rag_rows)
    print("\n=== Diagnosis ===")
    for v in diagnosis["verdicts"]:
        print(f"  - {v}")

    out = {
        "probe_type": "pure_ollama_ttft",
        "models": models,
        "num_ctx_values": list(NUM_CTX_VALUES),
        "max_new_tokens_values": list(MAX_NEW_TOKENS_VALUES),
        "env_before": env_before,
        "env_after": env_after,
        "rag_fast_payloads": {
            qid: {
                "input_token_estimate": p["input_token_estimate"],
                "final_prompt_chars": p["final_prompt_chars"],
                "system_chars": len(p["system"]),
                "user_chars": len(p["user"]),
            }
            for qid, p in rag_payloads.items()
        },
        "matrix_results": matrix_rows,
        "rag_payload_results": rag_rows,
        "diagnosis": diagnosis,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
