"""Accurate Rule/Guidance: evidence draft + short LLM-grounded Korean answer."""
from __future__ import annotations

import re
from typing import Any, Callable

from rag_answer_lib import RetrievedChunk, call_ollama_chat_timed
from rule_lookup_context import is_crossref_table_chunk, strip_metadata_prefix
from rule_lookup_structured_answer import build_rule_lookup_structured_answer
from rag_society_filter import filter_pool_for_society

MAX_LLM_CHUNKS = 3
MAX_CHUNK_CHARS = 600
MAX_TOTAL_CONTEXT_CHARS = 1800
RULE_GUIDANCE_NUM_CTX = 2048
ACCURATE_NUM_CTX = RULE_GUIDANCE_NUM_CTX
ACCURATE_NUM_PREDICT = 120
ACCURATE_TEMPERATURE = 0.0
KEEP_ALIVE = "30m"

RULE_GUIDANCE_SYSTEM_PROMPT = """해사 규정 검색 보조자. 제공 근거만 사용. 한국어 4섹션만 출력.

결론:
근거 조항:
실무 영향:
추가 확인:"""

FORBIDDEN_SOCIETIES = ("KR", "DNV", "ABS", "MEPC", "MSC")


def _is_substantive_chunk(c: Any) -> bool:
    if getattr(c, "is_catalog_table", False) or is_crossref_table_chunk(c):
        return False
    body = strip_metadata_prefix(getattr(c, "text", "") or "")
    return len(body.strip()) >= 70


def filter_evidence_chunks(
    chunks: list[Any],
    society: str,
    *,
    hard: bool = True,
) -> list[Any]:
    pool = list(chunks)
    if society:
        pool, had = filter_pool_for_society(pool, society, hard=hard)
        if hard and not had:
            return []
    return [c for c in pool if _is_substantive_chunk(c)]


def trim_chunks_for_llm(chunks: list[Any]) -> tuple[list[Any], str]:
    """Return up to 3 chunks with per-chunk and total char caps."""
    selected: list[Any] = []
    blocks: list[str] = []
    total = 0
    for i, c in enumerate(chunks[:MAX_LLM_CHUNKS], start=1):
        body = strip_metadata_prefix(getattr(c, "text", "") or "")
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) > MAX_CHUNK_CHARS:
            body = body[: MAX_CHUNK_CHARS - 1] + "…"
        block = (
            f"[{i}] society={getattr(c, 'source', '')} | "
            f"doc={getattr(c, 'file_name', '') or getattr(c, 'doc_id', '')} | "
            f"p{getattr(c, 'page_number', '?')} | "
            f"clause={getattr(c, 'clause_number', '') or '—'}\n"
            f"{body}"
        )
        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            remain = MAX_TOTAL_CONTEXT_CHARS - total
            if remain < 120:
                break
            block = block[: remain - 1] + "…"
        blocks.append(block)
        selected.append(c)
        total += len(block)
        if total >= MAX_TOTAL_CONTEXT_CHARS:
            break
    return selected, "\n\n".join(blocks)


def build_compact_evidence_draft(chunks: list[Any], society: str) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks[:MAX_LLM_CHUNKS], start=1):
        body = strip_metadata_prefix(getattr(c, "text", "") or "")
        body = re.sub(r"\s+", " ", body).strip()[:220]
        fn = getattr(c, "file_name", "") or getattr(c, "doc_id", "")
        pg = getattr(c, "page_number", "?")
        cl = getattr(c, "clause_number", "") or "—"
        lines.append(f"- [{i}] {society} {fn} p{pg} clause {cl}: {body}")
    return "\n".join(lines) if lines else "- (근거 없음)"


def ensure_rule_guidance_warm(
    model: str,
    ollama_base: str,
    *,
    timing=None,
) -> dict[str, Any]:
    from ollama_warmup import ensure_fast_warm_checked

    if timing is not None and hasattr(timing, "mark_wall"):
        timing.mark_wall("t_pre_llm_start")
    result = ensure_fast_warm_checked(
        model,
        ollama_base,
        timing=timing,
        allow_rewarm=True,
        num_ctx=RULE_GUIDANCE_NUM_CTX,
    )
    if timing is not None and hasattr(timing, "mark_wall") and "t_ollama_probe_end" not in timing.wall_clock:
        timing.mark_wall("t_ollama_probe_end")
    return result


def build_rule_guidance_user_prompt(
    *,
    question: str,
    society: str,
    evidence_draft: str,
    evidence_block: str,
) -> str:
    return f"""질문: {question}
society: {society or '—'}

draft:
{evidence_draft}

chunks:
{evidence_block}

위 근거만으로 4섹션 짧게 답하라."""


def fallback_no_evidence_answer(society: str) -> str:
    soc = society or "해당 선급"
    headline = f"{soc} 근거 부족" if soc else "근거 부족"
    return f"""결론:
- {headline}. {soc} Rule/Guidance 검색 결과에서 질문과 직접 연결되는 근거를 찾지 못했습니다. 다른 선급 문서로 대체하지 않았습니다.

근거 조항:
- 없음

실무 영향:
- {soc} 원문 Rule/Guidance 확인이 필요합니다.

추가 확인:
- {soc} 문서명·Section·조항을 지정해 재검색하세요."""


def llm_grounded_check_pass(answer: str, chunks: list[Any], society: str) -> bool:
    if not answer or not chunks:
        return False
    for soc in FORBIDDEN_SOCIETIES:
        if soc != (society or "").upper() and re.search(rf"\b{soc}\b", answer, re.I):
            if soc != society.upper():
                return False
    if society and society.upper() not in answer.upper():
        fn = getattr(chunks[0], "file_name", "") or ""
        if society.upper() not in fn.upper() and not re.search(r"\[\d+\]", answer):
            pass
    required = ("결론", "근거", "실무", "추가")
    return all(k in answer for k in required)


def _fabricated_query_markers(question: str) -> list[str]:
    markers: list[str] = []
    m = re.search(r"존재하지\s*않는\s+(\S+)", question)
    if m:
        markers.append(m.group(1).strip(".,;"))
    markers.extend(re.findall(r"\bXYZ[-\w]*\d*\b", question, re.I))
    return [x for x in markers if x]


def _markers_absent_from_chunks(markers: list[str], chunks: list[Any]) -> bool:
    if not markers:
        return False
    corpus = " ".join(
        strip_metadata_prefix(getattr(c, "text", "") or "") for c in chunks
    ).lower()
    fnames = " ".join(
        str(getattr(c, "file_name", "") or getattr(c, "doc_id", "") or "") for c in chunks
    ).lower()
    blob = corpus + " " + fnames
    return any(m.lower() not in blob for m in markers)


def generate_rule_guidance_accurate_answer(
    row: dict,
    retrieved: list[RetrievedChunk],
    *,
    pool: list[RetrievedChunk] | None = None,
    model: str,
    ollama_base: str,
    timing=None,
    on_token: Callable[[str], None] | None = None,
    temperature: float = ACCURATE_TEMPERATURE,
) -> tuple[str, str, str, dict[str, Any]]:
    """
    Accurate rule guidance: structured draft + short LLM summary.
    Returns (answer, provider, model_name, answer_generation_meta).
    """
    question = str(row.get("question") or "")
    society = str(row.get("class_society_hint") or "")
    pool = pool or retrieved
    gen_meta: dict[str, Any] = {
        "answer_source": "fallback_no_evidence",
        "llm_used": False,
        "llm_call_function": None,
        "llm_prompt_chars": 0,
        "llm_context_chunks": 0,
        "llm_output_chars": 0,
        "llm_grounded_check_pass": False,
        "fallback_reason": None,
    }

    evidence_chunks = filter_evidence_chunks(retrieved or [], society, hard=True)
    if not evidence_chunks:
        evidence_chunks = filter_evidence_chunks(pool or [], society, hard=True)

    if not evidence_chunks:
        gen_meta["fallback_reason"] = "society_evidence_insufficient"
        answer = fallback_no_evidence_answer(society)
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    markers = _fabricated_query_markers(question)
    if _markers_absent_from_chunks(markers, evidence_chunks):
        gen_meta["fallback_reason"] = "query_terms_not_in_evidence"
        answer = fallback_no_evidence_answer(society)
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    warnings = list(row.get("warning_flags") or [])
    evidence_draft = build_compact_evidence_draft(evidence_chunks, society)
    try:
        structured_draft, ans_warnings = build_rule_lookup_structured_answer(
            evidence_chunks,
            question=question,
            pool=pool,
            warning_flags=warnings,
        )
        row["warning_flags"] = list(dict.fromkeys(warnings + ans_warnings))
        if structured_draft:
            evidence_draft = structured_draft[:600]
    except Exception:
        pass

    llm_chunks, evidence_block = trim_chunks_for_llm(evidence_chunks)
    if not llm_chunks or not evidence_block.strip():
        gen_meta["fallback_reason"] = "no_substantive_chunks_for_llm"
        answer = fallback_no_evidence_answer(society)
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    draft_budget = max(280, MAX_TOTAL_CONTEXT_CHARS - len(evidence_block) - 60)
    evidence_draft_trim = evidence_draft[:draft_budget]
    if len(evidence_draft) > draft_budget:
        evidence_draft_trim = evidence_draft_trim.rstrip() + "…"

    user = build_rule_guidance_user_prompt(
        question=question,
        society=society,
        evidence_draft=evidence_draft_trim,
        evidence_block=evidence_block,
    )
    system = RULE_GUIDANCE_SYSTEM_PROMPT
    prompt_chars = len(system) + len(user)

    ensure_rule_guidance_warm(model, ollama_base, timing=timing)

    gen_meta.update(
        {
            "answer_source": "llm_grounded_summary",
            "llm_used": True,
            "llm_call_function": "call_ollama_chat_timed",
            "llm_prompt_chars": prompt_chars,
            "llm_context_chunks": len(llm_chunks),
            "llm_num_ctx": ACCURATE_NUM_CTX,
            "llm_num_predict": ACCURATE_NUM_PREDICT,
            "llm_temperature": temperature,
            "keep_alive": KEEP_ALIVE,
        }
    )

    if timing is not None and hasattr(timing, "mark_wall"):
        timing.mark_wall("t_prompt_build_end")
        timing.mark_wall("t_accurate_llm_request_start")

    answer = call_ollama_chat_timed(
        model,
        system,
        user,
        ollama_base,
        temperature=temperature,
        num_predict=ACCURATE_NUM_PREDICT,
        num_ctx=ACCURATE_NUM_CTX,
        timing=timing,
        on_token=on_token,
    )
    gen_meta["llm_output_chars"] = len(answer or "")
    gen_meta["llm_grounded_check_pass"] = llm_grounded_check_pass(answer, llm_chunks, society)
    if timing is not None and hasattr(timing, "monotonic"):
        mono = timing.monotonic
        t_req = mono.get("t_llm_request_start")
        t_tok = mono.get("t_first_token")
        t_ret = mono.get("t_retrieval_end") or mono.get("t_context_build_end")
        pre = max(0.0, (t_req - t_ret)) if t_req and t_ret else 0.0
        ttft = max(0.0, (t_tok - t_req)) if t_tok and t_req else 0.0
        if ttft:
            combined = round(pre + ttft, 4)
            gen_meta["rule_guidance_first_token_latency"] = combined
            gen_meta["rule_guidance_first_token_3s_pass"] = combined <= 3.0
    row["_answer_generation"] = gen_meta
    row["_rule_guidance_llm_chunks"] = llm_chunks
    row["_rule_guidance_skip_heavy_postprocess"] = True

    return answer, "ollama", model, gen_meta
