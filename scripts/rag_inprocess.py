"""In-process RAG search/answer (shared by Streamlit UI and timing benchmark)."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_answer_lib import (
    DEFAULT_OLLAMA_BASE,
    DEFAULT_OLLAMA_MODEL,
    RetrievedChunk,
    build_answer_verification,
    check_ollama_model,
    generate_answer,
    load_unified_collection,
    reference_for_question,
    run_retrieval_only,
)
from rag_fast_mode import (
    FAST_RETRIEVAL,
    RULE_GUIDANCE_FAST_RETRIEVAL,
    TABLE_FAST_RETRIEVAL,
    fast_summary_lines,
    generate_fast_answer,
    run_fast_retrieval_only,
)
from ollama_warmup import ensure_fast_warm, ensure_fast_warm_checked, mark_accurate_llm_run
from accurate_streaming import mark_accurate_initial_ack, wrap_accurate_on_token
from retrieval_timing import TimingTrace, populate_timing_meta, set_run_context
from retrieval_verification import append_retrieval_trace_log, serialize_chunk_list

DEFAULT_UNIFIED = "full_corpus_v1"
TABLE_QA_UNIFIED = "kr_tables_v1"
TABLE_QA_ACCURATE = {
    "top_k": 10,
    "fetch_k": 30,
    "max_doc": 5,
    "max_docs": 3,
    "use_rerank": False,
}
DEFAULT_INDEX_DIR = Path("data/processed/index")
DEFAULT_CHUNKS_DIR = Path("data/processed/chunks")
TRACE_LOG = Path("data/processed/logs/pilot_validation/retrieval_trace_ui.jsonl")


def normalize_table_question_row(row: dict) -> dict:
    """Map table_questions.jsonl fields to in-process RAG row shape."""
    out = dict(row)
    out.setdefault("question_id", str(out.get("qid") or out.get("question_id") or ""))
    out.setdefault("category", str(out.get("question_type") or "table_qa"))
    out["_table_qa"] = True
    return out


def is_table_qa_row(row: dict) -> bool:
    return bool(row.get("_table_qa") or row.get("question_type") or row.get("gold_table_id"))


def _init_timing(user_submit_ts: float | None = None) -> TimingTrace:
    timing = TimingTrace()
    if user_submit_ts is not None:
        timing.set_user_click(user_submit_ts)
    return timing


def _finalize_timing(timing: TimingTrace) -> dict:
    return timing.to_log_row()


def _chunk_from_dict(d: dict) -> RetrievedChunk:
    return RetrievedChunk(**d)


def _maybe_collection(
    collection,
    embed_model: str | None,
    manifest: dict | None,
    *,
    unified_id: str,
    index_dir: Path,
    timing: TimingTrace,
):
    if collection is not None and embed_model is not None:
        from rag_resource_cache import apply_cache_flags_to_timing

        apply_cache_flags_to_timing(timing, unified_id, index_dir, embed_model)
        return collection, embed_model, manifest or {}
    timing.mark("t_retrieval_start")
    return load_unified_collection(unified_id, index_dir, timing=timing)


def _attach_rag_debug_trace(
    timing: TimingTrace,
    *,
    row: dict,
    result: dict,
    latency_mode: str,
    top_k: int,
    fetch_k: int,
    unified_id: str,
    llm_params: dict | None = None,
    final_chunks: list | None = None,
) -> None:
    from rag_debug_trace import build_rag_debug_trace, log_debug_trace

    route = row.get("_pipeline_route") or timing.meta.get("_pipeline_route") or {}
    pool = result.get("retrieval_pool") or []
    pool_before = result.get("pool_before_society_filter") or pool
    chunks = final_chunks or result.get("retrieved") or []
    if row.get("_rule_guidance_llm_chunks"):
        chunks = row["_rule_guidance_llm_chunks"]
    where = {"source": row.get("class_society_hint")} if row.get("class_society_hint") else None
    ag = row.get("_answer_generation") or (llm_params or {}).get("answer_generation")
    trace = build_rag_debug_trace(
        run_id=timing.run_id,
        row=row,
        route={**route, "latency_mode": latency_mode},
        pool_before=pool_before,
        pool_after=pool,
        final_chunks=chunks,
        retrieval_params={
            "corpus": unified_id,
            "top_k": top_k,
            "fetch_k": fetch_k,
            "rerank": (result.get("retrieval_config") or {}).get("use_rerank"),
            "max_docs": (result.get("retrieval_config") or {}).get("max_docs"),
            "max_chunks_per_doc": (result.get("retrieval_config") or {}).get("max_chunks_per_doc"),
            "where_filter": where,
        },
        llm_params=llm_params or timing.meta.get("llm_params") or {},
        timing_metrics=timing.compute_metrics() if hasattr(timing, "compute_metrics") else {},
        where_filter=where,
        answer_generation=ag,
    )
    timing.meta["rag_debug_trace"] = trace
    log_debug_trace(trace, run_id=timing.run_id)


def run_search_inprocess(
    row: dict,
    *,
    top_k: int = 10,
    fetch_k: int = 120,
    max_doc: int = 3,
    max_docs: int = 10,
    use_rerank: bool = True,
    eval_constrained: bool = False,
    user_submit_ts: float | None = None,
    unified_id: str = DEFAULT_UNIFIED,
    index_dir: Path | None = None,
    chunks_dir: Path | None = None,
    collection=None,
    embed_model: str | None = None,
    manifest: dict | None = None,
    start_type: str = "warm",
    run_index: int = 1,
    timing: TimingTrace | None = None,
    latency_mode: str = "accurate",
) -> dict[str, Any]:
    from rag_query_router import enrich_row_for_routing, is_rule_guidance_lookup

    row = enrich_row_for_routing(dict(row), latency_mode=latency_mode)
    index_dir = index_dir or DEFAULT_INDEX_DIR
    chunks_dir = chunks_dir or DEFAULT_CHUNKS_DIR
    timing = timing or _init_timing(user_submit_ts)
    set_run_context(timing, start_type=start_type, run_index=run_index)
    timing.meta["latency_mode"] = latency_mode
    timing.meta["_pipeline_route"] = row.get("_pipeline_route")
    if "t_retrieval_start" not in timing.monotonic:
        timing.mark("t_retrieval_start")
    if "t_retrieval_start" not in timing.wall_clock:
        timing.mark_wall("t_retrieval_start")
    collection, embed_model, manifest = _maybe_collection(
        collection,
        embed_model,
        manifest,
        unified_id=unified_id,
        index_dir=index_dir,
        timing=timing,
    )
    if latency_mode == "fast":
        fast_cfg = TABLE_FAST_RETRIEVAL if is_table_qa_row(row) else None
        if fast_cfg is None and is_rule_guidance_lookup(
            str(row.get("question") or ""),
            row,
            category=str(row.get("category") or ""),
        ):
            fast_cfg = RULE_GUIDANCE_FAST_RETRIEVAL
        result = run_fast_retrieval_only(
            row,
            collection,
            embed_model,
            chunks_dir=chunks_dir,
            timing=timing,
            retrieval_cfg=fast_cfg,
            eval_constrained=eval_constrained,
        )
        cfg_used = fast_cfg or FAST_RETRIEVAL
        top_k = cfg_used["top_k"]
        fetch_k = cfg_used["fetch_k"]
        mode = result.get("answer_mode", "fast_rag")
    else:
        acc_top_k = top_k
        acc_fetch_k = fetch_k
        acc_max_doc = max_doc
        acc_max_docs = max_docs
        acc_rerank = use_rerank
        rule_guidance_acc = is_rule_guidance_lookup(
            str(row.get("question") or ""),
            row,
            category=str(row.get("category") or ""),
        )
        if rule_guidance_acc:
            result = run_fast_retrieval_only(
                row,
                collection,
                embed_model,
                chunks_dir=chunks_dir,
                timing=timing,
                retrieval_cfg=RULE_GUIDANCE_FAST_RETRIEVAL,
                eval_constrained=eval_constrained,
            )
            top_k = RULE_GUIDANCE_FAST_RETRIEVAL["top_k"]
            fetch_k = RULE_GUIDANCE_FAST_RETRIEVAL["fetch_k"]
            mode = result.get("answer_mode", "rule_guidance_lookup")
        elif is_table_qa_row(row):
            acc_top_k = TABLE_QA_ACCURATE["top_k"]
            acc_fetch_k = TABLE_QA_ACCURATE["fetch_k"]
            acc_max_doc = TABLE_QA_ACCURATE["max_doc"]
            acc_max_docs = TABLE_QA_ACCURATE["max_docs"]
            acc_rerank = TABLE_QA_ACCURATE["use_rerank"]
        if not rule_guidance_acc:
            result = run_retrieval_only(
                row,
                collection,
                embed_model,
                chunks_dir=chunks_dir,
                top_k=acc_top_k,
                fetch_k=acc_fetch_k,
                use_diversity_rerank=acc_rerank,
                max_chunks_per_doc=acc_max_doc,
                max_docs=acc_max_docs,
                eval_constrained_mode=eval_constrained,
                gold_doc_filter=False if not eval_constrained else None,
                timing=timing,
            )
            top_k = acc_top_k
            fetch_k = acc_fetch_k
            mode = result.get("answer_mode", "standard_rag")
    if hasattr(timing, "mark_wall"):
        timing.mark_wall("t_retrieval_end")
    populate_timing_meta(
        timing,
        row=row,
        mode=mode,
        top_k=top_k,
        fetch_k=fetch_k,
        retrieved=result["retrieved"],
        pool=result.get("retrieval_pool") or [],
        action="search",
    )
    timing.set_cache("llm_server_ready", timing.cache_flags.get("llm_server_ready", False))
    log_row = _finalize_timing(timing)
    summary = result.get("verification_summary") or {}
    summary["timing_metrics"] = log_row["timing_metrics"]
    summary["latency_mode"] = latency_mode
    summary["timing_summary_lines"] = timing.summary_lines()
    if latency_mode == "fast":
        summary["timing_summary_lines"].extend(fast_summary_lines(timing.meta))
    _attach_rag_debug_trace(
        timing,
        row=row,
        result=result,
        latency_mode=latency_mode,
        top_k=top_k,
        fetch_k=fetch_k,
        unified_id=unified_id,
    )
    return {
        "retrieved": result["retrieved"],
        "retrieved_serialized": serialize_chunk_list(result["retrieved"]),
        "retrieval_pool": result.get("retrieval_pool") or [],
        "retrieval_pool_serialized": serialize_chunk_list(result.get("retrieval_pool") or []),
        "retrieval_metrics": result["retrieval_metrics"],
        "retrieval_config": result["retrieval_config"],
        "answer_mode": mode,
        "question_category": result.get("question_category"),
        "question_category_label": result.get("question_category_label"),
        "broad_summary_mode": result.get("broad_summary_mode", False),
        "doc_groups": result.get("doc_groups", []),
        "pipeline_warnings": result.get("pipeline_warnings", []),
        "evidence_table": result["evidence_table"],
        "must_cover_coverage": result["must_cover_coverage"],
        "verification_summary": summary,
        "timing_metrics": log_row["timing_metrics"],
        "timing_log": log_row,
        "embed_model": embed_model,
        "collection": collection,
        "manifest": manifest,
        "table_retrieval_debug": row.get("_table_retrieval_debug"),
        "pool_before_society_filter": result.get("pool_before_society_filter"),
    }


def run_answer_inprocess(
    *,
    row: dict,
    chunks: list[RetrievedChunk],
    pool: list[RetrievedChunk] | None = None,
    config_dict: dict | None = None,
    metrics: dict | None = None,
    doc_groups: list | None = None,
    answer_mode: str = "standard_rag",
    question_category: str | None = None,
    llm_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    temperature: float = 0.15,
    save_trace: bool = False,
    multi_doc_strategy: str = "single_pass",
    max_llm_docs: int = 6,
    top_k: int = 10,
    fetch_k: int = 120,
    user_submit_ts: float | None = None,
    start_type: str = "warm",
    run_index: int = 1,
    timing: TimingTrace | None = None,
    latency_mode: str = "accurate",
    on_token=None,
    auto_llm_warm: bool = True,
    mark_initial_ack: bool = False,
    skip_ollama_probe: bool = False,
) -> dict[str, Any]:
    if latency_mode == "fast" and auto_llm_warm:
        ensure_fast_warm(llm_model, ollama_base, timing=timing)
    pool = pool or chunks
    row = dict(row)
    if question_category:
        row["category"] = question_category
    elif config_dict and config_dict.get("question_category"):
        row["category"] = config_dict["question_category"]
    timing = timing or _init_timing(user_submit_ts)
    set_run_context(timing, start_type=start_type, run_index=run_index)
    timing.meta["latency_mode"] = latency_mode
    if mark_initial_ack and latency_mode == "accurate":
        mark_accurate_initial_ack(timing)
    if latency_mode == "fast":
        if hasattr(timing, "mark_wall"):
            timing.mark_wall("t_pre_llm_start")
        if not auto_llm_warm:
            warm_meta = ensure_fast_warm_checked(
                llm_model,
                ollama_base,
                timing=timing,
                allow_rewarm=False,
            )
            timing.meta.setdefault("rewarm_triggered", warm_meta.get("rewarm_triggered"))
            timing.meta.setdefault("rewarm_reason", warm_meta.get("rewarm_reason"))
        if skip_ollama_probe:
            llm_ok = True
        elif timing.cache_flags.get("llm_server_ready"):
            llm_ok = True
        else:
            llm_ok, _ = check_ollama_model(ollama_base, llm_model)
            timing.set_cache("llm_server_ready", llm_ok)
        if hasattr(timing, "mark_wall"):
            timing.mark_wall("t_ollama_probe_end")
    else:
        if skip_ollama_probe or timing.cache_flags.get("llm_server_ready"):
            llm_ok = bool(timing.cache_flags.get("llm_server_ready", True))
        else:
            llm_ok, _ = check_ollama_model(ollama_base, llm_model)
            timing.set_cache("llm_server_ready", llm_ok)
        if hasattr(timing, "mark_wall"):
            timing.mark_wall("t_ollama_probe_end")
    prompt_meta: dict = {}
    if latency_mode == "fast":
        token_cb = on_token
        if on_token is not None:
            first_rendered = {"done": False}

            def _fast_token_cb(tok: str) -> None:
                if not first_rendered["done"]:
                    timing.mark_first_token_rendered()
                    first_rendered["done"] = True
                on_token(tok)

            token_cb = _fast_token_cb
        answer, prompt_meta = generate_fast_answer(
            row,
            chunks,
            model=llm_model,
            ollama_base=ollama_base,
            timing=timing,
            on_token=token_cb,
            temperature=temperature,
            auto_llm_warm=auto_llm_warm,
            pool=pool,
            fast_meta=(config_dict or {}).get("fast_meta"),
        )
        provider, model = "ollama", llm_model
        verification = {
            "evidence_table": [],
            "answer_citation_mapping": [],
            "must_cover_coverage": [],
            "verification_summary": {
                "answer_mode": prompt_meta.get("answer_mode")
                or ("structured_meeting" if prompt_meta.get("structured_meeting") else "fast_rag"),
                "latency_mode": "fast",
                "final_chunk_count": len(chunks),
            },
        }
        from retrieval_verification import meeting_routing_fields_from_row

        verification["verification_summary"].update(
            meeting_routing_fields_from_row(row, answer=answer)
        )
        timing.mark_wall("t_answer_complete")
    else:
        token_cb = wrap_accurate_on_token(on_token, timing) if on_token else None
        if token_cb is None and latency_mode == "accurate":
            token_cb = wrap_accurate_on_token(None, timing)
        is_rule_guidance = (
            answer_mode == "rule_guidance_lookup"
            or str(row.get("category") or question_category or "") == "rule_lookup"
            or row.get("_rule_guidance_lookup")
        )
        if latency_mode == "accurate" and is_rule_guidance:
            from rule_guidance_accurate import generate_rule_guidance_accurate_answer

            answer, provider, model, gen_meta = generate_rule_guidance_accurate_answer(
                row,
                chunks,
                pool=pool,
                model=llm_model,
                ollama_base=ollama_base,
                timing=timing,
                on_token=token_cb,
                temperature=0.0,
            )
            prompt_meta = {
                "answer_mode": "rule_guidance_lookup",
                "answer_generation": gen_meta,
                "num_ctx": gen_meta.get("llm_num_ctx"),
                "max_new_tokens": gen_meta.get("llm_num_predict"),
                "temperature": gen_meta.get("llm_temperature"),
                "final_prompt_chars": gen_meta.get("llm_prompt_chars"),
                "llm_skipped": not gen_meta.get("llm_used"),
            }
            verification = {
                "evidence_table": [],
                "answer_citation_mapping": [],
                "must_cover_coverage": [],
                "verification_summary": {
                    "answer_mode": "rule_guidance_lookup",
                    "latency_mode": "accurate",
                    "final_chunk_count": len(row.get("_rule_guidance_llm_chunks") or chunks),
                    "answer_source": gen_meta.get("answer_source"),
                },
            }
            from retrieval_verification import meeting_routing_fields_from_row

            verification["verification_summary"].update(
                meeting_routing_fields_from_row(row, answer=answer)
            )
            timing.mark_wall("t_answer_complete")
            if gen_meta.get("llm_used"):
                from ollama_warmup import mark_fast_llm_run
                from rule_guidance_accurate import RULE_GUIDANCE_NUM_CTX

                mark_fast_llm_run(llm_model, RULE_GUIDANCE_NUM_CTX)
            else:
                timing.mark_wall("t_full_report_complete")
                timing.mark_wall("t_evidence_table_complete")
                timing.mark_wall("t_coverage_check_complete")
        else:
            answer, provider, model = generate_answer(
                row,
                chunks,
                provider="ollama",
                model=llm_model,
                ollama_base=ollama_base,
                temperature=temperature,
                allow_extractive_fallback=True,
                reference=reference_for_question(row),
                answer_mode=answer_mode,
                pool=pool,
                category=question_category,
                doc_groups=doc_groups,
                multi_doc_strategy=multi_doc_strategy,
                max_llm_docs=max_llm_docs,
                timing=timing,
                on_token=token_cb,
            )
            timing.mark_wall("t_full_report_complete")
            verification = build_answer_verification(
                row,
                chunks,
                answer,
                config_dict=config_dict,
                pool=pool,
                metrics=metrics,
                doc_groups=doc_groups,
            )
            timing.mark_wall("t_evidence_table_complete")
            timing.mark_wall("t_coverage_check_complete")
            mark_accurate_llm_run(llm_model, ollama_base)
            timing.mark_wall("t_answer_complete")
    if save_trace:
        entry = verification.get("trace", {})
        entry["llm_provider"] = provider
        entry["llm_model"] = model
        entry["answer_mode"] = answer_mode
        append_retrieval_trace_log(TRACE_LOG, entry)
    populate_timing_meta(
        timing,
        row=row,
        mode="fast_rag" if latency_mode == "fast" else answer_mode,
        top_k=top_k,
        fetch_k=fetch_k,
        retrieved=chunks,
        pool=pool,
        answer=answer,
        action="answer",
    )
    log_row = _finalize_timing(timing)
    if latency_mode == "accurate":
        timing.mark_wall("t_all_done")
        log_row = _finalize_timing(timing)
    summary = verification.get("verification_summary") or {}
    summary["timing_metrics"] = log_row["timing_metrics"]
    summary["latency_mode"] = latency_mode
    summary["timing_summary_lines"] = timing.summary_lines()
    if prompt_meta:
        summary["timing_summary_lines"].extend(fast_summary_lines(prompt_meta))
        timing.meta["llm_params"] = {
            "model": llm_model,
            "temperature": prompt_meta.get("temperature") or temperature,
            "num_ctx": prompt_meta.get("num_ctx"),
            "num_predict": prompt_meta.get("max_new_tokens"),
            "keep_alive": "30m",
            "prompt_char_count": prompt_meta.get("final_prompt_chars"),
            "llm_skipped": prompt_meta.get("llm_skipped"),
            "answer_generation": prompt_meta.get("answer_generation") or row.get("_answer_generation"),
        }
    _attach_rag_debug_trace(
        timing,
        row=row,
        result={
            "retrieved": chunks,
            "retrieval_pool": pool,
            "retrieval_config": config_dict or {},
            "pool_before_society_filter": row.get("_pool_before_society_filter"),
        },
        latency_mode=latency_mode,
        top_k=top_k,
        fetch_k=fetch_k,
        unified_id=str(row.get("unified_id") or DEFAULT_UNIFIED),
        llm_params=timing.meta.get("llm_params"),
        final_chunks=chunks,
    )
    return {
        "answer": answer,
        "provider": provider,
        "model": model,
        "prompt_meta": prompt_meta,
        "evidence_table": verification["evidence_table"],
        "answer_citation_mapping": verification["answer_citation_mapping"],
        "must_cover_coverage": verification["must_cover_coverage"],
        "verification_summary": summary,
        "timing_metrics": log_row["timing_metrics"],
        "timing_log": log_row,
    }


def run_full_inprocess(
    row: dict,
    *,
    top_k: int = 10,
    fetch_k: int = 120,
    max_doc: int = 3,
    max_docs: int = 10,
    use_rerank: bool = True,
    eval_constrained: bool = False,
    llm_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    temperature: float = 0.15,
    multi_doc_strategy: str = "single_pass",
    max_llm_docs: int = 6,
    user_submit_ts: float | None = None,
    unified_id: str = DEFAULT_UNIFIED,
    index_dir: Path | None = None,
    chunks_dir: Path | None = None,
    collection=None,
    embed_model: str | None = None,
    manifest: dict | None = None,
    start_type: str = "warm",
    run_index: int = 1,
    skip_llm: bool = False,
    latency_mode: str = "accurate",
    on_token=None,
    auto_llm_warm: bool = True,
    mark_initial_ack: bool = False,
    timing: TimingTrace | None = None,
    skip_ollama_probe: bool = False,
) -> dict[str, Any]:
    if latency_mode == "fast" and auto_llm_warm and not skip_llm:
        ensure_fast_warm(llm_model, ollama_base)
    user_submit_ts = user_submit_ts if user_submit_ts is not None else time.time()
    timing = timing or _init_timing(user_submit_ts)
    if timing.run_id and not timing.wall_clock.get("t_user_click"):
        timing.set_user_click(user_submit_ts)
    if mark_initial_ack and latency_mode == "accurate":
        mark_accurate_initial_ack(timing)
    search_out = run_search_inprocess(
        row,
        top_k=top_k,
        fetch_k=fetch_k,
        max_doc=max_doc,
        max_docs=max_docs,
        use_rerank=use_rerank,
        eval_constrained=eval_constrained,
        user_submit_ts=user_submit_ts,
        unified_id=unified_id,
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        start_type=start_type,
        run_index=run_index,
        timing=timing,
        latency_mode=latency_mode,
    )
    if skip_llm:
        return {
            "question_id": row.get("question_id"),
            "answer_mode": search_out.get("answer_mode"),
            "answer_chars": 0,
            "timing_metrics": search_out["timing_metrics"],
            "timing_log": search_out["timing_log"],
            "timing_summary_lines": timing.summary_lines(),
            "search_out": search_out,
        }
    answer_out = run_answer_inprocess(
        row=row,
        chunks=search_out["retrieved"],
        pool=search_out["retrieval_pool"],
        config_dict=search_out.get("retrieval_config"),
        metrics=search_out.get("retrieval_metrics"),
        doc_groups=search_out.get("doc_groups"),
        answer_mode=search_out.get("answer_mode", "standard_rag"),
        question_category=search_out.get("question_category"),
        llm_model=llm_model,
        ollama_base=ollama_base,
        temperature=temperature,
        save_trace=False,
        multi_doc_strategy=multi_doc_strategy,
        max_llm_docs=max_llm_docs,
        top_k=top_k,
        fetch_k=fetch_k,
        user_submit_ts=user_submit_ts,
        start_type=start_type,
        run_index=run_index,
        timing=timing,
        latency_mode=latency_mode,
        on_token=on_token,
        auto_llm_warm=auto_llm_warm,
        mark_initial_ack=False,
        skip_ollama_probe=skip_ollama_probe,
    )
    populate_timing_meta(
        timing,
        row=row,
        mode="fast_rag" if latency_mode == "fast" else search_out.get("answer_mode", "standard_rag"),
        top_k=top_k,
        fetch_k=fetch_k,
        retrieved=search_out["retrieved"],
        pool=search_out["retrieval_pool"],
        answer=answer_out["answer"],
        action="full_rag",
    )
    log_row = _finalize_timing(timing)
    if latency_mode == "accurate":
        timing.mark_wall("t_all_done")
        log_row = _finalize_timing(timing)
    return {
        "question_id": row.get("question_id"),
        "answer_mode": search_out.get("answer_mode"),
        "answer_chars": len(answer_out["answer"]),
        "answer": answer_out["answer"],
        "timing_metrics": log_row["timing_metrics"],
        "timing_log": log_row,
        "timing_summary_lines": timing.summary_lines(),
        "search_out": search_out,
        "answer_out": answer_out,
    }


def chunks_to_session(chunks: list[RetrievedChunk]) -> list[dict]:
    return [asdict(c) if isinstance(c, RetrievedChunk) else c for c in chunks]


def chunks_from_session(items: list[dict]) -> list[RetrievedChunk]:
    return [_chunk_from_dict(d) for d in items]
