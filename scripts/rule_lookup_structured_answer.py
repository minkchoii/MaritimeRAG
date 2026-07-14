"""Rule/Guidance lookup: 4-section answer with confirmed/candidate grouping."""
from __future__ import annotations

import re
from typing import Any

from answer_depth_guidance import join_four_sections
from bm25_index import extract_document_codes
from rule_lookup_context import (
    doc_code_in_corpus,
    is_crossref_table_chunk,
    strip_metadata_prefix,
)
from rule_lookup_document_analysis import (
    AUTONOMOUS_QUERY_RE,
    ENGLISH_SENTENCE_RE,
    DocAnalysis,
    analyze_documents,
    format_citations,
)
from rule_lookup_presentation import (
    CandidateGroup,
    RuleLookupPresentation,
    build_presentation,
    confirmed_relevance_ko,
    society_from_question,
)
from rule_lookup_alt_fuel import (
    ClauseTheme,
    compute_alt_fuel_must_cover,
    extract_clause_themes,
    format_theme_citations,
    format_theme_grounds,
    is_alt_fuel_question,
    priority_section4_themes,
    select_alt_fuel_work_chunks,
)

ENGLISH_LEAK_RE = re.compile(r"[A-Za-z][A-Za-z\s,;:'\"()-]{50,}")


def _is_clause_reference(code: str) -> bool:
    return bool(re.match(r"^Section\s+\d+", code.strip(), re.I))


def detect_doc_name_mismatches(chunks: list[Any]) -> list[str]:
    warnings: list[str] = []
    for c in chunks:
        if getattr(c, "is_catalog_table", False) or is_crossref_table_chunk(c):
            continue
        fn = str(getattr(c, "file_name", "") or "")
        if not fn:
            continue
        body = strip_metadata_prefix(getattr(c, "text", ""))
        for code in extract_document_codes(body):
            if _is_clause_reference(code):
                continue
            if not doc_code_in_corpus(code, {fn}):
                warnings.append(f"doc_name_mismatch: 본문 '{code}' ↔ file_name '{fn}'")
    return list(dict.fromkeys(warnings))


def _collect_catalog_candidates(chunks: list[Any]) -> list[str]:
    seen_files = {str(getattr(c, "file_name", "")) for c in chunks}
    out: list[str] = []
    for c in chunks:
        if not (getattr(c, "is_catalog_table", False) or is_crossref_table_chunk(c)):
            continue
        for code in getattr(c, "catalog_doc_candidates", []) or extract_document_codes(
            strip_metadata_prefix(getattr(c, "text", ""))
        ):
            if doc_code_in_corpus(code, seen_files):
                continue
            if code not in out:
                out.append(code)
    return out


def _format_grounds(d: DocAnalysis) -> str:
    page = f"p.{d.page}" if d.page is not None else "p.?"
    cite = format_citations(d.citation_ids) if d.citation_ids else ""
    parts = [d.file_name, page]
    if cite:
        parts.append(cite)
    return ", ".join(parts)


def _format_group_grounds(g: CandidateGroup) -> str:
    names = ", ".join(g.file_names[:3])
    if len(g.file_names) > 3:
        names += " 등"
    cite = format_citations(g.citation_ids) if g.citation_ids else ""
    return f"{names}{(' ' + cite) if cite else ''}"


def _section1(pres: RuleLookupPresentation) -> str:
    lines: list[str] = []
    society = pres.society or "해당 선급"

    if pres.clause_themes:
        for theme in pres.clause_themes[:2]:
            cite = format_theme_citations(theme)
            lines.append(f"- **{theme.title}**: {theme.content_ko}. {cite}")
        if len(pres.clause_themes) > 2:
            extra = pres.clause_themes[2]
            cite = format_theme_citations(extra)
            lines.append(f"- {extra.title}: {extra.content_ko}. {cite}")
        return "\n".join(lines[:3])

    for d in pres.confirmed[:2]:
        cite = format_citations(d.citation_ids) if d.citation_ids else ""
        if "cg-0264" in d.doc_code.lower() and society == "DNV":
            lines.append(
                f"- {society}의 자율운항·원격운항 선박 관련 핵심 Guidance는 **{d.doc_code}**로 확인됩니다. "
                f"해당 문서는 자율·원격운항 선박의 scope, autonomy level, notation, "
                f"class 승인 및 검증 요건을 다룹니다. {cite}"
            )
        else:
            lines.append(
                f"- {society} 관련 핵심 Rule/Guidance는 **{d.doc_code}**({d.doc_type})로 확인됩니다. "
                f"{d.summary_ko} {cite}"
            )

    if pres.candidate_groups:
        g = pres.candidate_groups[0]
        cite = format_citations(g.citation_ids) if g.citation_ids else ""
        if g.group_id in ("ru_ou_negative", "ru_ou_family"):
            lines.append(
                "- Smart notation 관련 **DNV-RU-OU** 계열 문서도 검색되었으나, "
                f"자율·원격운항 자산 적용 제외 문구가 있어 확정 Rule로 보기 어렵습니다. {cite}"
            )
        else:
            codes = "/".join(g.doc_codes[:4])
            if len(g.doc_codes) > 4:
                codes += " 등"
            lines.append(
                f"- {codes}도 검색되었으나, {g.relevance_ko.rstrip('.')}. {cite}"
            )
    elif not lines:
        lines.append("- 검색된 Rule/Guidance 본문이 없습니다.")

    return "\n".join(lines[:3])


def _section2(pres: RuleLookupPresentation) -> str:
    lines: list[str] = []
    autonomous_q = bool(AUTONOMOUS_QUERY_RE.search(pres.question))

    if pres.clause_themes:
        lines.append(
            "- 저인화점·대체연료 사용 선박은 연료 저장·공급 계통, 기관(crankcase) 안전성 평가, "
            "dual fuel arrangement를 설계·승인·survey 절차에 반영해야 합니다."
        )
        cite = format_theme_citations(pres.clause_themes[0]) if pres.clause_themes else ""
        if cite:
            lines[0] = lines[0].rstrip(".") + f". {cite}"
        return lines[0]

    if pres.confirmed:
        d = pres.confirmed[0]
        cite = format_citations(d.citation_ids[:1]) if d.citation_ids else ""
        if autonomous_q:
            lines.append(
                f"- 자율운항 또는 원격운항 선박을 검토할 경우, **{d.doc_code}**의 적용 범위, notation, "
                f"class 승인·검증 절차를 프로젝트 요건과 대조해야 합니다. {cite}"
            )
        else:
            lines.append(
                f"- **{d.doc_code}**의 적용 범위·요건을 설계·승인·운항 프로세스에 반영해야 합니다. {cite}"
            )
    if pres.candidate_groups and autonomous_q:
        lines.append(
            "- Smart notation 적용 여부는 선박 유형과 자율운항 기능 적용 범위에 따라 추가 검토가 필요합니다."
        )
    if not lines:
        lines.append("- 검색된 Rule/Guidance의 적용 범위·승인 절차를 프로젝트별로 대조·검토해야 합니다.")
    return "\n".join(lines[:2])


def _section3(pres: RuleLookupPresentation) -> str:
    lines: list[str] = []
    autonomous_q = bool(AUTONOMOUS_QUERY_RE.search(pres.question))

    if pres.clause_themes:
        amend = next((t for t in pres.clause_themes if t.theme_id == "notice_2025_amendment"), None)
        if amend:
            cite = format_theme_citations(amend)
            lines.append(
                f"- [추가 확인 필요] 2025 Notice 개정·적용 시점 및 IMO IGF/IGC Code와의 정합 여부를 "
                f"프로젝트별로 확인해야 합니다. {cite}"
            )
        else:
            lines.append(
                "- [추가 확인 필요] LR Notice 개정(2025)과 IMO IGF/IGC Code 정합 여부는 본문 추가 검색이 필요합니다."
            )
        lines.append(
            "- [해석 근거] 검색된 조항은 **관련 조항 후보**이며, 적용 선종·연료 종류별 세부 요건은 "
            "Notice No.1 해당 Section 본문 대조가 필요합니다."
        )
        return "\n".join(lines[:3])

    if pres.candidate_groups:
        g = pres.candidate_groups[0]
        codes = "/".join(g.doc_codes[:4])
        if len(g.doc_codes) > 4:
            codes += " 등"
        if g.group_id in ("ru_ou_negative", "ru_ou_family"):
            lines.append(
                f"- [미확정 규제] {codes}는 Smart notation 관련 후보로 검색되었으나, "
                "자율·원격운항 자산 적용 제외 문구가 있어 직접 적용 가능 여부를 추가 확인해야 합니다."
            )
        else:
            lines.append(f"- [추가 확인 필요] {g.label}: {g.reason_ko}")

    for code in pres.catalog_codes[:1]:
        lines.append(
            f"- [미확정 규제] **{code}** 등 catalog 표 후보 — file_name 본문 미검색."
        )

    if pres.confirmed and autonomous_q:
        primary = pres.confirmed[0].doc_code
        lines.append(
            f"- [해석 근거] {primary}와 AROS class notation 또는 DNV-RU-SHIP 관련 조항의 "
            "연결 여부도 추가 확인이 필요합니다."
        )

    if not lines:
        lines.append(
            "- [해석 근거] 검색 본문에 없는 문서번호·조항은 답변에 포함하지 않았으며, "
            "추가 Rule은 해당 file_name 본문 검색이 필요합니다."
        )
    return "\n".join(lines[:4])


def _section4(pres: RuleLookupPresentation) -> str:
    """Primary section: confirmed + candidate group, or clause themes for alt-fuel."""
    lines: list[str] = []
    n = 0

    if pres.clause_themes:
        for theme in priority_section4_themes(pres.clause_themes):
            n += 1
            lines.extend(
                [
                    f"{n}. {theme.title}",
                    f"- 유형: {theme.doc_type}",
                    f"- 관련성: {theme.relevance_ko}",
                    f"- 주요 내용: {theme.content_ko}",
                    f"- 근거: {format_theme_grounds(theme)}",
                    f"- 확정 여부: {theme.confirmation}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    for d in pres.confirmed[:2]:
        n += 1
        dtype = d.doc_type if d.doc_type not in {"", "Unknown"} else "Rule/Guidance"
        lines.extend(
            [
                f"{n}. {d.doc_code}",
                f"- 유형: {dtype}",
                f"- 관련성: {confirmed_relevance_ko(d, pres.question)}",
                f"- 근거: {_format_grounds(d)}",
                f"- 확정 여부: 확정",
                "",
            ]
        )

    for g in pres.candidate_groups[:1]:
        n += 1
        lines.extend(
            [
                f"{n}. {g.label}",
                f"- 유형: {g.doc_type}",
                f"- 관련성: {g.relevance_ko}",
                f"- 근거: {_format_group_grounds(g)}",
                f"- 확정 여부: {g.confirmation}",
                "",
            ]
        )

    return "\n".join(lines).strip() if lines else "- 해당 없음"


def _strip_english_leaks(text: str) -> str:
    return ENGLISH_LEAK_RE.sub("", text)


def expand_rule_lookup_chunks(
    retrieved: list[Any],
    pool: list[Any] | None = None,
    *,
    max_files: int = 8,
    question: str = "",
) -> list[Any]:
    source = list(pool or retrieved)
    if not source:
        return retrieved

    if is_alt_fuel_question(question):
        return select_alt_fuel_work_chunks(source, retrieved, question=question)

    best_by_file: dict[str, Any] = {}
    file_order: list[str] = []
    for c in source:
        fn = str(getattr(c, "file_name", "") or "")
        if not fn:
            continue
        if fn not in file_order:
            file_order.append(fn)
        prev = best_by_file.get(fn)
        cur_len = len(strip_metadata_prefix(getattr(c, "text", "")))
        if prev is None or cur_len > len(strip_metadata_prefix(getattr(prev, "text", ""))):
            best_by_file[fn] = c
    out = [best_by_file[fn] for fn in file_order if fn in best_by_file][:max_files]
    return out if out else retrieved


def build_rule_lookup_structured_answer(
    chunks: list[Any],
    *,
    question: str = "",
    pool: list[Any] | None = None,
    warning_flags: list[str] | None = None,
) -> tuple[str, list[str]]:
    warnings = list(warning_flags or [])
    warnings.extend(detect_doc_name_mismatches(chunks))

    work_chunks = expand_rule_lookup_chunks(chunks, pool, question=question)
    catalog = _collect_catalog_candidates(work_chunks)
    clause_themes = extract_clause_themes(
        list(pool or chunks),
        question=question,
        citation_chunks=chunks,
        citation_fallback=pool,
    )
    analyses = analyze_documents(
        work_chunks,
        question=question,
        citation_chunks=chunks,
        citation_fallback=pool,
        max_docs=8,
    )
    pres = build_presentation(
        analyses,
        question=question,
        catalog_codes=catalog,
        clause_themes=clause_themes,
    )

    answer = join_four_sections(
        {
            "1": _section1(pres),
            "2": _section2(pres),
            "3": _section3(pres),
            "4": _section4(pres),
        }
    )
    answer = _strip_english_leaks(answer)

    warnings.extend(pres.warnings)
    if clause_themes:
        must_rows = compute_alt_fuel_must_cover(list(pool or chunks), answer)
        missing = [
            r["must_cover"]
            for r in must_rows
            if r["found_in_chunks"] == "Yes" and r["included_in_answer"] == "No"
        ]
        if missing:
            warnings.append("must_cover_gap")
    if ENGLISH_SENTENCE_RE.search(answer):
        warnings.append("raw_chunk_leak")
    cite_count = len(re.findall(r"\[\d+\]", answer))
    if cite_count > 8:
        warnings.append("too_many_citations")
    if not chunks:
        warnings.append("no_substantive_rule_chunks")

    return answer, list(dict.fromkeys(warnings))


def _legacy_build_section1(chunks: list[Any], *, max_bullets: int = 3) -> str:
    substantive = [
        c
        for c in chunks
        if not getattr(c, "is_catalog_table", False) and not is_crossref_table_chunk(c)
    ]
    by_file: dict[str, Any] = {}
    for c in substantive:
        fn = str(getattr(c, "file_name", "") or "")
        if fn:
            by_file[fn] = c
    bullets: list[str] = []
    for fn in sorted(by_file)[:max_bullets]:
        c = by_file[fn]
        body = strip_metadata_prefix(getattr(c, "text", ""))
        body = re.sub(r"\s+", " ", body).strip()[:220]
        ids = [f"[{i}]" for i, x in enumerate(chunks, 1) if getattr(x, "file_name", "") == fn]
        bullets.append(f"- **{fn.replace('.pdf', '')}**: {body} {''.join(ids)}")
    return "\n".join(bullets) if bullets else "- 없음"


def build_rule_lookup_legacy_answer(chunks: list[Any]) -> str:
    from rule_lookup_answer import build_deterministic_section4, build_fallback_section2

    s1 = _legacy_build_section1(chunks)
    s2 = build_fallback_section2(chunks)
    s3 = "- [해석 근거] (legacy)"
    s4 = build_deterministic_section4(chunks, s1)
    return join_four_sections({"1": s1, "2": s2, "3": s3, "4": s4})


def build_rule_lookup_ungrouped_answer(
    chunks: list[Any],
    *,
    question: str = "",
    pool: list[Any] | None = None,
) -> str:
    """Previous multi-item listing (for before/after comparison)."""
    from rule_lookup_document_analysis import detect_answer_quality_warnings

    work = expand_rule_lookup_chunks(chunks, pool, question=question)
    analyses = analyze_documents(work, question=question, citation_chunks=chunks, citation_fallback=pool, max_docs=6)
    lines_s4: list[str] = []
    for n, d in enumerate(analyses[:5], 1):
        cite = format_citations(d.citation_ids) if d.citation_ids else ""
        lines_s4.extend(
            [
                f"{n}. {d.doc_code}",
                f"- 유형: {d.doc_type}",
                f"- 관련성: {d.relevance_ko}",
                f"- 근거: {d.file_name}, {cite}",
                f"- 확정 여부: {d.confirmation}",
                "",
            ]
        )
    s1 = "\n".join(
        f"- **{d.doc_code}** ({d.confirmation}): {d.summary_ko} {format_citations(d.citation_ids)}"
        for d in analyses[:3]
    )
    return join_four_sections({"1": s1, "2": "- (ungrouped)", "3": "- (ungrouped)", "4": "\n".join(lines_s4)})
