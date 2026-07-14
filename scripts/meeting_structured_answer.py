"""Evidence-based 4-section answers for meeting/regulation categories."""
from __future__ import annotations

import re
from typing import Any

from answer_depth_guidance import join_four_sections, category_bullet_budget
from meeting_answer_dedup import apply_answer_dedup, detect_section1_topics
from meeting_category_profile import MeetingRetrievalProfile, TOP_LEVEL_AUTO, TOP_LEVEL_ENV, TOP_LEVEL_TREND
from meeting_coverage_check import run_coverage_check
from meeting_topic_cluster import (
    MSC_OUTCOME_TOPIC_PRIORITY,
    cluster_chunks,
    dedupe_page_chunks,
    pick_diverse_topic_chunks,
    outcome_topic_id,
    _topic_id_for_text,
    _topic_label_ko,
)
from meeting_topic_scoring import (
    intent_chunk_adjustment,
    is_excluded_chunk,
    exclude_topics_for_intent,
    topic_caps_for_intent,
    NON_MAND_RE,
    TIMELINE_RE,
    MASS_RE,
    ALT_FUEL_RE,
    OUTCOME_RE,
)
from source_tier_lib import (
    classify_source_tier,
    count_impact_signals,
    count_outcome_signals,
    tier_label,
)

ENGLISH_LEAK_RE = re.compile(r"[A-Za-z][A-Za-z\s,;:'\"()-]{55,}")
CITATION_RE = re.compile(r"\[(\d+)\]")


def _strip_meta(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if re.match(r"^(source|file_name|page|doc_id)\s*:", line.strip(), re.I):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _cite(chunk: Any, citation_map: dict[str, int]) -> str:
    cid = str(getattr(chunk, "chunk_id", "") or "")
    n = citation_map.get(cid)
    return f"[{n}]" if n else ""


def _build_citation_map(chunks: list[Any]) -> dict[str, int]:
    return {str(getattr(c, "chunk_id", "")): i for i, c in enumerate(chunks, 1)}


def score_chunk(chunk: Any, *, profile: MeetingRetrievalProfile) -> float:
    text = _strip_meta(getattr(chunk, "text", ""))
    tier = classify_source_tier(chunk)
    score = 0.0
    score += {0: 3.0, 1: 1.5, 2: 0.5, 3: -1.0}.get(tier, 1.0)
    score += count_outcome_signals(text) * 0.4
    if profile.top_level_category == TOP_LEVEL_ENV:
        score += count_impact_signals(text) * 0.45
    score += intent_chunk_adjustment(text, internal_intent=profile.internal_intent)
    if profile.answer_variant == "official_dense":
        score += {0: 2.5, 1: 1.2, 2: -0.8, 3: -2.5}.get(tier, 0.0)
    elif profile.answer_variant == "topic_diverse":
        score += count_outcome_signals(text) * 0.55
        score += count_impact_signals(text) * 0.25
    if getattr(chunk, "bm25_score", None):
        score += float(chunk.bm25_score) * 0.02
    if getattr(chunk, "dense_score", None):
        score += float(chunk.dense_score) * 0.5
    if getattr(chunk, "rrf_score", None):
        score += float(chunk.rrf_score) * 8.0
    setattr(chunk, "_meeting_score", score)
    setattr(chunk, "_source_tier", tier)
    setattr(chunk, "_topic_id", _topic_id_for_text(text))
    return score


REFERENCE_OUTCOME_RE = re.compile(
    r"outcome of (?:a|c|tc|mepc)\s*\d|decisions of other imo|already adopted at a\s*\d|"
    r"upon the recommendation of the (?:assembly|council|technical)|\ba\s*\d{1,3}\b.{0,50}\bapproved",
    re.I,
)
OUTCOME_ACTION_RE = re.compile(
    r"\b(adopted|approved|endorsed|agreed|finalized|finalised|mandatory|non-mandatory|"
    r"entry into force|entered into force|amendment|resolution|guideline)\b",
    re.I,
)
RESOLUTION_REF_RE = re.compile(
    r"\b(?:resolution|res\.?)\s+([A-Z]{2,5}\.\d+\(\d+\)|MSC\.\d+\(\d+\)|MEPC\.\d+\(\d+\)[^,\.;]*)",
    re.I,
)


def _truncate_snippet(text: str, max_len: int = 180) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_len:
        return t
    cut = t[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def _extract_best_claim(body: str) -> str:
    """Pick the highest-signal outcome sentence from chunk text."""
    candidates: list[tuple[float, str]] = []
    for sent in re.split(r"(?<=[.!?])\s+", body):
        sent = sent.strip()
        if len(sent) < 20:
            continue
        score = count_outcome_signals(sent) * 2.0
        if OUTCOME_ACTION_RE.search(sent):
            score += 4.0
        if re.search(r"\b(noted|invited|recalled)\b", sent, re.I) and score < 4:
            score -= 2.0
        if score > 0:
            candidates.append((score, sent))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    lines = [ln.strip() for ln in body.splitlines() if len(ln.strip()) > 35]
    if lines:
        return max(lines, key=len)
    paras = [p.strip() for p in re.split(r"\n{2,}", body) if len(p.strip()) > 40]
    if paras:
        return paras[0]
    return body.strip()


def _shorten_object_phrase(phrase: str, *, max_len: int = 90) -> str:
    obj = re.sub(r"\s+", " ", phrase.strip(" ,;"))
    obj = re.sub(r"^(?:the|a|an)\s+", "", obj, flags=re.I)
    replacements = [
        ("non-mandatory and goal-based", "비강제·goal-based"),
        ("non-mandatory", "비강제"),
        ("maritime autonomous surface ships", "MASS(자율운항)"),
        ("maritime autonomous", "자율운항"),
        ("alternative fuels and", "대체연료·"),
        ("alternative fuels", "대체연료"),
        ("alternative fuel", "대체연료"),
        ("interim guidelines for the safety of ships using", "선박 대체연료 안전 임시 지침"),
        ("interim guidelines", "임시 지침"),
        ("guidelines", "지침"),
        ("amendments to", "개정 —"),
        ("amendments related to", "개정 —"),
        ("amendments", "개정"),
        ("amendment to", "개정 —"),
        ("amendment", "개정"),
    ]
    low = obj.lower()
    for eng, ko in replacements:
        if eng in low:
            obj = re.sub(re.escape(eng), ko, obj, count=1, flags=re.I)
            low = obj.lower()
    return _truncate_snippet(obj, max_len)


def _english_outcome_to_ko_clause(sent: str, *, max_gloss_len: int = 160) -> str:
    """Rule-based Korean clause from an English outcome sentence."""
    s = sent.strip()
    low = s.lower()
    parts: list[str] = []

    m_res = RESOLUTION_REF_RE.search(s)
    if m_res:
        parts.append(f"결의안 {m_res.group(1).strip()}")

    if "non-mandatory" in low and "mass code" in low:
        parts.append("비강제·goal-based MASS Code 채택")
    elif "mass code" in low:
        parts.append(f"MASS Code {_outcome_verb_ko(s)}")

    for fuel in ("ammonia", "hydrogen", "methanol", "lng"):
        if fuel in low and ("fuel" in low or "guideline" in low):
            parts.append(f"{fuel.upper()} 연료 관련 지침·안전 {_outcome_verb_ko(s)}")
            break

    if "lrit" in low:
        parts.append(f"LRIT(long-range identification) {_outcome_verb_ko(s)}")
    if "vdes" in low:
        parts.append(f"VDES(VHF data exchange) {_outcome_verb_ko(s)}")
    if "hormuz" in low or "strait of hormuz" in low:
        parts.append(f"호르무즈 해협 관련 결의 {_outcome_verb_ko(s)}")
    if "solas" in low and "gmdss" in low:
        parts.append(f"SOLAS/GMDSS 개정 {_outcome_verb_ko(s)}")
    elif "solas" in low:
        parts.append(f"SOLAS 개정 {_outcome_verb_ko(s)}")
    elif "gmdss" in low:
        parts.append(f"GMDSS 관련 {_outcome_verb_ko(s)}")

    if "cii" in low or "carbon intensity" in low:
        parts.append(f"CII·carbon intensity 보고 {_outcome_verb_ko(s)}")
    if ("ghg" in low or "net-zero" in low or "gfi" in low) and not any("ghg" in p.lower() for p in parts):
        parts.append(f"GHG/Net-Zero framework {_outcome_verb_ko(s)}")

    m_obj = re.search(
        r"\b(?:adopted|approved|endorsed|agreed(?: to)?|finalized|finalised)\s+(?:the\s+)?(.{12,120}?)(?:[.;,]|$)",
        s,
        re.I,
    )
    if m_obj and not parts:
        obj = _shorten_object_phrase(m_obj.group(1))
        if obj:
            parts.append(f"{obj} {_outcome_verb_ko(s)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = p[:40].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    if deduped:
        return " — ".join(deduped[:2])

    gloss = _truncate_snippet(s, max_gloss_len)
    gloss = re.sub(r"\badopted\b", "채택", gloss, flags=re.I)
    gloss = re.sub(r"\bapproved\b", "승인", gloss, flags=re.I)
    gloss = re.sub(r"\bendorsed\b", "지지", gloss, flags=re.I)
    gloss = re.sub(r"\bagreed\b", "합의", gloss, flags=re.I)
    gloss = re.sub(r"\bnoted\b", "노트(참고)", gloss, flags=re.I)
    gloss = re.sub(r"\binvited\b", "요청", gloss, flags=re.I)
    gloss = re.sub(r"\bresolution\b", "결의안", gloss, flags=re.I)
    gloss = re.sub(r"\bnon-mandatory\b", "비강제", gloss, flags=re.I)
    gloss = re.sub(r"\bmandatory\b", "강제", gloss, flags=re.I)
    gloss = re.sub(r"\bamendments?\b", "개정", gloss, flags=re.I)
    gloss = re.sub(r"\bguidelines?\b", "지침", gloss, flags=re.I)
    return gloss


def _compose_chunk_summary(
    body: str,
    *,
    topic_label: str,
    body_label: str,
    verb: str,
    max_claim_len: int = 180,
) -> str:
    """Always anchor bullet text on extracted chunk evidence."""
    claim = _extract_best_claim(body)
    clause = _english_outcome_to_ko_clause(claim, max_gloss_len=max_claim_len)
    label = topic_label.strip() or body_label
    if label and not clause.startswith(label):
        return f"{label}: {clause}"
    return clause


def _outcome_verb_ko(text: str) -> str:
    low = text.lower()
    if "adopted" in low:
        return "채택"
    if "approved" in low:
        return "승인"
    if "agreed" in low or "endorsed" in low:
        return "합의·지지"
    if "noted" in low:
        return "노트(참고)"
    if "invited" in low or "requested" in low:
        return "요청·촉구"
    return "논의·검토"


def summarize_chunk_ko(chunk: Any, *, topic_label: str = "", detail_level: str = "standard") -> str:
    body = _strip_meta(getattr(chunk, "text", ""))
    if not body:
        return f"{topic_label or '회의 결정'}: 검색 근거 텍스트 없음"
    fn = str(getattr(chunk, "file_name", "") or "").lower()
    is_msc = "msc" in fn or str(getattr(chunk, "source", "") or "").upper() == "MSC"
    body_label = "MSC 111" if is_msc else "MEPC"
    label = topic_label.strip() or body_label
    max_len = 260 if detail_level == "dense" else 180
    return _compose_chunk_summary(
        body, topic_label=label, body_label=body_label, verb=_outcome_verb_ko(body), max_claim_len=max_len
    )


def _filter_forbid_docs(chunks: list[Any], row: dict) -> list[Any]:
    forbid = {str(x).lower() for x in (row.get("forbid_gold_doc_ids") or [])}
    if not forbid:
        return chunks
    out: list[Any] = []
    for c in chunks:
        doc_id = str(getattr(c, "doc_id", "") or "").lower()
        if doc_id and doc_id in forbid:
            continue
        out.append(c)
    return out or chunks


def _format_bullet(
    chunk: Any,
    citation_map: dict[str, int],
    *,
    topic_label: str = "",
    detail_level: str = "standard",
    tier_note_mode: str = "default",
) -> str:
    cite = _cite(chunk, citation_map)
    summary = summarize_chunk_ko(chunk, topic_label=topic_label, detail_level=detail_level)
    tier = getattr(chunk, "_source_tier", classify_source_tier(chunk))
    if tier_note_mode == "official" and tier <= 1:
        tier_note = " (공식 보고 근거)" if tier == 0 else " (위원회 보고 근거)"
    elif tier_note_mode == "topic":
        tier_note = ""
    else:
        tier_note = f" ({tier_label(tier)} 근거)" if tier >= 2 else ""
    return f"- {summary}{tier_note} {cite}".strip()


def _is_penalty_reference_chunk(chunk: Any) -> bool:
    text = _strip_meta(getattr(chunk, "text", "")).lower()
    fn = str(getattr(chunk, "file_name", "") or "").lower()
    return bool(REFERENCE_OUTCOME_RE.search(text) or REFERENCE_OUTCOME_RE.search(fn))


def _rank_official_dense(item: tuple[float, Any]) -> float:
    score, chunk = item
    text = _strip_meta(getattr(chunk, "text", ""))
    low = text.lower()
    bonus = 0.0
    if _is_penalty_reference_chunk(chunk):
        bonus -= 12.0
    if RESOLUTION_REF_RE.search(text):
        bonus += 4.0
    if OUTCOME_ACTION_RE.search(text):
        bonus += 2.0
    bonus += count_outcome_signals(text) * 0.6
    tier = classify_source_tier(chunk)
    bonus += {0: 3.0, 1: 1.5, 2: -0.5, 3: -2.0}.get(tier, 0.0)
    return score + bonus


def _section1_official_dense(
    scored: list[tuple[float, Any]],
    *,
    n: int,
    citation_map: dict[str, int],
) -> tuple[str, list[str], list[Any]]:
    """Variant A: tier 0/1 공식 보고 중 결의안·adopted/approved 중심 (topic 중복 허용)."""
    warnings: list[str] = []
    official = [(s, c) for s, c in scored if classify_source_tier(c) <= 1 and not _is_penalty_reference_chunk(c)]
    pool = sorted(official if len(official) >= max(2, n // 2) else scored, key=_rank_official_dense, reverse=True)

    picked: list[Any] = []
    seen_res: set[str] = set()
    seen_cid: set[str] = set()
    for _s, chunk in pool:
        if len(picked) >= n:
            break
        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid in seen_cid:
            continue
        text = _strip_meta(getattr(chunk, "text", ""))
        if not OUTCOME_ACTION_RE.search(text) and count_outcome_signals(text) < 1:
            continue
        m = RESOLUTION_REF_RE.search(text)
        if m:
            res_key = m.group(1).strip().lower()
            if res_key in seen_res:
                continue
            seen_res.add(res_key)
        seen_cid.add(cid)
        picked.append(chunk)

    if len(picked) < n:
        for _s, chunk in pool:
            if len(picked) >= n:
                break
            cid = str(getattr(chunk, "chunk_id", "") or "")
            if cid in seen_cid:
                continue
            seen_cid.add(cid)
            picked.append(chunk)

    lines = [
        _format_bullet(
            c,
            citation_map,
            topic_label=_topic_label_ko(outcome_topic_id(str(getattr(c, "text", "")))),
            detail_level="dense",
            tier_note_mode="official",
        )
        for c in picked[:n]
    ]
    if len(picked) < n:
        warnings.append("answer_count_mismatch")
    return "\n".join(lines), warnings, picked[:n]


def _section1_topic_diverse(
    scored: list[tuple[float, Any]],
    *,
    n: int,
    citation_map: dict[str, int],
    profile: MeetingRetrievalProfile,
) -> tuple[str, list[str], list[Any]]:
    """Variant B: topic당 1개 — GHG·LRIT·안전·CII 등 분산."""
    from meeting_trend_ab import TOPIC_DIVERSE_PRIORITY

    warnings: list[str] = []
    topic_caps = {tid: 1 for tid in TOPIC_DIVERSE_PRIORITY}
    topic_caps["ghg_framework"] = 1
    topic_caps["ghg_safety"] = 1

    # B는 공식 tier보다 topic coverage 우선 — penalty reference는 제외
    filtered = [(s, c) for s, c in scored if not _is_penalty_reference_chunk(c)]
    pool = filtered if filtered else scored

    diverse = pick_diverse_topic_chunks(
        pool,
        n,
        topic_priority=TOPIC_DIVERSE_PRIORITY,
        topic_caps=topic_caps,
        topic_fn=outcome_topic_id,
    )
    if len(diverse) < n:
        extra = pick_diverse_topic_chunks(
            pool,
            n,
            topic_priority=TOPIC_DIVERSE_PRIORITY,
            topic_caps={tid: 2 for tid in TOPIC_DIVERSE_PRIORITY},
            topic_fn=outcome_topic_id,
        )
        seen = {str(getattr(c, "chunk_id", "")) for c in diverse}
        for c in extra:
            cid = str(getattr(c, "chunk_id", ""))
            if cid not in seen:
                diverse.append(c)
                seen.add(cid)
            if len(diverse) >= n:
                break

    topics = [outcome_topic_id(str(getattr(c, "text", ""))) for c in diverse]
    if len(set(topics)) < len(topics):
        warnings.append("duplicate_topic")

    lines = [
        _format_bullet(
            c,
            citation_map,
            topic_label=_topic_label_ko(outcome_topic_id(str(getattr(c, "text", "")))),
            detail_level="standard",
            tier_note_mode="topic",
        )
        for c in diverse[:n]
    ]
    return "\n".join(lines), warnings, diverse[:n]


def _best_chunk_matching(scored: list[tuple[float, Any]], pattern: re.Pattern[str]) -> Any | None:
    for _, chunk in scored:
        if pattern.search(_strip_meta(getattr(chunk, "text", ""))):
            return chunk
    return None


def _topic_priority_for_profile(profile: MeetingRetrievalProfile) -> tuple[str, ...] | None:
    from meeting_trend_ab import TOPIC_DIVERSE_PRIORITY

    if profile.answer_variant == "topic_diverse":
        return TOPIC_DIVERSE_PRIORITY
    if profile.internal_intent == "meeting_outcome":
        return MSC_OUTCOME_TOPIC_PRIORITY
    return None


def _topic_impact_ko(topic_id: str) -> str:
    return {
        "mass_code": "자율운항 선박 설계·시험·승인·SMS/위험평가 절차 검토 필요",
        "ghg_safety": "대체연료 저장·공급·화재/폭발 위험 관리가 운항·설계에 반영",
        "ghg_framework": "SEEMP·GFI·fleet GHG 보고·MARPOL Annex VI 준수 부담 증가",
        "cii_reporting": "CII·연료소비 데이터 수집·검증·fleet 보고 프로세스 강화",
        "lrit_vdes": "선박 원격식별·VHF 데이터 교환·GMDSS 운영 체계 변경 가능",
        "maritime_safety": "SOLAS·통신/안전 장비 개정이 설계·검사·운항에 영향",
        "hormuz": "항로·안전·통신 대응 및 운항 리스크 관리 필요",
        "general_outcome": "회의 결정이 규제 보고·승인·운영 절차에 단계적으로 반영",
    }.get(topic_id, "회의 결정이 선박 운항·규제 준수에 영향")


def _section1_meeting_outcome(
    scored: list[tuple[float, Any]],
    *,
    n: int,
    citation_map: dict[str, int],
    profile: MeetingRetrievalProfile,
) -> tuple[str, list[str], list[Any]]:
    warnings: list[str] = []
    if profile.answer_variant == "official_dense":
        return _section1_official_dense(scored, n=n, citation_map=citation_map)
    if profile.answer_variant == "topic_diverse":
        return _section1_topic_diverse(scored, n=n, citation_map=citation_map, profile=profile)

    lines: list[str] = []
    picked: list[Any] = []
    used_ids: set[str] = set()

    mass_best = next(
        (
            c
            for _s, c in scored
            if MASS_RE.search(_strip_meta(getattr(c, "text", "")))
            or outcome_topic_id(str(getattr(c, "text", ""))) == "mass_code"
        ),
        None,
    )
    if mass_best:
        used_ids.add(str(getattr(mass_best, "chunk_id", "")))
        picked.append(mass_best)
        lines.append(
            _format_bullet(
                mass_best,
                citation_map,
                topic_label=_topic_label_ko("mass_code"),
            )
        )

    remaining = [(s, c) for s, c in scored if str(getattr(c, "chunk_id", "")) not in used_ids]
    diverse = pick_diverse_topic_chunks(
        remaining,
        max(0, n - len(lines)),
        topic_priority=_topic_priority_for_profile(profile),
        exclude_topics={"mass_code"},
        topic_fn=outcome_topic_id,
    )
    for chunk in diverse:
        tid = outcome_topic_id(str(getattr(chunk, "text", "")))
        lines.append(_format_bullet(chunk, citation_map, topic_label=_topic_label_ko(tid)))
        picked.append(chunk)

    topics = [outcome_topic_id(str(getattr(c, "text", ""))) for c in picked]
    if len(set(topics)) < len(topics) or topics.count("mass_code") > 1:
        warnings.append("duplicate_topic")
    if len(lines) < n:
        warnings.append("answer_count_mismatch")
    return "\n".join(lines[:n]), warnings, picked[:n]


def _section1_altfuel_ghg(
    scored: list[tuple[float, Any]],
    *,
    citation_map: dict[str, int],
    profile: MeetingRetrievalProfile,
) -> tuple[str, list[str], list[Any]]:
    warnings: list[str] = []
    alt_scored = [
        (s, c)
        for s, c in scored
        if not is_excluded_chunk(c, profile=profile)
        and (
            ALT_FUEL_RE.search(_strip_meta(getattr(c, "text", "")))
            or _topic_id_for_text(str(getattr(c, "text", ""))) == "ghg_safety"
        )
    ]
    if len(alt_scored) < 2:
        warnings.append("weak_altfuel_evidence")
        alt_scored = [(s, c) for s, c in scored if not MASS_RE.search(_strip_meta(getattr(c, "text", "")))]

    diverse = pick_diverse_topic_chunks(
        alt_scored or scored,
        5,
        exclude_topics={"mass_code"},
        allowed_topics={"ghg_safety", "ghg_framework"} if alt_scored else None,
    )
    if not diverse:
        return "- [추가 확인 필요] 대체연료·GHG safety 관련 검색 근거가 부족합니다.", ["weak_altfuel_evidence"], []

    lines = [_format_bullet(c, citation_map) for c in diverse[:5]]
    s1_text = "\n".join(lines)
    if MASS_RE.search(s1_text.lower()):
        warnings.append("wrong_topic_in_answer")
    return s1_text, warnings, diverse[:5]


def _section1_mass_timeline(
    scored: list[tuple[float, Any]],
    *,
    citation_map: dict[str, int],
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    mass_scored = [(s, c) for s, c in scored if MASS_RE.search(_strip_meta(getattr(c, "text", "")))]
    pool = mass_scored or scored
    used_ids: set[str] = set()
    lines: list[str] = []

    decision = _best_chunk_matching(pool, OUTCOME_RE)
    if decision:
        used_ids.add(str(getattr(decision, "chunk_id", "")))
        lines.append(_format_bullet(decision, citation_map, topic_label="핵심 결정사항"))

    non_mand = _best_chunk_matching(
        [(s, c) for s, c in pool if str(getattr(c, "chunk_id", "")) not in used_ids],
        NON_MAND_RE,
    )
    if non_mand:
        used_ids.add(str(getattr(non_mand, "chunk_id", "")))
        lines.append(
            f"- **non-mandatory 여부**: {summarize_chunk_ko(non_mand)} {_cite(non_mand, citation_map)}".strip()
        )
    else:
        lines.append("- **non-mandatory 여부**: [추가 확인 필요] 검색 근거에서 non-mandatory 명시를 확인하지 못했습니다.")
        warnings.append("missing_non_mandatory")

    timeline = _best_chunk_matching(
        [(s, c) for s, c in pool if str(getattr(c, "chunk_id", "")) not in used_ids],
        TIMELINE_RE,
    )
    if timeline:
        body = _strip_meta(getattr(timeline, "text", ""))
        if TIMELINE_RE.search(body) and OUTCOME_RE.search(body):
            lines.append(
                f"- **mandatory code 일정 / experience-building**: {summarize_chunk_ko(timeline)} {_cite(timeline, citation_map)}".strip()
            )
        else:
            lines.append(
                "- **mandatory code 일정 / experience-building**: [추가 확인 필요] mandatory 전환 일정·EBP 세부는 근거 추가 대조 필요."
            )
            warnings.append("schedule_not_supported_by_evidence")
    else:
        lines.append(
            "- **mandatory code 일정 / experience-building**: [추가 확인 필요] experience-building phase·mandatory code roadmap 근거가 제한적입니다."
        )
        warnings.append("missing_mandatory_timeline")

    norms = [_normalize_bullet(l) for l in lines]
    if len(set(norms)) < len(norms):
        warnings.append("duplicate_topic")
    return "\n".join(lines), warnings


def _normalize_bullet(line: str) -> str:
    return re.sub(r"\[\d+\]", "", line).strip()[:80]


def _section1(
    clusters: list,
    *,
    profile: MeetingRetrievalProfile,
    citation_map: dict[str, int],
    row: dict,
    scored: list[tuple[float, Any]] | None = None,
) -> tuple[str, list[str], list[Any]]:
    lo, hi, _ = category_bullet_budget(row.get("category", ""), row)
    n = profile.requested_bullet_count or hi
    n = max(lo, min(hi, n))
    if profile.top_level_category == TOP_LEVEL_TREND and profile.requested_bullet_count:
        n = profile.requested_bullet_count

    scored = scored or []
    intent = profile.internal_intent
    extra_warnings: list[str] = []

    if profile.answer_variant == "official_dense":
        return _section1_official_dense(scored, n=n, citation_map=citation_map)
    if profile.answer_variant == "topic_diverse":
        return _section1_topic_diverse(scored, n=n, citation_map=citation_map, profile=profile)

    if intent == "meeting_outcome" and profile.requested_bullet_count:
        return _section1_meeting_outcome(scored, n=n, citation_map=citation_map, profile=profile)

    if intent == "altfuel_ghg_safety":
        s1, w, picked = _section1_altfuel_ghg(scored, citation_map=citation_map, profile=profile)
        return s1, w, picked

    if intent == "mass_code_timeline":
        s1, w = _section1_mass_timeline(scored, citation_map=citation_map)
        return s1, w, []

    if intent == "trend_summary" and scored:
        cap = min(n, 5)
        diverse = pick_diverse_topic_chunks(
            scored,
            cap,
            topic_priority=_topic_priority_for_profile(profile),
            topic_caps={"ghg_framework": 2, "general_outcome": 1, "maritime_safety": 1, "cii_reporting": 1},
            exclude_topics={"mass_code"},
            topic_fn=outcome_topic_id,
        )
        lines = [
            _format_bullet(c, citation_map, topic_label=_topic_label_ko(outcome_topic_id(str(getattr(c, "text", "")))))
            for c in diverse
        ]
        return "\n".join(lines), extra_warnings, diverse

    lines: list[str] = []
    used_chunk_ids: set[str] = set()
    exclude = exclude_topics_for_intent(intent)
    caps = topic_caps_for_intent(intent)

    if profile.requested_bullet_count and scored:
        diverse = pick_diverse_topic_chunks(
            scored,
            n,
            topic_priority=MSC_OUTCOME_TOPIC_PRIORITY if intent == "meeting_outcome" else None,
            topic_caps=caps,
            exclude_topics=exclude,
        )
        for chunk in diverse:
            used_chunk_ids.add(str(getattr(chunk, "chunk_id", "") or ""))
            tid = _topic_id_for_text(str(getattr(chunk, "text", "") or ""))
            lines.append(_format_bullet(chunk, citation_map, topic_label=_topic_label_ko(tid)))
    else:
        used_topics: set[str] = set()
        for cluster in clusters:
            if len(lines) >= n:
                break
            if cluster.topic_id in exclude:
                continue
            if cluster.topic_id in used_topics and profile.requested_bullet_count:
                continue
            used_topics.add(cluster.topic_id)
            rep = cluster.representative
            if not rep:
                continue
            used_chunk_ids.add(str(getattr(rep, "chunk_id", "") or ""))
            lines.append(_format_bullet(rep, citation_map, topic_label=cluster.label_ko))

    if len(lines) < n and scored:
        for chunk in pick_diverse_topic_chunks(
            scored, n - len(lines), used_chunk_ids=used_chunk_ids, exclude_topics=exclude, topic_caps=caps
        ):
            lines.append(_format_bullet(chunk, citation_map))

    if not lines:
        lines.append("- 검색된 회의 자료에서 핵심 결정·논의를 확인하지 못했습니다. 추가 확인이 필요합니다.")
    return "\n".join(lines[:n]), extra_warnings, []


def _section2(
    chunks: list[Any],
    citation_map: dict[str, int],
    *,
    profile: MeetingRetrievalProfile,
    s1_chunks: list[Any] | None = None,
) -> str:
    focus = s1_chunks or chunks
    if profile.answer_variant == "official_dense":
        lines: list[str] = []
        for c in focus[:4]:
            text = _strip_meta(getattr(c, "text", ""))
            m = RESOLUTION_REF_RE.search(text)
            prefix = f"결의안 {m.group(1).strip()} — " if m else ""
            tid = outcome_topic_id(text)
            lines.append(f"- {prefix}{_topic_impact_ko(tid)} {_cite(c, citation_map)}".strip())
        if not lines:
            lines.append("- 공식 채택·승인 결정은 운항·규제 보고·승인 절차에 단계적으로 반영됩니다.")
        return "\n".join(lines[:4])

    if profile.answer_variant == "topic_diverse":
        lines = []
        seen: set[str] = set()
        for c in focus:
            tid = outcome_topic_id(str(getattr(c, "text", "")))
            if tid in seen:
                continue
            seen.add(tid)
            label = _topic_label_ko(tid)
            lines.append(f"- **{label}**: {_topic_impact_ko(tid)} {_cite(c, citation_map)}")
            if len(lines) >= 4:
                break
        if not lines:
            lines.append("- 주제별로 선박 운항·설계·규제 보고에 서로 다른 영향이 예상됩니다.")
        return "\n".join(lines[:4])

    impact_chunks = sorted(
        chunks,
        key=lambda c: -count_impact_signals(_strip_meta(getattr(c, "text", ""))),
    )
    lines = []

    if profile.internal_intent == "altfuel_ghg_safety":
        lines.append("- **선박 운항 영향**: 대체연료 저장·공급·운용 절차, 화재·폭발 위험 관리가 운항계획에 반영될 수 있습니다.")
        lines.append("- **설계·안전·승인 영향**: ammonia/hydrogen/methanol 등 연료별 safety assessment·interim guideline 준수가 필요합니다.")
        lines.append("- **선원 교육/운영 절차**: 연료 취급·비상대응·crew competence 요건을 SMS에 반영해야 합니다.")
    elif profile.top_level_category == TOP_LEVEL_ENV:
        lines.append("- **선박 운항 영향**: CII·연료소비·운항효율 데이터가 fleet 보고·등급 산정에 반영될 수 있어 운항계획·속도관리 검토가 필요합니다.")
        lines.append("- **규제 보고/데이터 제출 영향**: SEEMP·DCS·연간 보고·검증(verification) 절차와 연계된 compliance 부담이 있을 수 있습니다.")
        lines.append("- **설계·안전·승인 영향**: 대체연료 전환 시 저장·공급·기관 안전성 평가·승인 요건을 대조해야 합니다.")
    elif profile.top_level_category == TOP_LEVEL_AUTO:
        lines.append("- 자율운항 선박의 설계·시험·승인·검증, safety case·class notation·원격운항 운영 절차에 영향이 있습니다.")
        lines.append("- 비강제 MASS Code 적용 시에도 SMS·위험평가·선원/원격운항자 역량 요건을 사전 검토해야 합니다.")
    else:
        lines.append("- 회의 결정은 선박 설계·운항·안전·규제 보고 프로세스에 단계적으로 반영될 수 있습니다.")

    if impact_chunks:
        cite = _cite(impact_chunks[0], citation_map)
        if cite and cite not in lines[0]:
            lines[0] = lines[0].rstrip(".") + f". {cite}"

    return "\n".join(lines[:4])


def _section3(
    chunks: list[Any],
    *,
    profile: MeetingRetrievalProfile,
    answer: str,
    s1_chunks: list[Any] | None = None,
) -> str:
    focus = s1_chunks or chunks[:12]
    if profile.answer_variant == "official_dense":
        lines: list[str] = []
        for c in focus[:3]:
            text = _strip_meta(getattr(c, "text", ""))
            low = text.lower()
            if "entry into force" in low or "entered into force" in low:
                lines.append(f"- [발효 시점] {_truncate_snippet(text, 130)}")
            m = RESOLUTION_REF_RE.search(text)
            if m:
                lines.append(f"- [결의안 후속] {m.group(1).strip()} — 적용 범위·발효 일정 추가 대조 필요")
        lines.append("- [공식 보고 한계] working paper·agenda item만으로는 확정 규제로 단정하기 어렵습니다.")
        return "\n".join(lines[:4])

    if profile.answer_variant == "topic_diverse":
        lines = []
        seen: set[str] = set()
        for c in focus:
            tid = outcome_topic_id(str(getattr(c, "text", "")))
            if tid in seen:
                continue
            seen.add(tid)
            lines.append(f"- [{_topic_label_ko(tid)}] 후속 MEPC/MSC 세션·guideline 확정·entry into force 추적 필요")
            if len(lines) >= 4:
                break
        if not lines:
            lines.append("- [후속 회의] topic별 guideline·mandatory instrument 확정 여부를 추적해야 합니다.")
        return "\n".join(lines[:4])

    lines: list[str] = []
    blob = " ".join(_strip_meta(getattr(c, "text", "")) for c in chunks[:12]).lower()

    if profile.top_level_category == TOP_LEVEL_AUTO:
        if "mandatory" in blob and "non-mandatory" not in answer.lower():
            lines.append("- [추가 확인 필요] mandatory code 전환 일정·entry into force는 본문 추가 대조가 필요합니다.")
        if "experience-building" in blob or "work plan" in blob:
            lines.append("- [후속 회의] experience-building phase·후속 MSC/IMO 작업계획 세부 일정을 확인해야 합니다.")
        else:
            lines.append("- [추가 확인 필요] mandatory code 일정·experience-building phase 여부는 검색 근거가 제한적입니다.")
        lines.append("- [해석 근거] IMO MASS Code와 선급 Rule(DNV-CG-0264 등)은 별도 체계이므로 연계 적용 범위를 추가 확인해야 합니다.")

    elif profile.top_level_category == TOP_LEVEL_ENV:
        lines.append("- [미확정 규제] draft guideline·interim requirement는 본회의 채택·발효 시점 추가 확인 필요.")
        lines.append("- [후속 회의] MEPC/MSC 후속 세션에서 mandatory instrument·code 확정 여부 추적 필요.")
        if any(tier >= 2 for tier in (getattr(c, "_source_tier", 1) for c in chunks[:8])):
            lines.append("- [근거 한계] agenda/submission 문서만으로 확정 규제로 단정하기 어려운 항목이 있습니다.")

    else:
        lines.append("- [미확정 규제] noted/invited 수준 항목은 확정 결정이 아니므로 후속 회의에서 재확인이 필요합니다.")
        lines.append("- [후속 회의] guideline 확정·entry into force 일정은 본문 추가 검색이 필요합니다.")

    return "\n".join(lines[:4])


def _section4(chunks: list[Any], citation_map: dict[str, int]) -> str:
    class_chunks = [
        c
        for c in chunks
        if str(getattr(c, "source", "")).upper() in {"DNV", "LR", "ABS", "KR"}
    ]
    if not class_chunks:
        return "- 검색된 근거 내에서는 관련 선급 Rule/Guidance가 명확히 확인되지 않음 (IMO 회의 자료 중심 질의)."

    lines: list[str] = []
    seen: set[str] = set()
    n = 0
    for c in class_chunks[:3]:
        fn = str(getattr(c, "file_name", "") or "")
        if fn in seen:
            continue
        seen.add(fn)
        n += 1
        cite = _cite(c, citation_map)
        lines.extend(
            [
                f"{n}. {fn.replace('.pdf', '')}",
                "- 유형: 관련 참고 Rule/Guidance 후보",
                "- 관련성: 회의 논의와 연계해 선급 요건 대조가 필요한 참고 문서",
                f"- 근거: {fn}, p.{getattr(c, 'page_number', '?')} {cite}",
                "- 확정 여부: 추가 확인 필요",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_meeting_structured_answer(
    chunks: list[Any],
    *,
    question: str,
    row: dict,
    profile: MeetingRetrievalProfile,
    warning_flags: list[str] | None = None,
) -> tuple[str, list[str], dict]:
    warnings = list(warning_flags or [])
    work = _filter_forbid_docs(dedupe_page_chunks(list(chunks)), row)
    work = [c for c in work if not is_excluded_chunk(c, profile=profile)]
    if not work:
        work = _filter_forbid_docs(dedupe_page_chunks(list(chunks)), row)
    scored = [(score_chunk(c, profile=profile), c) for c in work]
    scored.sort(key=lambda x: -x[0])

    weak_tier_only = all(classify_source_tier(c) >= 2 for _, c in scored[:5]) if scored else True
    if weak_tier_only:
        warnings.append("weak_source_tier")

    if not any(count_outcome_signals(_strip_meta(getattr(c, "text", ""))) for _, c in scored[:8]):
        if profile.internal_intent != "altfuel_ghg_safety":
            warnings.append("no_outcome_signal")

    clusters = cluster_chunks(scored)
    citation_map = _build_citation_map(work[:12])
    s1, s1_warnings, s1_chunks = _section1(clusters, profile=profile, citation_map=citation_map, row=row, scored=scored)
    warnings.extend(s1_warnings)

    answer = join_four_sections(
        {
            "1": s1,
            "2": _section2(work, citation_map, profile=profile, s1_chunks=s1_chunks),
            "3": _section3(work, profile=profile, answer="", s1_chunks=s1_chunks),
            "4": _section4(work, citation_map),
        }
    )
    answer = ENGLISH_LEAK_RE.sub("", answer)

    deduped, dedup_warnings = apply_answer_dedup(
        answer,
        profile_intent=profile.internal_intent,
        requested_count=profile.requested_bullet_count,
    )
    answer = deduped
    warnings.extend(dedup_warnings)

    qid = str(row.get("question_id") or "")
    coverage, cov_warnings = run_coverage_check(qid, answer, work, row=row)
    warnings.extend(cov_warnings)
    if not CITATION_RE.search(answer):
        warnings.append("citation_missing")

    s1_for_topics = answer.split("## 2)")[0] if "## 2)" in answer else answer
    meta = {
        "coverage_check": coverage,
        "used_citations": sorted(int(x) for x in CITATION_RE.findall(answer)),
        "detected_topics": detect_section1_topics(s1_for_topics),
        "top_level_category": profile.top_level_category,
        "internal_intent": profile.internal_intent,
    }
    return answer, list(dict.fromkeys(warnings)), meta
