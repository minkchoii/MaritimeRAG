"""Top-level category, internal intent, and retrieval profile for meeting QA."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from meeting_outcome_retrieval import parse_outcome_item_count
from meeting_summary_intent import is_meeting_summary_intent
from retrieval_query_analysis import is_meeting_outcome_question

# User-facing top-level categories
TOP_LEVEL_TREND = "latest_trend_summary"
TOP_LEVEL_ENV = "env_regulation_response"
TOP_LEVEL_AUTO = "autonomous_mass"
TOP_LEVEL_RULE = "rule_guidance_lookup"

LEGACY_TO_TOP: dict[str, str] = {
    "trend_summary": TOP_LEVEL_TREND,
    "meeting_outcome": TOP_LEVEL_TREND,
    "env_regulation": TOP_LEVEL_ENV,
    "autonomous": TOP_LEVEL_AUTO,
    "rule_lookup": TOP_LEVEL_RULE,
}

TOP_LEVEL_LABELS_KO: dict[str, str] = {
    TOP_LEVEL_TREND: "최신 동향 요약",
    TOP_LEVEL_ENV: "환경규제 대응",
    TOP_LEVEL_AUTO: "자율운항(MASS)",
    TOP_LEVEL_RULE: "Rule/Guidance 조회",
}


@dataclass
class MeetingRetrievalProfile:
    top_level_category: str
    internal_intent: str
    profile_id: str
    use_dense: bool = True
    use_bm25: bool = True
    use_rrf: bool = True
    use_source_tier: bool = True
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    sub_queries: list[str] = field(default_factory=list)
    section_emphasis: dict[str, str] = field(default_factory=dict)
    requested_bullet_count: int | None = None
    answer_variant: str = "default"

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "top_level_category": self.top_level_category,
            "internal_intent": self.internal_intent,
            "retrieval_profile": self.profile_id,
            "use_dense": self.use_dense,
            "use_bm25": self.use_bm25,
            "use_rrf": self.use_rrf,
            "use_source_tier": self.use_source_tier,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "sub_queries": self.sub_queries,
            "section_emphasis": self.section_emphasis,
            "requested_bullet_count": self.requested_bullet_count,
        }


def resolve_internal_intent(question: str, row: dict, *, legacy_category: str) -> str:
    explicit = str(row.get("internal_intent") or "").strip()
    if explicit:
        return explicit

    q = question or ""
    ql = q.lower()

    if legacy_category == "autonomous" or (
        "mass code" in ql and any(k in q for k in ("mandatory", "일정", "timeline", "experience-building"))
    ):
        return "mass_code_timeline"

    if legacy_category == "env_regulation":
        if any(
            k in q
            for k in (
                "대체연료",
                "GHG 안전",
                "ghg safety",
                "alternative fuel",
                "저인화점",
                "ammonia",
                "hydrogen",
            )
        ) or ("ghg" in ql and "안전" in q):
            return "altfuel_ghg_safety"
        return "env_regulation"

    if is_meeting_summary_intent(question, row):
        return "meeting_outcome"
    if legacy_category == "meeting_outcome":
        return "meeting_outcome"
    if is_meeting_outcome_question(question, row):
        return "meeting_outcome"
    if legacy_category == "trend_summary":
        return "trend_summary"
    return legacy_category or "general"


def resolve_top_level_category(legacy_category: str) -> str:
    return LEGACY_TO_TOP.get(legacy_category, TOP_LEVEL_TREND)


def build_meeting_retrieval_profile(question: str, row: dict, *, legacy_category: str) -> MeetingRetrievalProfile:
    top = resolve_top_level_category(legacy_category)
    intent = resolve_internal_intent(question, row, legacy_category=legacy_category)
    q = question or ""
    ql = q.lower()

    if top == TOP_LEVEL_TREND:
        n = None
        subs: list[str] = []
        if intent == "meeting_outcome" or row.get("outcome_item_count") or re.search(
            r"\d+\s*개\s*(?:항목|개)", q
        ):
            n = parse_outcome_item_count(q, row)
        if intent == "meeting_outcome" and "msc" in ql:
            subs = [
                "MSC 111 MASS Code adopted approved non-mandatory",
                "MSC 111 alternative fuel GHG safety ammonia hydrogen",
                "MSC 111 LRIT VDES GMDSS SOLAS Hormuz maritime safety",
            ]
        return MeetingRetrievalProfile(
            top_level_category=top,
            internal_intent=intent,
            profile_id="trend_dense_first",
            use_dense=True,
            use_bm25=True,
            use_rrf=True,
            use_source_tier=True,
            dense_weight=1.2,
            bm25_weight=0.8,
            requested_bullet_count=n,
            sub_queries=subs,
            section_emphasis={"1": "primary", "2": "secondary", "3": "secondary", "4": "optional"},
        )

    if top == TOP_LEVEL_ENV:
        subs: list[str] = []
        if intent == "altfuel_ghg_safety":
            subs = [
                "MSC 111 alternative fuel ammonia hydrogen methanol LNG safety",
                "MSC 111 GHG safety interim guidelines approved agreed conclusion",
                "MSC 111 ISE new technology low-flashpoint fuel risk assessment",
            ]
        return MeetingRetrievalProfile(
            top_level_category=top,
            internal_intent=intent,
            profile_id="env_balanced" if intent == "env_regulation" else "env_altfuel_safety",
            use_dense=True,
            use_bm25=True,
            use_rrf=True,
            use_source_tier=True,
            dense_weight=1.0,
            bm25_weight=1.1 if intent == "altfuel_ghg_safety" else 1.0,
            sub_queries=subs,
            section_emphasis={"1": "secondary", "2": "primary", "3": "primary", "4": "optional"},
        )

    if top == TOP_LEVEL_AUTO:
        session = "MSC 111"
        subs = [
            f"{session} MASS Code adopted approved decision",
            f"{session} MASS Code non-mandatory voluntary application",
            f"{session} MASS Code mandatory code timeline experience-building phase",
        ]
        return MeetingRetrievalProfile(
            top_level_category=top,
            internal_intent="mass_code_timeline",
            profile_id="autonomous_bm25_first",
            use_dense=True,
            use_bm25=True,
            use_rrf=True,
            use_source_tier=True,
            dense_weight=0.9,
            bm25_weight=1.3,
            sub_queries=subs,
            section_emphasis={"1": "primary", "2": "secondary", "3": "primary", "4": "optional"},
        )

    return MeetingRetrievalProfile(
        top_level_category=top,
        internal_intent=intent,
        profile_id="default",
    )


def uses_structured_meeting_answer(row: dict, *, legacy_category: str) -> bool:
    """Categories 1–3 use evidence-based 4-section structured answers (no LLM)."""
    top = resolve_top_level_category(legacy_category)
    return top in {TOP_LEVEL_TREND, TOP_LEVEL_ENV, TOP_LEVEL_AUTO}
