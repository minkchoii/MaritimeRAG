"""Shared Accurate-mode answer depth guidance (format unchanged, report-style bullets)."""

CITATION_GUIDANCE = """
## Citation (필수 — §1·§2 모든 bullet)
- context 청크는 **[1], [2], …** 로 번호가 매겨져 있다 (헤더 `[N] source=…` 참조).
- **§1·§2의 모든 bullet** 문장 끝에 해당 근거 번호를 **반드시** `[N]` 또는 `[2][3]` 형식으로 붙인다.
- citation **없는 bullet은 작성하지 말 것** (근거 없는 문장 금지).
- `[근거]`, `[근거 MEPC…]` placeholder **금지** — 숫자만 사용.
- 존재하지 않는 번호 인용 금지 (context에 없는 [99] 등).
- §3 추후 확인 항목은 context에 없을 때 citation 생략 가능.
"""

EVIDENCE_DISPLAY_GUIDANCE = """
## 근거 표시 (§1·§2 필수)
각 bullet에 가능한 범위에서 **회의차수·문서명·조항/결의·가이드 명칭**을 명시한다.
- 회의: MEPC 84, MSC 111, ISWG-GHG 20차 등
- 문서: MEPC 84/7/14, DNV-CG-0264, LR Notice No.1 등
- 조항·결의·가이드: MARPOL Annex VI reg.36·37, Section 15, MASS Code, SEEMP Part III 등
"""

CATEGORY_BULLET_DEFAULTS: dict[str, tuple[int, int, int]] = {
    "trend_summary": (7, 10, 3),
    "env_regulation": (5, 7, 3),
    "autonomous": (5, 7, 3),
    "rule_lookup": (2, 3, 2),
}


def category_bullet_budget(category: str, row: dict | None = None) -> tuple[int, int, int]:
    """Section 1 bullet (min, max, priority_top_n). Eval row may override."""
    row = row or {}
    dmin, dmax, dpriority = CATEGORY_BULLET_DEFAULTS.get(category, (5, 7, 3))
    bmin = int(row.get("answer_bullets_min") or dmin)
    bmax = int(row.get("answer_bullets_max") or dmax)
    priority = int(row.get("summary_priority_bullets") or dpriority)
    return bmin, bmax, priority


SECTION_TITLES = {
    "1": "## 1) 핵심 요약",
    "2": "## 2) 선박 운항/업무 영향",
    "3": "## 3) 추후 확인 필요사항",
    "4": "## 4) 관련 선급 Rule / Guidance",
}


def join_four_sections(parts: dict[str, str]) -> str:
    out: list[str] = []
    for key in ("1", "2", "3", "4"):
        body = (parts.get(key) or "").strip()
        if not body:
            continue
        out.append(SECTION_TITLES[key])
        out.append(body)
    return "\n\n".join(out).strip()

ANSWER_DENSITY_GUIDANCE = """
## bullet 작성 원칙 (전 섹션 — 보고서형 문장)
각 bullet은 **키워드·항목 나열로 끝내지 말고**, 완전한 한국어 **2~3문장**으로 작성한다.
아래 **3요소 중 최소 2개**를 반드시 포함한다 (context에 있는 정보만):
1) **무엇이 논의·개정·요구되었는가** — 회의차수, 문서번호, 안건
2) **규제적으로 어떤 의미인가** — "이는 ~ 보고·검증 체계와 연결된다" 등
3) **선박 운항·선사 업무에 어떤 영향을 주는가** — "따라서 선사는 ~해야 한다" 등

**금지 패턴:**
- `A ↔ B`, `A, B, C`만 나열하고 **문장을 완성하지 않는 bullet**
- bullet이 **연결 기호(↔, /, ·)** 또는 **쉼표 구분 키워드**로 끝나는 것
- "논의되었습니다", "정리되었습니다", "필요합니다", "필요함"으로 **끝내는 것** (영향 문장 없이)
- 영문 고유명사·약어만 나열 (GFI, SEEMP, CII 등은 **문장 안에서** 풀어 쓸 것)

**전문 용어는 유지**하되 보고서 문장으로 다듬는다:
GFI, MARPOL Annex VI reg.36·37, SEEMP Part III, CII fleet carbon intensity,
well-to-tank emission factor, LCA/sustainability themes 등

**bullet 템플릿 (한 줄):**
`- **[주제]**: [회의·문서에서 무엇이 논의/개정/요구되었는지]. [규제적 의미 — "이는 …와 연결된다"]. [업무 영향 — "따라서 선사/운항부는 …"]. [N]`
"""

SECTION2_OPERATIONAL_GUIDANCE = """
## 2) 선박 운항/업무 영향
- bullet **2~4개**, 각 **2문장 이상**, §1과 **완전히 다른 문장** (복사·패러프레이즈 금지)
- §1의 규제 요약·문서 나열을 반복하지 말고 **실무 조치**만: 항차·연료·CII·SEEMP·보고·검증·설계·승인 절차
- citation [N]
"""

SECTION2_RULE_LOOKUP_GUIDANCE = """
## 2) 선박 운항/업무 영향
- bullet **1~2개** (§1 문서별 요약·동일 결론 문장 반복 금지)
- 검색된 Rule/Guidance 전체에 대한 **통합 실무 조치**만 (notation 검토, 적용 범위·class 승인 절차 등)
- citation [N]
"""

SECTION3_FOLLOWUP_GUIDANCE = """
## 3) 추후 확인 필요사항
- bullet **2~4개**, §1·§2에 없는 항목만
- 각 bullet은 아래 태그 **하나로 시작**:
  - `- [미확정 규제] …`
  - `- [해석 논란] …` 또는 `- [해석 근거] …`
  - `- [선급별 상이 요구] …`
- 무엇을·왜·어느 문서에서 확인할지 **완전한 문장**으로
- context 없음: `- [해석 근거] 검색 결과 내 확인 불가 — (이유)`
"""

SECTION4_GUIDANCE = """
## 4) 관련 선급 Rule / Guidance
- context에 DNV/LR/KR/ABS Rule·Guidance가 있으면 문서명·scope bullet (§1과 **완전 중복 나열은 피함**)
- catalog_table·cross-ref 표에만 있는 문서는 **후보**로 표시
- IMO 회의 자료만 검색된 경우: `- 본 검색은 IMO 회의 자료 중심. 선급 Rule/Guidance는 별도 검색 필요.`
- 해당 없으면: `- 해당 없음`
"""

FORMAT_RULES = """
**출력 형식 (필수 — 아래 4개 섹션 순서 고정):**
1) `## 1) 핵심 요약`
2) `## 2) 선박 운항/업무 영향`
3) `## 3) 추후 확인 필요사항`
4) `## 4) 관련 선급 Rule / Guidance`

- 섹션 제목은 `## N) …` (`###` 금지)
- 모든 bullet = `- ` 한 줄 (sub-bullet·들여쓰기 중첩 금지)
- §1에 §2(실무 영향) 내용 넣지 말 것
- §1·§2 bullet: 2~3문장 + citation [N] (rule_lookup은 §1 bullet 1~2문장 허용)
"""

ANTI_REPETITION_GUIDANCE = """
## 반복 금지 (최우선 — 모든 섹션·카테고리)
- **동일·유사 문장을 2회 이상 쓰지 말 것** (§1 bullet 간, §1↔§2, 문서별 bullet 간).
- 문서별 bullet마다 **고유한 사실**(문서번호, scope, notation, 적용대상, 핵심 요건)만 다르게 쓸 것.
- **고정 결론 문구를 모든 bullet에 붙이지 말 것** — 예: "이는 Smart/autonomous… class compliance 범위를 명시", "따라서 설계·운항 부서는 fleet별로 검토해야 한다"를 CG/RP마다 복사 금지.
- "이는 ~와 연결된다", "따라서 ~해야 한다" 문형을 **bullet마다 같은 말로** 반복하지 말고, context에 맞는 **다른 표현** 또는 **한 번만** 통합 서술.
- §2는 §1 문장·문서 나열을 **다시 쓰지 말 것** — 실무 조치를 **1~2 bullet로 통합** (rule_lookup은 특히 엄격).
- 유사한 선급 RP/CG가 여러 개면: 문서별 bullet **또는** 주제 통합 bullet 중 하나만 — **둘 다 같은 영향 문장으로 끝내지 말 것**.
"""

RULE_LOOKUP_GUIDANCE = """
## Rule/Guidance 조회 (카테고리 rule_lookup)
- §1 bullet **2~3개** — 검색된 file_name별 scope·notation·적용대상·핵심 요건 (서로 다른 내용).
- **문서명·번호는 user prompt 「인용 허용 문서」·「Citation 매핑」의 file_name만** 사용.
- citation [N]과 bullet 문서명이 **일치**해야 함.
- placeholder `(context의 …)` 출력 금지.
- §2는 **1~2 bullet** 통합 실무 조치만.
- §4는 §1에 없는 추가 선급 Rule·catalog 후보를 정리 (있을 때만).
"""

RULE_LOOKUP_OUTPUT_SCOPE = ""

RULE_LOOKUP_EVIDENCE_GUIDANCE = """
## Rule/Guidance 근거 (최우선)
- 답변의 모든 문서명은 context 헤더 `doc=파일명` 또는 user prompt 인용 목록과 **완전히 일치**해야 한다.
- context에 없는 DNV-RP-*, DNV-RU-SHIP Pt.* 등은 **작성 금지** (코퍼스 미수록 가능).
- cross-reference 표( Document code / Title 목록만 있는 청크)는 주제 설명 근거로 쓰지 말고, **해당 표에 등장하는 문서가 별도 [N] 본문 청크로 있을 때만** 언급.
"""

ENV_REGULATION_V01_HINT = """
[V01 — §1 최소 7개 bullet, context에 있을 때 주제별 1 bullet]

[나쁨 — 키워드 나열·↔ 종결]
- "GFI compliance/reporting/verification ↔ MARPOL Annex VI reg.36·37"
- "SEEMP Guidelines 개정, Fifth IMO GHG Study, LCA/sustainability themes"
- "MEPC 84에서 CII 관련 검토가 논의되었습니다."

[좋음 — 보고서형 2~3문장]
- "**GFI·MARPOL Annex VI reg.36·37**: ISWG-GHG 20은 GFI compliance·reporting·verification을 MARPOL Annex VI regulation 36·37과 정합되게 다루었다. 이는 선박 연료·에너지 사용에 대한 국제 보고·검증 의무 체계를 확장하는 쪽이다. 따라서 선사는 GFI 산정·제출 데이터와 기존 DCS/연료 보고 필드 간 매핑을 점검해야 한다 [1][8]"
- "**CII fleet carbon intensity**: MEPC 84-6-2는 2024 fleet CII 결과 제출을 보고했다. 이는 fleet carbon intensity 추세를 공개하는 자료이다. 따라서 운항부는 SEEMP Part III 갱신 주기와 연료·속도 데이터 수집 체계를 재점검해야 한다 [4]"
- "**GESAMP-LCA WG**: 2차 회의는 well-to-tank emission factor의 representativeness 기준을 정리했다. 이는 대체연료 GFI·LCA/sustainability themes 산정 시 데이터 품질 기준을 구체화한다. 그 결과 연료 공급망 LCA 증빙·내부 사전 수집 범위를 확대해야 할 수 있다 [13]"
"""

GOOD_BAD_EXAMPLES = """
[Citation 예시 — 모든 문항 공통]
- 나쁨: "DNV-CG-0264는 자율운항 및 원격운항 선박에 대한 guidance를 제공한다." (citation 없음)
- 좋음: "DNV-CG-0264는 autonomous·remotely operated vessels의 scope와 notation(AUTO, REMO 등) 요건을 정의한다 [2]"

[반복 금지 — Rule/Guidance 나열]
- 나쁨: 검색되지 않은 RP-C205·RU-SHIP 이름을 붙이거나, 모든 bullet 끝에 동일한 "fleet별 검토" 문장 반복
- 나쁨: "(context의 고유 주제)" 같은 placeholder를 그대로 출력
- 좋음: **context에 있는 file_name만** — "DNV-CG-0264.pdf: autoremote·remote link 요건 [2]", "DNV-RU-OU-0103.pdf: Smart notation 별도 적용 [6]" — 공통 실무 조치는 §2에 **1회만**

[문장 품질 — GFI 예시]
- 나쁨: "GFI compliance/reporting/verification ↔ MARPOL Annex VI reg.36·37"
- 좋음: "ISWG-GHG 20은 GFI 준수·보고·검증(verification)을 MARPOL Annex VI regulation 36·37과 연동해 정비하고 있다. 이는 선박 연료 GHG 배출에 대한 국제 보고·검증 프레임과 직결된다. 따라서 선사는 GFI 관련 데이터 필드와 내부 연료·배출 ledger를 대조해야 한다 [1]"

[문장 품질 — SEEMP/CII 예시]
- 나쁨: "SEEMP Part III 갱신 주기 재점검 필요"
- 좋음: "2024 fleet CII 결과 보고는 fleet carbon intensity 추세를 공개한다. 이는 SEEMP Part III 갱신·연료 효율 조치와 연동된다. 따라서 운항·기술 부서는 SEEMP Part III 갱신 일정과 CII rating 목표를 함께 재설정해야 한다 [4]"
"""
