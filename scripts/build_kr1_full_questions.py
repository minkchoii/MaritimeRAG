"""Build full-range kr_1_2025 eval question set (JSONL)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PILOT = Path("data/eval/kr_1_2025_questions_pilot_30p.jsonl")
OUT = Path("data/eval/kr_1_2025_questions.jsonl")

# Pages 31–462: one question per ~50-page band (and key mid-band pages)
EXTENSION = [
    {
        "question_id": "KR1_Q035",
        "question": "106절 계선 시 정기검사는 어떻게 되는가?",
        "expected_keywords": ["106", "계선", "정기적 검사"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 33,
        "gold_clause": "106",
        "page_band": "p031-p050",
        "note": "106. 계선 — kr_1_2025_p0033_m003",
    },
    {
        "question_id": "KR1_Q036",
        "question": "109절 부재의 쇠모한도 초과 시 어떻게 해야 하는가?",
        "expected_keywords": ["109", "쇠모한도", "부재"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 35,
        "gold_clause": "109",
        "page_band": "p031-p050",
        "note": "109. 부재의 쇠모한도 — kr_1_2025_p0035_m001",
    },
    {
        "question_id": "KR1_Q037",
        "question": "정기검사 시 불활성가스장치 검사에서 역류방지장치는 무엇을 확인하는가?",
        "expected_keywords": ["불활성가스", "역류방지", "데크 씰"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 45,
        "gold_clause": "",
        "gold_chunk_ids": ["kr_1_2025_p0045_m000"],
        "page_band": "p031-p050",
        "note": "정기검사 화재/소화 — kr_1_2025_p0045_m000",
    },
    {
        "question_id": "KR1_Q038",
        "question": "401절 첫 번째 정기검사는 언제까지 시행해야 하는가?",
        "expected_keywords": ["401", "정기검사", "5년"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 55,
        "gold_clause": "401",
        "page_band": "p051-p100",
        "note": "401. 검사시기 — kr_1_2025_p0055_m001",
    },
    {
        "question_id": "KR1_Q039",
        "question": "502절 정기검사(기관)에서 주기관 검사사항은?",
        "expected_keywords": ["502", "정기검사", "주기관"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 65,
        "gold_clause": "502",
        "page_band": "p051-p100",
        "note": "502. 검사사항 기관 — kr_1_2025_p0065_m002",
    },
    {
        "question_id": "KR1_Q040",
        "question": "603절 검사사항에서 선저외판 검사를 위해 선박은 어떤 상태여야 하는가?",
        "expected_keywords": ["603", "선저외판", "입거"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 70,
        "gold_clause": "603",
        "page_band": "p051-p100",
        "note": "603. 검사사항(선체) p70 — kr_1_2025_p0070_m004",
    },
    {
        "question_id": "KR1_Q041",
        "question": "입거주기 연장제도 시행을 위해 선박소유자가 제출해야 하는 문서는?",
        "expected_keywords": ["입거주기", "제출", "문서"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 75,
        "gold_clause": "3",
        "page_band": "p051-p100",
        "note": "승인 및 검사 — kr_1_2025_p0075_m001",
    },
    {
        "question_id": "KR1_Q042",
        "question": "1101절 원격검사는 어떤 조건에서 실시할 수 있는가?",
        "expected_keywords": ["1101", "원격검사", "승인"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 85,
        "gold_clause": "1101",
        "page_band": "p051-p100",
        "note": "1101. 원격검사 — kr_1_2025_p0085_m004",
    },
    {
        "question_id": "KR1_Q043",
        "question": "1602절 연차검사의 목적은?",
        "expected_keywords": ["1602", "연차검사", "유지"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 95,
        "gold_clause": "1602",
        "page_band": "p051-p100",
        "note": "1602. 연차검사 — kr_1_2025_p0096_m007",
    },
    {
        "question_id": "KR1_Q044",
        "question": "1801절 선수갑판 작은 창구의 강도 규정은 어떤 선박에 적용되는가?",
        "expected_keywords": ["1801", "선수갑판", "2004"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 105,
        "gold_clause": "1801",
        "page_band": "p101-p150",
        "note": "1801 — kr_1_2025_p0105_m004",
    },
    {
        "question_id": "KR1_Q045",
        "question": "102절 검사계획서는 어떤 검사 전에 작성해야 하는가?",
        "expected_keywords": ["102", "검사계획서", "정기검사"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 109,
        "gold_clause": "102",
        "page_band": "p101-p150",
        "note": "102. 검사준비 — kr_1_2025_p0109_m000",
    },
    {
        "question_id": "KR1_Q046",
        "question": "105절 쇠모에 대한 허용기준은 어떤 선박에 적용되는가?",
        "expected_keywords": ["105", "쇠모", "IACS"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 115,
        "gold_clause": "105",
        "page_band": "p101-p150",
        "note": "105. 쇠모 허용기준 — kr_1_2025_p0115_m003",
    },
    {
        "question_id": "KR1_Q047",
        "question": "선령 5년 초과 10년 이하 산적화물선의 두께계측 범위는?",
        "expected_keywords": ["두께계측", "5년", "10년"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 125,
        "gold_clause": "4",
        "page_band": "p101-p150",
        "note": "두께계측 범위 — kr_1_2025_p0125_m002",
    },
    {
        "question_id": "KR1_Q048",
        "question": "정기검사 시 탱크 압력시험 최소범위는 어디에 따르는가?",
        "expected_keywords": ["탱크 압력시험", "표 1.3.6"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 135,
        "gold_clause": "5",
        "page_band": "p101-p150",
        "note": "5. 탱크 압력시험 — kr_1_2025_p0135_m004",
    },
    {
        "question_id": "KR1_Q049",
        "question": "502절 연차검사 시기는 어디 규정을 따르는가?",
        "expected_keywords": ["502", "연차검사", "201"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 145,
        "gold_clause": "502",
        "page_band": "p101-p150",
        "note": "502. 연차검사 — kr_1_2025_p0145_m002",
    },
    {
        "question_id": "KR1_Q050",
        "question": "정기검사 시 평형수탱크 검사는 언제 시행하는가?",
        "expected_keywords": ["평형수탱크", "정기검사", "검사가 필요"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 153,
        "gold_clause": "5",
        "page_band": "p151-p200",
        "note": "5. 평형수탱크 검사 — kr_1_2025_p0153_m003",
    },
    {
        "question_id": "KR1_Q051",
        "question": "이중선체 산적화물선 두께계측 범위(선령 5–10년)는?",
        "expected_keywords": ["이중선체", "두께계측", "산적화물선"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 155,
        "gold_clause": "4",
        "page_band": "p151-p200",
        "note": "두께계측 이중선체 — kr_1_2025_p0155_m005",
    },
    {
        "question_id": "KR1_Q052",
        "question": "선체관계 제출도면에는 일반배치도 외에 무엇을 포함해야 하는가?",
        "expected_keywords": ["제출도면", "일반배치도", "강재배치도"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 175,
        "gold_clause": "1",
        "page_band": "p151-p200",
        "note": "1. 제출도면 — kr_1_2025_p0175_m001",
    },
    {
        "question_id": "KR1_Q053",
        "question": "701절 일반에서 청수시료시험은 무엇을 참조하는가?",
        "expected_keywords": ["701", "청수시료", "IACS"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 185,
        "gold_clause": "701",
        "page_band": "p151-p200",
        "note": "701. 일반 — kr_1_2025_p0185_m005",
    },
    {
        "question_id": "KR1_Q054",
        "question": "903절 예방정비제도에서 확인검사 및 연차심사는 어떻게 정의되는가?",
        "expected_keywords": ["903", "예방정비", "확인검사"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 190,
        "gold_clause": "903",
        "page_band": "p151-p200",
        "note": "903. 예방정비제도 — kr_1_2025_p0190_m002",
    },
    {
        "question_id": "KR1_Q055",
        "question": "What is section 10 about in the survey programme (minimum thickness)?",
        "expected_keywords": ["Minimum thickness", "hull structures", "survey programme"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 230,
        "gold_clause": "10",
        "page_band": "p201-p250",
        "note": "English survey programme — kr_1_2025_p0230_m001",
    },
    {
        "question_id": "KR1_Q056",
        "question": "표 4 정기검사 시 두께계측에서 의심지역은 어떻게 다루는가?",
        "expected_keywords": ["표 4", "의심지역", "정기검사"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 275,
        "gold_clause": "",
        "gold_chunk_ids": ["kr_1_2025_p0275_e000"],
        "page_band": "p251-p300",
        "note": "표 4 두께계측 — kr_1_2025_p0275_e000",
    },
    {
        "question_id": "KR1_Q057",
        "question": "Thickness measurement report form: what is recorded in section 1?",
        "expected_keywords": ["thickness measurements", "transverse sections", "report form"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 310,
        "gold_clause": "1",
        "page_band": "p301-p350",
        "note": "English report form — kr_1_2025_p0310_m002_s01",
    },
    {
        "question_id": "KR1_Q058",
        "question": "CMS 기관장 점검의 확인검사에서 기관장 자격 요건은?",
        "expected_keywords": ["CMS", "기관장", "면허"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 335,
        "gold_clause": "2",
        "page_band": "p301-p350",
        "note": "CMS 기관장 — kr_1_2025_p0335_m002",
    },
    {
        "question_id": "KR1_Q059",
        "question": "상태 감시(CM) 및 상태 기반 정비(CBM)는 언제 승인되는가?",
        "expected_keywords": ["상태 감시", "CBM", "승인"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 350,
        "gold_clause": "4",
        "page_band": "p301-p350",
        "note": "4. 상태 감시 — kr_1_2025_p0350_m003",
    },
    {
        "question_id": "KR1_Q060",
        "question": "검사원의 수는 어떤 기준으로 결정되는가?",
        "expected_keywords": ["검사원", "건조 절차", "유자격"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 395,
        "gold_clause": "4",
        "page_band": "p351-p400",
        "note": "4. 검사원의 수 — kr_1_2025_p0395_m001",
    },
    {
        "question_id": "KR1_Q061",
        "question": "규칙 표 1.2.8 등에서 정하는 선종별 탱크 압력시험 범위는?",
        "expected_keywords": ["표 1.2.8", "탱크", "압력시험"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 325,
        "gold_clause": "1",
        "page_band": "p301-p350",
        "note": "표 참조 — kr_1_2025_p0324_m001",
    },
    {
        "question_id": "KR1_Q062",
        "question": "선급 승인 절차에서 선박소유자가 제출해야 하는 자료는?",
        "expected_keywords": ["승인", "제출", "표 1"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 345,
        "gold_clause": "1",
        "page_band": "p301-p350",
        "note": "1. 일반사항 승인 — kr_1_2025_p0345_m001",
    },
    {
        "question_id": "KR1_Q063",
        "question": "403절 계류장치 권고사항에서 선주가 고려할 해저면 조건은?",
        "expected_keywords": ["403", "계류장치", "해저면"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 455,
        "gold_clause": "403",
        "page_band": "p451-p462",
        "note": "403. 계류장치 — kr_1_2025_p0455_m004",
    },
    {
        "question_id": "KR1_Q064",
        "question": "503절 콜드 계선은 통상적으로 어떤 기간 경과 후에 적합한가?",
        "expected_keywords": ["503", "콜드 계선", "12개월"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 457,
        "gold_clause": "503",
        "page_band": "p451-p462",
        "note": "503. 콜드 계선 — kr_1_2025_p0457_m006",
    },
    {
        "question_id": "KR1_Q065",
        "question": "현측문 쌍립문 도면(완전개방)은 몇 번 도면인가?",
        "expected_keywords": ["현측문", "쌍립문", "34"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 430,
        "gold_clause": "",
        "gold_chunk_ids": ["kr_1_2025_p0430_e005"],
        "page_band": "p401-p450",
        "note": "도면 캡션 — kr_1_2025_p0430_e005",
    },
    {
        "question_id": "KR1_Q066",
        "question": "항내 평형수 적재 화물창에 대한 누설 및 구조시험 요건은?",
        "expected_keywords": ["평형수", "누설", "구조시험"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 445,
        "gold_clause": "",
        "gold_chunk_ids": ["kr_1_2025_p0445_m008"],
        "page_band": "p401-p450",
        "note": "화물창 시험 — kr_1_2025_p0445_m008",
    },
    {
        "question_id": "KR1_Q067",
        "question": "노출갑판 공기관헤드 외관검사는 어떤 규칙 항을 참조하는가?",
        "expected_keywords": ["공기관헤드", "외관검사", "202"],
        "gold_doc_id": "kr_1_2025",
        "gold_page": 460,
        "gold_clause": "",
        "gold_chunk_ids": ["kr_1_2025_p0459_e003"],
        "page_band": "p451-p462",
        "note": "부록 검사 — kr_1_2025_p0459_e003",
    },
]


def add_page_bands(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        row = dict(row)
        if "page_band" not in row:
            p = int(row["gold_page"])
            start = ((p - 1) // 50) * 50 + 1
            end = start + 49
            row["page_band"] = f"p{start:03d}-p{end:03d}"
        out.append(row)
    return out


def main() -> None:
    if not PILOT.exists():
        src = Path("data/eval/kr_1_2025_questions.jsonl")
        if src.exists():
            shutil.copy2(src, PILOT)
        else:
            raise FileNotFoundError("No pilot questions file to seed from.")

    pilot_rows = []
    with PILOT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pilot_rows.append(json.loads(line))

    merged = add_page_bands(pilot_rows) + EXTENSION
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pages = [int(r["gold_page"]) for r in merged]
    print(f"Wrote {len(merged)} questions to {OUT}")
    print(f"Page range: {min(pages)} – {max(pages)}")


if __name__ == "__main__":
    main()
