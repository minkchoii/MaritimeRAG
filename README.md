# MaritimeRAG Document Layout Preprocessing

DNV/KR/ABS 선급 PDF 문서를 대상으로 다음 전처리 단계를 수행합니다.

1. PDF -> 페이지 이미지 변환
2. YOLOv10 문서 레이아웃 탐지 (`text`, `title`, `table`, `figure`, `caption`, `list`, `header`, `footer`)
3. 페이지별 레이아웃 결과 JSON 저장
4. bbox 기반 요소 crop 이미지 저장

## 프로젝트 구조

요청하신 구조를 그대로 따릅니다.

## 설치

```bash
pip install -r requirements.txt
```

Windows 환경에서는 `pdf2image` 사용을 위해 Poppler 설치가 필요할 수 있습니다.



### OneDrive로 PDF 공유

1. OneDrive에 폴더 구조를 저장소와 맞춥니다.
   - 예: `MaritimeRAG-data/raw_pdfs/` → 폴더 그대로
2. 아래 링크를 본인 OneDrive 공유 URL로 바꿉니다.
3. 클론 후 링크에서 받아 `data/raw_pdfs/`에 풀고 파이프라인을 실행합니다.

| 자료 | OneDrive 링크 (본인 URL로 교체) |
|------|--------------------------------|
| 원본 PDF 전체 (`raw_pdfs`) | `https://onedrive.live.com/...` |
| (선택) 전처리 결과 (`processed`) | `https://onedrive.live.com/...` |
| (선택) 레이아웃 모델 `.pt` | `https://onedrive.live.com/...` |

> **주의:** 규칙 PDF는 저작권이 있습니다. **공개(public) GitHub·공개 OneDrive 링크**에 올리지 마세요.  
> 사내/팀 **제한 공유** 링크만 사용하세요.

### 레이아웃 모델 (Git 대신 다운로드)

```bash
# models/layout/yolov10m_doclaynet.pt
# Hugging Face: hantian/yolo-doclaynet → yolov10m-doclaynet.pt
```

## 준비

- 원본 PDF를 아래에 배치 (또는 OneDrive에서 받기):
  - `data/raw_pdfs/KR Rules`
  - `data/raw_pdfs/ABS rules`
  - `data/raw_pdfs/dnv-class_2026-04`
  - `data/raw_pdfs/MSC`
- 레이아웃 모델 가중치:
  - `models/layout/yolov10m_doclaynet.pt`

## 실행

### 1) 매니페스트 생성

`doc_id`는 **상대 경로 + SHA1 8자**로 고유화합니다. 파일럿 문서 `KR Rules/1편_2025.pdf`는 `kr_1_2025`로 고정됩니다.

```bash
python scripts/00_build_manifest.py --input-dir data/raw_pdfs --output data/manifests/pdf_manifest.csv
```

### 1-1) 문서별 RAG 배치 (전처리 + 인덱스 + eval)

```bash
# KR 1편 전편 (462p) — 레이아웃 모델 필요: models/layout/yolov10m_doclaynet.pt
python scripts/run_rag_batch.py --doc-id kr_1_2025 --run-eval

# 단계만 선택
python scripts/run_rag_batch.py --doc-id kr_1_2025 --steps pdf,layout,merge,crop,chunks,index

# 질의 (직접 입력 — JSONL 불필요)
python scripts/rag_query.py --doc-id kr_1_2025 --query "902절 탈급 절차는?"
python scripts/rag_query.py --doc-id kr_1_2025 -i --full-text --top-k 3

# 전편 질문세트 자동 채점 + 수동 스팟체크(상위 3청크 전문)
python scripts/13_rag_validate.py --doc-id kr_1_2025 --mode both

# 검색 개선 후 인덱스 재빌드 (조문 부스팅·임베딩 enrichment 반영)
python scripts/09_build_index.py --doc-id kr_1_2025
# 결과: data/processed/logs/kr_1_2025_retrieval_eval.txt
#       data/processed/logs/kr_1_2025_spot_check.txt  ← 직접 O/X 판정
```

질문 세트: `data/eval/kr_1_2025_questions.jsonl` (p3–p460, 67문항). 앞부분 34문항 백업: `data/eval/kr_1_2025_questions_pilot_30p.jsonl`.
```

### 2) PDF -> 페이지 이미지(PyMuPDF)

```bash
python scripts/01_pdf_to_images.py --pdf "data/raw_pdfs/dnv/sample.pdf" --doc-id DNV_SAMPLE --out-dir data/processed/pages --dpi 200 --start-page 1 --end-page 10
```

### 3) 레이아웃 탐지(JSON)

```bash
python scripts/02_layout_detect.py --image-dir data/processed/pages/DNV_SAMPLE --doc-id DNV_SAMPLE --model models/layout/yolov10m_doclaynet.pt --out-dir data/processed/layout_json --conf 0.25 --save-vis
```

### 4) 요소 크롭

```bash
# 원본 layout 기준
python scripts/03_crop_elements.py --doc-id kr_1_2025

# 병합 layout 기준 (layout_json_merged → crops_merged)
python scripts/03_crop_elements.py --doc-id kr_1_2025 --merged
```

### 4-1) Layout bbox 시각화 (병합 전후 비교)

```bash
python scripts/07_visualize_layout.py --doc-id kr_1_2025 --layout-dir data/processed/layout_json/kr_1_2025 --pages 11,28 --tag before
python scripts/07_visualize_layout.py --doc-id kr_1_2025 --layout-dir data/processed/layout_json_merged/kr_1_2025 --pages 11,28 --tag merged
```

### 5) 전체 파이프라인 일괄 실행

```bash
python scripts/04_run_preprocess_pipeline.py
```

### 5-1) Layout JSON image_path 정리

프로젝트를 옮긴 뒤 `image_path`가 OneDrive 등 옛 절대 경로로 남아 있으면:

```bash
# 전체 processed 산출물 (layout JSON + crops manifest + logs CSV)
python scripts/08_fix_layout_image_paths.py

# 특정 문서만
python scripts/08_fix_layout_image_paths.py --doc-id kr_1_2025
```

`data/processed/pages/<doc_id>/<page_name>` 기준으로 경로를 현재 프로젝트 루트(`c:/projects/MaritimeRAG`)로 맞춥니다.  
OneDrive 등 옛 절대 경로가 남아 있으면 이 스크립트를 실행하세요.

### 6) RAG 청크 추출 (유형별)

```bash
python scripts/07_extract_chunks.py --doc-id kr_1_2025
```

`layout_json_merged` + `crops_merged` + PDF 텍스트 레이어에서 청크를 만듭니다.

- 출력: `data/processed/chunks/<doc_id>/chunks.jsonl`
- 요약: `data/processed/chunks/<doc_id>/chunks_summary.txt`
- 기본 포함 유형: `text`, `table`, `picture`, `title`, `section-header` 등
- `caption`은 인접 `picture`/`figure` 청크 텍스트에 병합
- `page-header` / `page-footer`는 기본 제외 (`--include-headers`로 포함)

### 8) 벡터 인덱스 구축 (임베딩)

임베딩 정책 (`scripts/embedding_policy.py`):

- **로컬 전용**: `sentence-transformers`로 HuggingFace 모델을 PC에서 직접 실행 (OpenAI 등 API 사용 안 함)
- **한국어·영어**: 다국어(multilingual) 모델만 허용
- **중국계 차단**: BAAI/bge, Qwen, DeepSeek 등

```bash
pip install chromadb sentence-transformers
python scripts/09_build_index.py --doc-id kr_1_2025 --embedding-preset e5-base
```

| 프리셋 | 모델 | 비고 |
|--------|------|------|
| `e5-base` (기본) | `intfloat/multilingual-e5-base` | 로컬, 한·영, 규정 문서 권장 |
| `e5-large` | `intfloat/multilingual-e5-large` | 로컬, 한·영, 품질↑ |
| `snowflake-arctic` | `Snowflake/snowflake-arctic-embed-l-v2.0` | 로컬, 다국어 |
| `paraphrase-multilingual` | `paraphrase-multilingual-MiniLM-L12-v2` | 로컬, 가벼움 |

- 인덱스: `data/processed/index/<doc_id>/chroma/`
- 매니페스트: `data/processed/index/<doc_id>/index_manifest.json`
- 기본: `text`, `table`, `picture`(캡션 있음)만 인덱싱, `suspicious_chunks.csv` 제외

검색 테스트 (동일 로컬 모델):

```bash
python scripts/10_query_index.py --doc-id kr_1_2025 --query "14. 과도한 부식 정의는?"
```

### 검색 성능 평가용 질문 세트 (파일럿)

파일럿 문서 `kr_1_2025`(30페이지)에 대한 수동 골드 질문은 JSONL 한 줄당 한 질문으로 작성합니다.

- 파일: `data/eval/kr_1_2025_questions.jsonl`
- 범위: 현재 파이프라인이 처리한 페이지(1~30)와 `chunks.jsonl`에 존재하는 조항만 포함

#### 필드 설명

| 필드 | 설명 |
|------|------|
| `question_id` | 고유 ID (예: `KR1_Q001`) |
| `question` | 사용자가 검색·질의할 자연어 문장 (한국어·영어 모두 가능) |
| `expected_keywords` | 정답 청크 본문에 포함되어야 할 키워드 배열 (Recall·자동 채점용) |
| `gold_doc_id` | 정답 문서 ID (`kr_1_2025`) |
| `gold_page` | PDF 페이지 번호 (`chunks.jsonl`의 `page_number`와 동일) |
| `gold_clause` | 정답 조항 번호 (`chunks.jsonl`의 `clause_number`, 예: `"14"`) |
| `note` | 작성 근거·참고 청크 ID 등 (평가 로직에는 사용하지 않음) |

#### 작성 방법

1. **정답 위치 확인**: `data/processed/chunks/kr_1_2025/chunks.jsonl`에서 `page_number`, `clause_number`, `chunk_id`를 확인합니다.
2. **골드 라벨**: `gold_page`·`gold_clause`는 청크 메타데이터와 일치시킵니다. 규칙 본문이 `101.` 절 아래 `1.`~`n.` 항목인 경우 `gold_clause`는 **항목 번호**(예: `"9"`)만 적고, 절 번호는 `note`에 적습니다.
3. **키워드**: 정답 청크에 실제로 등장하는 한·영 표현을 2~5개 넣습니다. 동의어·약어를 함께 두면 다국어 검색 평가에 유리합니다.
4. **질문 유형**: 정의형, 적용일·절차, 비교(인접 조항 구분) 등을 골고루 넣습니다.
5. **한 줄 JSON**: UTF-8, 줄마다 완전한 JSON 객체 하나 (쉼표·배열 문법 주의).

예시 한 줄:

```json
{"question_id":"KR1_Q001","question":"과도한 부식의 정의는 무엇인가?","expected_keywords":["과도한 부식","substantial corrosion"],"gold_doc_id":"kr_1_2025","gold_page":28,"gold_clause":"14","note":"부식 정의 — p0028_m002"}
```

목표 규모는 파일럿 **20~50문항**입니다.

#### Recall@k 평가 실행

```bash
python scripts/11_eval_retrieval.py --doc-id kr_1_2025 --top-k 5
```

- 입력: `data/eval/kr_1_2025_questions.jsonl` (기본 경로)
- 지표: **Recall@k** (상위 k개 중 `gold_page` + `gold_clause` 일치), 페이지만 일치, 골드 청크 인덱스 포함 여부
- 출력:
  - `data/processed/logs/<doc_id>_retrieval_eval.json`
  - `data/processed/logs/<doc_id>_retrieval_eval.txt`

인덱스가 없거나 임베딩 프리셋이 다르면 먼저 `09_build_index.py`를 실행합니다.

### 9) Chunk 품질 자동 점검 (임베딩 전)

```bash
python scripts/08_analyze_chunks_quality.py --doc-id kr_1_2025
```

`chunks.jsonl`을 읽어 RAG 인덱싱 전 품질을 점검합니다.

- 리포트: `data/processed/logs/<doc_id>_chunk_quality_report.txt`
- 의심 청크 CSV: `data/processed/logs/<doc_id>_suspicious_chunks.csv`
- 타입별 샘플: `data/processed/logs/<doc_id>_chunk_samples.txt`

### 10) Text/List bbox 병합 (layout 후처리)

```bash
python scripts/00_build_manifest.py
python scripts/06_merge_text_blocks.py --doc-id kr_1_2025
```

같은 페이지 내 `text`, `list`/`list-item` bbox를 reading order 기준으로 병합합니다.  
PDF 텍스트 레이어에서 `13.`, `14.` 같은 **십진 조항 번호**가 바뀌면 병합을 끊습니다 (`pdf_manifest.csv`의 `file_path` 사용).

- 입력: `data/processed/layout_json/<doc_id>/page_*.json`
- 출력: `data/processed/layout_json_merged/<doc_id>/page_*.json`
- 비교 리포트: `data/processed/logs/<doc_id>_merge_comparison.txt`

병합 전후 layout 품질 비교:

```bash
python scripts/05_analyze_crop_quality.py --doc-id kr_1_2025 --layout-only
python scripts/05_analyze_crop_quality.py --doc-id kr_1_2025 --layout-dir data/processed/layout_json_merged --layout-only
# 리포트: ..._layout_before.txt / ..._layout_merged.txt
```

### 7) Crop 품질 자동 점검

```bash
python scripts/05_analyze_crop_quality.py --doc-id kr_1_2025
```

`crops_manifest.jsonl`과 `layout_json`을 함께 읽어 통계 리포트와 의심 crop 목록을 생성합니다.

- 리포트: `data/processed/logs/<doc_id>_crop_quality_report.txt`
- 의심 crop CSV: `data/processed/logs/<doc_id>_suspicious_crops.csv`
- 샘플 이미지: `outputs/quality_samples/<doc_id>/<element_type>/` (타입별 최대 20장)

#### 품질 판단 기준

개별 crop 기준 (`crops_manifest.jsonl`의 bbox 기준):

| 조건 | 임계값 | 의미 |
|------|--------|------|
| 너무 좁음 | `width < 40` | 가로가 지나치게 작아 내용이 잘릴 수 있음 |
| 너무 낮음 | `height < 20` | 세로가 지나치게 작아 한 줄 조각일 수 있음 |
| 면적 비율 과소 | `area_ratio < 0.0005` | 페이지 대비 bbox 면적이 지나치게 작음 |
| table 면적 과소 | `area_ratio < 0.01` (type=table) | 표로 보기엔 영역이 너무 작음 |
| table 폭 과소 | `width_ratio < 0.25` (type=table) | 페이지 폭 대비 표 폭이 지나치게 좁음 |
| figure 면적 과소 | `area_ratio < 0.01` (type=figure/picture) | 그림 영역이 지나치게 작음 |
| figure 폭 과소 | `width_ratio < 0.20` (type=figure/picture) | 그림 폭이 페이지 대비 지나치게 좁음 |

페이지 단위 기준 (`layout_json`의 탐지 element 수):

| 조건 | 임계값 | 의미 |
|------|--------|------|
| element 과다 | 한 페이지 전체 element 수 > 40 | 과탐지·중복 탐지 가능성 |
| text 과다 | 한 페이지 text element 수 > 20 | 본문이 과도하게 쪼개짐 |
| table 과다 | 한 페이지 table element 수 > 5 | 표 탐지가 과도함 |
| figure 과다 | 한 페이지 figure/picture element 수 > 5 | 그림 탐지가 과도함 |

`area_ratio` = crop bbox 면적 / 페이지 면적, `width_ratio` = crop bbox 폭 / 페이지 폭.

페이지 단위 조건에 해당하면 해당 페이지의 모든 crop이 의심 목록에 포함됩니다.

## 출력 경로

- 페이지 이미지: `data/processed/pages/<doc_id>/page_0001.png`
- 레이아웃 JSON: `data/processed/layout_json/<doc_id>/page_0001.json`
- 병합 레이아웃 JSON: `data/processed/layout_json_merged/<doc_id>/page_0001.json`
- 병합 비교 리포트: `data/processed/logs/<doc_id>_merge_comparison.txt`
- 크롭 이미지: `data/processed/crops/<doc_id>/`
- 병합 layout 크롭: `data/processed/crops_merged/<doc_id>/`
- 크롭 메타데이터: `.../crops_manifest.jsonl`
- layout 시각화: `outputs/visualized_layout/<doc_id>/{before,merged}/`
- RAG 청크: `data/processed/chunks/<doc_id>/chunks.jsonl`
- 품질 리포트: `data/processed/logs/<doc_id>_crop_quality_report.txt`
- 의심 crop CSV: `data/processed/logs/<doc_id>_suspicious_crops.csv`
- 청크 품질 리포트: `data/processed/logs/<doc_id>_chunk_quality_report.txt`
- 의심 청크 CSV: `data/processed/logs/<doc_id>_suspicious_chunks.csv`
- 검색 평가 질문: `data/eval/<doc_id>_questions.jsonl`

## 비고

현재 단계는 요청하신 범위대로 `PDF -> page image -> layout JSON -> crop image`까지 구현되어 있습니다.
