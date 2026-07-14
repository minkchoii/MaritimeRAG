# MaritimeRAG

MaritimeRAG는 선급 규칙과 IMO 회의 문서를 로컬에서 전처리하고 검색·요약하는 RAG 프로젝트입니다. PDF 레이아웃 분석, 본문·표 청크 생성, 다국어 벡터 검색과 BM25 하이브리드 검색, 근거 기반 답변, Streamlit UI를 제공합니다.

> 원본 선급 규칙 PDF와 일부 IMO 자료에는 저작권 또는 배포 제한이 있을 수 있습니다. 원본 문서, 모델 가중치, 생성된 인덱스는 Git 저장소에 포함하지 않습니다.

## 현재 구현 범위

- PDF 페이지 렌더링과 YOLOv10 기반 문서 레이아웃 탐지
- 텍스트 블록 병합, 표·그림 crop, 본문·표·그림 청크 생성
- multilingual E5 계열 임베딩과 Chroma 벡터 인덱스
- BM25 + 벡터 검색, 문서 메타데이터 재정렬, 선급사 필터
- IMO MSC/MEPC 회의 결과 및 동향 요약
- DNV/KR/ABS/LR Rule·Guidance 검색
- 표 schema·summary·markdown·row 기반 표 질의
- 빠른 모드와 정확 모드, Ollama 기반 로컬 답변 생성
- Recall@k, 표 검색, 응답 지연시간 및 회귀 평가
- 리소스 캐시와 스트리밍 답변을 적용한 Streamlit UI

## Corpus 현황

`data/manifests/rag_corpus_457.csv` 기준 문서 구성은 다음과 같습니다.

| 출처 | 문서 수 |
|---|---:|
| MEPC | 203 |
| MSC | 106 |
| KR | 73 |
| DNV | 60 |
| ABS | 14 |
| LR | 1 |
| 합계 | 457 |

주요 collection ID:

- `full_corpus`: IMO 회의 자료와 선급 Rule/Guidance 통합 검색
- `kr_tables`: KR 표 질의용 구조화 표 인덱스
- `pilot_100`: 100개 문서 파일럿 인덱스

Manifest와 평가 데이터는 Git에 포함되지만 PDF, 전처리 산출물, Chroma/BM25 인덱스는 로컬에서 준비하거나 별도 저장소에서 받아야 합니다.

## 권장 환경

- Windows 10/11 또는 Linux
- Python 3.11 권장(현재 개발 환경: Python 3.11.9)
- NVIDIA GPU 권장
  - 레이아웃 탐지와 대규모 임베딩은 CPU에서도 가능하지만 오래 걸립니다.
- Ollama
  - 기본 답변 모델: `llama3.1:8b`
  - 기본 API 주소: `http://localhost:11434`
- 레이아웃 모델
  - `models/layout/yolov10m_doclaynet.pt`
  - Hugging Face의 DocLayNet 호환 YOLOv10 가중치를 사용합니다.

PDF 렌더링은 PyMuPDF를 사용하므로 현재 파이프라인에는 별도 Poppler 설치가 필요하지 않습니다.

## 설치

PowerShell 예시:

```powershell
git clone https://github.com/minkchoii/MaritimeRAG.git
cd MaritimeRAG

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ollama 설치 후 기본 모델을 준비합니다.

```powershell
ollama pull llama3.1:8b
ollama serve
```

Ollama가 이미 백그라운드에서 실행 중이면 `ollama serve`를 다시 실행할 필요가 없습니다. UI에서 다른 설치된 모델을 지정할 수도 있습니다.

## 데이터 배치

원본 문서를 다음 구조로 배치합니다.

```text
data/raw_pdfs/
├─ ABS rules/
├─ dnv-class-2026-04/
├─ KR Rules/
├─ LR Rules/
├─ MEPC/
└─ MSC/
```

레이아웃 모델은 다음 위치에 둡니다.

```text
models/layout/yolov10m_doclaynet.pt
```

원본 PDF와 모델 파일은 `.gitignore`로 제외되어 있습니다. 공개 저장소나 공개 공유 링크에 업로드하지 마십시오.

## 빠른 실행

이미 `full_corpus`와 `kr_tables` 인덱스가 준비되어 있다면 다음 명령으로 UI를 실행합니다.

```powershell
streamlit run scripts/15_rag_ui.py
```

UI는 기본적으로 다음 리소스를 사용합니다.

- Chroma 인덱스: `data/processed/index/`
- 청크: `data/processed/chunks/`
- 일반 검색 collection: `full_corpus`
- 표 검색 collection: `kr_tables`
- Ollama 모델: `llama3.1:8b`

## 전처리와 인덱스 구축

### 1. PDF manifest 생성

```powershell
python scripts/00_build_manifest.py `
  --input-dir data/raw_pdfs `
  --output data/manifests/pdf_manifest.csv
```

### 2. 문서별 전처리

한 문서를 전체 단계로 처리하는 예시입니다.

```powershell
python scripts/run_rag_batch.py `
  --doc-id kr_1_2025 `
  --steps pdf,layout,merge,crop,chunks,index
```

주요 산출물:

```text
data/processed/pages/<doc_id>/
data/processed/layout_json/<doc_id>/
data/processed/layout_json_merged/<doc_id>/
data/processed/crops_merged/<doc_id>/
data/processed/chunks/<doc_id>/chunks.jsonl
data/processed/index/<doc_id>/
```

### 3. 통합 벡터 인덱스 구축

457개 corpus manifest를 이용하는 예시입니다.

```powershell
python scripts/10_build_unified_index.py `
  --doc-list data/manifests/rag_corpus_457.csv `
  --collection-id full_corpus `
  --embedding-preset e5-base
```

기본 임베딩 preset은 `e5-base`이며 `intfloat/multilingual-e5-base`를 사용합니다. 임베딩 모델이나 청크 표현을 변경하면 기존 인덱스를 재사용하지 말고 다시 구축해야 합니다.

### 4. BM25 인덱스 구축

```powershell
python scripts/35_build_bm25_index.py --unified full_corpus --rebuild
```

### 5. 표 인덱스

표 추출과 schema 청크 생성에 관련된 주요 스크립트는 다음과 같습니다.

```text
scripts/07b_extract_table_chunks.py
scripts/33_regen_table_schema_chunks.py
scripts/table_schema_lib.py
scripts/table_schema_retrieval.py
```

표 인덱스는 `kr_tables` collection으로 분리하여 운영합니다. 전건 구축 전에는 `data/manifests/kr_table_top22.csv`로 소규모 검증을 먼저 수행하는 것을 권장합니다.

## CLI 검색과 평가

대화형 검색:

```powershell
python scripts/rag_query.py --unified full_corpus -i --full-text --top-k 5
```

파일럿 검색 평가:

```powershell
python scripts/13_rag_pilot_validation.py `
  --unified full_corpus `
  --skip-llm `
  --top-k 8
```

표 검색 평가:

```powershell
python scripts/30_table_schema_retrieval_benchmark.py
```

기본 단위 테스트:

```powershell
python scripts/test_hybrid_retrieval.py
python scripts/test_rule_lookup_answer.py
```

## Ollama와 답변 모드

기본 설정은 `scripts/rag_answer_lib.py`에 정의되어 있습니다.

| 항목 | 기본값 |
|---|---|
| provider | Ollama |
| model | `llama3.1:8b` |
| base URL | `http://localhost:11434` |
| 빠른 모드 | 검색 및 응답 지연 최소화 |
| 정확 모드 | 더 넓은 문맥과 근거 기반 LLM 합성 |

일부 평가 스크립트는 OpenAI provider도 지원하며, 이 경우 키를 코드나 설정 파일에 저장하지 말고 환경변수로 전달합니다.

```powershell
$env:OPENAI_API_KEY = "..."
```

## 인코딩

저장소의 README, Python, CSV, JSONL 파일은 UTF-8을 기준으로 합니다. Windows PowerShell의 출력 코드페이지 때문에 정상적인 한글이 깨져 보일 수 있습니다.

PowerShell에서 다음 명령을 먼저 실행하면 UTF-8 출력 문제를 줄일 수 있습니다.

```powershell
chcp 65001
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
```

Python 파일 인코딩 확인 예시:

```powershell
python -c "from pathlib import Path; Path('README.md').read_text(encoding='utf-8'); print('UTF-8 OK')"
```

## Git에 포함하지 않는 파일

- 원본 PDF와 배포 제한 자료: `data/raw_pdfs/`
- 페이지, crop, chunk, 로그, 인덱스: `data/processed/`
- 레이아웃 및 임베딩 모델 가중치: `models/`
- UI·진단 출력: `outputs/`
- Python 가상환경과 캐시
- API 키, `.env`, 인증 파일

대용량 인덱스를 공유할 때는 별도 스토리지를 사용하고, Git에는 생성 명령, embedding preset, corpus manifest, 평가 결과와 체크섬만 기록하는 방식을 권장합니다.

## 프로젝트 구조

```text
MaritimeRAG/
├─ scripts/                  # 전처리, 검색, 답변, UI, 평가 코드
├─ data/
│  ├─ eval/                  # 평가 질문과 회귀 테스트
│  ├─ manifests/             # 문서 corpus와 처리 목록
│  ├─ raw_pdfs/              # 로컬 원본 문서(Git 제외)
│  └─ processed/             # 생성 산출물과 인덱스(Git 제외)
├─ models/layout/            # 레이아웃 모델(Git 제외)
├─ outputs/                  # 시각화·진단 출력(Git 제외)
├─ .streamlit/config.toml
├─ requirements.txt
└─ README.md
```
