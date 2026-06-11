"""
Single-figure VLM pilot: caption + VLM description -> E5 embedding.

Usage (one image):
  python scripts/12_figure_vlm_pilot.py \\
    --chunk-id kr_1_2025_p0030_e000 \\
    --vlm-provider ollama --vlm-model llava:latest --output-suffix llava

  python scripts/12_figure_vlm_pilot.py \\
    --chunk-id kr_1_2025_p0030_e000 \\
    --description-file data/processed/figures/kr_1_2025/p0030_e000_description_manual.txt \\
    --output-suffix manual

Output: data/processed/figures/kr_1_2025/p0030_e000_vlm_llava.json
        (without --output-suffix: p0030_e000_vlm.json)

Requires: Ollama running with a vision model, or OPENAI_API_KEY for openai provider.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config

PLACEHOLDER_MARKERS = ("[picture element", "refer to crop image")
FIGURE_LABEL_RE = re.compile(r"그림\s*[\d.]+|figure\s*[\d.]+", re.IGNORECASE)
DECORATIVE_PAGE_MAX = 2  # cover pages: skip unless caption/figure label present

VLM_PROMPT = """해양 선급 규칙 PDF의 기술 그림을 검색용으로 설명하세요.
반드시 한국어로 작성하고, 핵심 용어에 영문을 괄호로 병기하세요.
다음만 포함: 도면 종류(단면·개략도 등), 라벨된 구조 부재명, 치수·기호(h_stf, b_stf, 0.25 등),
부식 표현 방식, 검색에 유용한 기술 키워드.
이미지에 없는 규정·수치는 추측하지 마세요. 120~200자, 문단 하나, 마크다운 금지."""


def load_chunk(chunks_path: Path, chunk_id: str) -> dict:
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("chunk_id") == chunk_id:
                return row
    raise KeyError(f"chunk_id not found: {chunk_id}")


def resolve_crop_path(chunk: dict, project_root: Path) -> Path:
    for key in ("crop_path",):
        raw = str(chunk.get(key, "")).strip()
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            return p
        alt = project_root / "data" / "processed" / "crops" / chunk["doc_id"] / "picture" / p.name
        if alt.exists():
            return alt
        alt2 = project_root / "data" / "processed" / "crops_merged" / chunk["doc_id"] / "picture" / p.name
        if alt2.exists():
            return alt2
    raise FileNotFoundError(f"crop not found for chunk {chunk.get('chunk_id')}")


def is_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in PLACEHOLDER_MARKERS)


def is_decorative(chunk: dict, text: str) -> bool:
    page = int(chunk.get("page_number", 0))
    if page <= DECORATIVE_PAGE_MAX and not FIGURE_LABEL_RE.search(text):
        if is_placeholder(text) or len(text.strip()) < 20:
            return True
    if is_placeholder(text) and not chunk.get("linked_caption_id"):
        return True
    return False


def extract_caption_text(chunk: dict) -> str:
    """Preserve existing caption / PDF text from chunk (step 3)."""
    return str(chunk.get("text", "")).strip()


def build_content_for_embedding(caption: str, description: str, chunk: dict) -> str:
    parts: list[str] = []
    doc_id = chunk.get("doc_id", "")
    page = chunk.get("page_number", "")
    chunk_id = chunk.get("chunk_id", "")
    header = f"[figure] doc={doc_id} page={page} chunk={chunk_id}"
    parts.append(header)
    if caption:
        parts.append(f"caption: {caption}")
    if description:
        parts.append(f"description: {description}")
    return "\n".join(parts)


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_ollama_vlm(model: str, image_path: Path, prompt: str, base_url: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_to_base64(image_path)]}],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed ({base_url}): {exc}") from exc
    message = body.get("message") or {}
    text = str(message.get("content", "")).strip()
    if not text:
        raise RuntimeError(f"Empty Ollama VLM response: {body}")
    return text


def call_openai_vlm(model: str, image_path: Path, prompt: str) -> str:
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    b64 = image_to_base64(image_path)
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        max_tokens=600,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Empty OpenAI VLM response")
    return text


def run_vlm(provider: str, model: str, image_path: Path, ollama_base: str) -> str:
    if provider == "ollama":
        return call_ollama_vlm(model, image_path, VLM_PROMPT, ollama_base)
    if provider == "openai":
        return call_openai_vlm(model, image_path, VLM_PROMPT)
    raise ValueError(f"Unknown vlm provider: {provider}")


def output_json_path(output_dir: Path, stem: str, output_suffix: str | None) -> Path:
    if output_suffix:
        return output_dir / f"{stem}_vlm_{output_suffix}.json"
    return output_dir / f"{stem}_vlm.json"


def validate_output_suffix(suffix: str) -> str:
    cleaned = suffix.strip()
    if not cleaned:
        raise ValueError("--output-suffix must not be empty")
    if not re.fullmatch(r"[\w.-]+", cleaned):
        raise ValueError(
            f"Invalid --output-suffix {suffix!r}: use letters, digits, underscore, hyphen, or dot"
        )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot: one figure -> VLM description -> E5 embedding")
    parser.add_argument("--doc-id", default="kr_1_2025")
    parser.add_argument("--chunk-id", default="kr_1_2025_p0030_e000")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks/kr_1_2025/chunks.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/figures/kr_1_2025"),
    )
    parser.add_argument("--vlm-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--vlm-model", default="llava:latest")
    parser.add_argument("--ollama-base", default="http://localhost:11434")
    parser.add_argument("--embedding-preset", default=DEFAULT_EMBEDDING_PRESET)
    parser.add_argument("--skip-vlm", action="store_true", help="Only build embedding from existing description file")
    parser.add_argument("--description-file", type=Path, default=None)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Output filename suffix: {stem}_vlm_{suffix}.json (e.g. llava, manual)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSON if it already exists (default: abort)",
    )
    args = parser.parse_args()

    output_suffix = validate_output_suffix(args.output_suffix) if args.output_suffix else None

    project_root = Path.cwd()
    chunk = load_chunk(args.chunks_path, args.chunk_id)
    if str(chunk.get("element_type", "")).lower() != "picture":
        raise ValueError(f"chunk is not picture: {args.chunk_id}")

    stem = args.chunk_id.replace(f"{args.doc_id}_", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_json_path(args.output_dir, stem, output_suffix)

    if out_json.exists() and not args.overwrite:
        print(f"Output already exists (use --overwrite): {out_json}", file=sys.stderr)
        sys.exit(1)

    caption = extract_caption_text(chunk)
    if is_decorative(chunk, caption):
        print(f"SKIP decorative/placeholder: {args.chunk_id}")
        sys.exit(0)

    crop_path = resolve_crop_path(chunk, project_root)
    print(f"Crop: {crop_path}")

    if args.description_file:
        description = args.description_file.read_text(encoding="utf-8").strip()
        vlm_provider = "manual"
        vlm_model = "manual"
    elif args.skip_vlm and out_json.exists():
        prior = json.loads(out_json.read_text(encoding="utf-8"))
        description = str(prior.get("vlm_description", "")).strip()
        vlm_provider = prior.get("vlm_provider", args.vlm_provider)
        vlm_model = prior.get("vlm_model", args.vlm_model)
    else:
        print(f"VLM: {args.vlm_provider} / {args.vlm_model} ...")
        description = run_vlm(args.vlm_provider, args.vlm_model, crop_path, args.ollama_base)
        vlm_provider = args.vlm_provider
        vlm_model = args.vlm_model
        print(f"Description ({len(description)} chars):\n{description[:400]}...")

    content = build_content_for_embedding(caption, description, chunk)
    emb_cfg = resolve_embedding_config(args.embedding_preset)
    model_name = str(emb_cfg["model"])
    vector = embed_texts_local([content], model_name)[0]

    record = {
        "chunk_id": args.chunk_id,
        "doc_id": args.doc_id,
        "page_number": chunk.get("page_number"),
        "crop_path": str(crop_path),
        "caption_text": caption,
        "vlm_description": description,
        "content_for_embedding": content,
        "vlm_provider": vlm_provider,
        "vlm_model": vlm_model,
        "output_suffix": output_suffix,
        "embedding_preset": args.embedding_preset,
        "embedding_model": emb_cfg["model"],
        "embedding_dim": len(vector),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")
    print(f"Embedding dim: {len(vector)} ({emb_cfg['model']})")


if __name__ == "__main__":
    main()
