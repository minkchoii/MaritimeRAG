"""Capture Ollama GPU/CPU processor state and nvidia-smi for benchmark logging."""
from __future__ import annotations

import json
import subprocess
import urllib.request
from typing import Any


def _run(cmd: list[str], *, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def fetch_ollama_ps_api(base_url: str = "http://127.0.0.1:11434") -> list[dict]:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return list(data.get("models") or [])
    except Exception:
        return []


def parse_ollama_ps_cli() -> list[dict]:
    text = _run(["ollama", "ps"])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 6:
            continue
        name = parts[0]
        model_id = parts[1]
        size = f"{parts[2]} {parts[3]}" if len(parts) > 3 and parts[3] in ("GB", "MB", "B") else parts[2]
        if len(parts) > 5 and parts[4].endswith("%") and parts[5] == "GPU":
            processor = f"{parts[4]} {parts[5]}"
            context = parts[6] if len(parts) > 6 else ""
        else:
            processor = parts[4] if len(parts) > 4 else ""
            context = parts[5] if len(parts) > 5 else ""
        rows.append(
            {
                "name": name,
                "id": model_id,
                "size": size,
                "processor": processor,
                "context": context,
            }
        )
    return rows


def snapshot_nvidia_smi() -> dict[str, Any]:
    gpu_name = _run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        timeout=15,
    )
    procs = _run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        timeout=15,
    )
    llama_on_gpu = "llama-server" in procs.lower() or "ollama" in procs.lower()
    vram_used_mb: int | None = None
    gpu_detected = False
    name = ""
    if gpu_name and not gpu_name.startswith("ERROR"):
        gpu_detected = True
        first = gpu_name.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if parts:
            name = parts[0]
        if len(parts) >= 2:
            try:
                vram_used_mb = int(float(parts[1]))
            except ValueError:
                pass
    return {
        "gpu_detected": gpu_detected,
        "gpu_name": name or "RTX 3090",
        "gpu_vram_used_mb": vram_used_mb,
        "llama_server_on_gpu": llama_on_gpu,
        "nvidia_smi_processes": procs[:2000],
        "nvidia_smi_raw": gpu_name[:500],
    }


def _processor_from_api_model(m: dict) -> str:
    proc = str(m.get("processor") or "").strip()
    if proc and not proc.isdigit():
        return proc
    size_vram = m.get("size_vram") or 0
    size = m.get("size") or 0
    try:
        size_vram = int(size_vram)
        size = int(size)
    except (TypeError, ValueError):
        size_vram = size = 0
    if size_vram > 0 and size > 0:
        if size_vram >= size * 0.85:
            return "100% GPU"
        return "GPU+CPU"
    if size_vram > 0:
        return "GPU"
    return ""


def _processor_from_cli(model_name: str | None, cli_models: list[dict]) -> str:
    if not model_name:
        return cli_models[0].get("processor", "") if cli_models else ""
    base = model_name.split(":")[0]
    for row in cli_models:
        name = str(row.get("name") or "")
        if base in name or name.split(":")[0] in model_name:
            return str(row.get("processor") or "")
    return ""


def snapshot_ollama_env(
    model_name: str | None = None,
    base_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    api_models = fetch_ollama_ps_api(base_url)
    cli_models = parse_ollama_ps_cli()
    models = api_models or cli_models

    processor = ""
    ollama_context: int | None = None
    matched: dict | None = None
    for m in models:
        mname = str(m.get("name") or m.get("model") or "")
        if model_name and model_name.split(":")[0] not in mname and mname.split(":")[0] not in model_name:
            continue
        matched = m
        processor = _processor_from_api_model(m)
        ctx = m.get("context") or m.get("context_length")
        if ctx is not None:
            try:
                ollama_context = int(ctx)
            except (TypeError, ValueError):
                pass
        break

    if not matched and models:
        matched = models[0]
        processor = _processor_from_api_model(matched)
        ctx = matched.get("context") or matched.get("context_length")
        if ctx is not None:
            try:
                ollama_context = int(ctx)
            except (TypeError, ValueError):
                pass

    cli_proc = _processor_from_cli(model_name, cli_models)
    if cli_proc:
        processor = cli_proc

    gpu = snapshot_nvidia_smi()
    proc_upper = processor.upper()
    gpu_in_processor = "GPU" in proc_upper or "100% GPU" in proc_upper

    return {
        "model_name": model_name or (models[0].get("name") if models else ""),
        "ollama_processor": processor or ("GPU" if gpu["llama_server_on_gpu"] else "CPU"),
        "ollama_context": ollama_context,
        "ollama_ps_models": models,
        "ollama_ps_cli": _run(["ollama", "ps"])[:1500],
        "gpu_detected": gpu["gpu_detected"],
        "gpu_name": gpu["gpu_name"],
        "gpu_vram_used_mb": gpu["gpu_vram_used_mb"],
        "llama_server_on_gpu": gpu["llama_server_on_gpu"],
        "gpu_in_processor_field": gpu_in_processor,
    }


def warmup_ollama_model(model: str, base_url: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    """Legacy /api/generate num_ctx=512 warm-up (use ollama_warmup.warmup_fast_chat for Fast mode)."""
    from ollama_warmup import legacy_warmup_generate

    result = legacy_warmup_generate(model, base_url)
    env = snapshot_ollama_env(model, base_url)
    return {
        "warmup_elapsed_s": result.get("warmup_elapsed_s"),
        "warmup_mode": "legacy",
        "warmup_response": result.get("warmup_response"),
        "env_after_warmup": env,
    }
