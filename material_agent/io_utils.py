from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p


def read_yaml(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        import yaml  # type: ignore
    except Exception:
        return read_json(p, default)
    with p.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    return default if value is None else value


def write_yaml(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
    except Exception:
        return write_json(p, data)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return p


def safe_name(value: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    return text.strip("_") or "item"


def relative_or_abs(path: str | Path | None, base: str | Path) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((Path(base) / p).resolve())

