"""Small UTF-8 atomic-write helpers for important application metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """Flush a sibling temporary file before replacing ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Serialize UTF-8 JSON without allowing non-standard NaN values."""
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
