"""Validate the resource files used by YOLO Trainer UI i18n."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load(path: Path) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    try:
        pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=list)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Invalid JSON in {path}: {exc}"]
    if not isinstance(pairs, list):
        return {}, [f"Translation root is not an object: {path}"]
    values: dict[str, str] = {}
    for key, value in pairs:
        if key in values:
            errors.append(f"Duplicate key in {path.name}: {key}")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Empty/non-string translation in {path.name}: {key}")
        values[str(key)] = str(value)
    return values, errors


def main() -> int:
    en, errors = load(ROOT / "i18n" / "en_US.json")
    zh, zh_errors = load(ROOT / "i18n" / "zh_TW.json")
    errors.extend(zh_errors)
    for key in sorted(set(en) - set(zh)):
        errors.append(f"Missing in zh_TW: {key}")
    for key in sorted(set(zh) - set(en)):
        errors.append(f"Missing in en_US: {key}")
    for key in sorted(set(en) & set(zh)):
        if set(PLACEHOLDER.findall(en[key])) != set(PLACEHOLDER.findall(zh[key])):
            errors.append(f"Placeholder mismatch for {key}")
    print(f"en_US keys: {len(en)}")
    print(f"zh_TW keys: {len(zh)}")
    if errors:
        print("Translation validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Translation validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
