"""Small, resource-based application internationalization service.

The manager deliberately knows nothing about PySide pages.  UI code consumes
``tr`` or listens to ``language_changed``; command arguments and persisted
schemas remain their original, stable values.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal


LOGGER = logging.getLogger(__name__)
DEFAULT_LANGUAGE = "zh_TW"
FALLBACK_LANGUAGE = "en_US"


class I18nManager(QObject):
    language_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._language = DEFAULT_LANGUAGE
        self._resources: dict[str, dict[str, str]] = {}
        self._load_resources()

    @staticmethod
    def resource_directory() -> Path:
        """Return the bundled i18n directory in source and frozen builds."""
        candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            candidates.extend((bundle / "i18n", Path(sys.executable).parent / "i18n"))
        candidates.append(Path(__file__).resolve().parents[1] / "i18n")
        return next((path for path in candidates if path.is_dir()), candidates[-1])

    def _load_resources(self) -> None:
        for locale in (FALLBACK_LANGUAGE, DEFAULT_LANGUAGE):
            path = self.resource_directory() / f"{locale}.json"
            try:
                with path.open("r", encoding="utf-8") as handle:
                    content = json.load(handle)
                if not isinstance(content, dict):
                    raise ValueError("translation root must be an object")
                self._resources[locale] = {str(key): str(value) for key, value in content.items()}
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("Unable to load %s: %s", path, exc)
                self._resources[locale] = {}

    def available_languages(self) -> dict[str, str]:
        return {"zh_TW": "繁體中文", "en_US": "English"}

    def get_language(self) -> str:
        return self._language

    def set_language(self, locale: str | None, *, emit: bool = True) -> str:
        selected = locale if locale in self.available_languages() else DEFAULT_LANGUAGE
        changed = selected != self._language
        self._language = selected
        if changed and emit:
            self.language_changed.emit(selected)
        return selected

    def translate(self, key: str, **kwargs: object) -> str:
        value = self._resources.get(self._language, {}).get(key)
        if value is None:
            value = self._resources.get(FALLBACK_LANGUAGE, {}).get(key)
        if value is None:
            LOGGER.debug("Missing translation key: %s", key)
            return key
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError, ValueError) as exc:
            LOGGER.warning("Translation format error for %s: %s", key, exc)
            return value


_MANAGER = I18nManager()


def get_i18n() -> I18nManager:
    return _MANAGER


def tr(key: str, **kwargs: object) -> str:
    return _MANAGER.translate(key, **kwargs)
