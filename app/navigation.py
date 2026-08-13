"""Stable page identifiers and a small navigation intent controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class NavigationRequest:
    """A limited navigation request, optionally carrying page-owned context."""

    target: str
    context: Any = None


class NavigationController(QObject):
    """Emits validated navigation intents without importing concrete pages."""

    requested = Signal(object)

    def __init__(self, targets: set[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._targets = frozenset(targets)

    def go_to(self, target: str, context: Any = None) -> bool:
        if target not in self._targets:
            return False
        self.requested.emit(NavigationRequest(target, context))
        return True
