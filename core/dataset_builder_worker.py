from __future__ import annotations

from datetime import datetime
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from core.dataset_builder import build_dataset, preview_dataset_build


class DatasetBuilderWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    finished = Signal(dict)
    cancelled = Signal(dict)
    failed = Signal(str)

    def __init__(self, mode: str, options: dict[str, Any]) -> None:
        super().__init__()
        self.mode = mode
        self.options = dict(options)
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        self.log.emit(f"Started at {datetime.now().astimezone().isoformat(timespec='seconds')}")
        try:
            function = preview_dataset_build if self.mode == "preview" else build_dataset
            result = function(
                self.options,
                progress_callback=self._emit_progress,
                log_callback=self.log.emit,
                cancel_callback=self._cancel_event.is_set,
            )
        except Exception as exc:
            self.failed.emit(f"Unexpected Dataset Builder worker failure: {exc}")
            return
        if self._cancel_event.is_set() or result.get("cancelled"):
            result["cancelled"] = True
            self.cancelled.emit(result)
        else:
            self.finished.emit(result)

    def _emit_progress(self, value: int, status: str) -> None:
        self.progress.emit(value, status)
