from __future__ import annotations

import os
import subprocess
import threading

from PySide6.QtCore import QObject, Signal, Slot

from core.runtime_manager import CREATE_NO_WINDOW, RuntimeManager


class RuntimeWorker(QObject):
    progress = Signal(int)
    step_changed = Signal(str)
    log = Signal(str)
    diagnostics_ready = Signal(dict)
    settings_changed = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, manager: RuntimeManager, action: str) -> None:
        super().__init__()
        self.manager = manager
        self.action = action
        self._cancelled = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._process_lock:
            process = self._process
        if process and process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                    creationflags=CREATE_NO_WINDOW,
                    check=False,
                )
            except OSError:
                try:
                    process.terminate()
                except OSError:
                    pass

    @Slot()
    def run(self) -> None:
        try:
            if self.action == "diagnostics":
                self._run_diagnostics()
            elif self.action == "create":
                self._create_managed_runtime()
            else:
                self.finished.emit(False, f"Unknown runtime action: {self.action}")
        except Exception as exc:  # Worker boundary: report every failure to the UI.
            self.log.emit(f"ERROR: {exc}\n")
            self.finished.emit(False, str(exc))

    def _run_diagnostics(self) -> None:
        self.step_changed.emit("Running diagnostics...")
        self.progress.emit(0)
        result = self.manager.run_diagnostics()
        self.progress.emit(100)
        self.diagnostics_ready.emit(result)
        self.step_changed.emit("Completed")
        self.finished.emit(True, "Diagnostics completed.")

    def _create_managed_runtime(self) -> None:
        base_python = self.manager.discover_python(validate=True)
        if not base_python.get("available"):
            self.finished.emit(False, "A usable Python installation was not found. Configure Python in Settings first.")
            return

        target = self.manager.managed_runtime_folder()
        managed_python = self.manager.managed_python()
        managed_yolo = self.manager.managed_yolo()
        steps: list[tuple[str, list[str], int]] = [
            (
                "Creating virtual environment",
                [str(base_python["program"]), *base_python.get("prefix_args", []), "-m", "venv", str(target)],
                10,
            ),
            ("Upgrading pip", [str(managed_python), "-m", "pip", "install", "--upgrade", "pip"], 30),
            ("Installing Ultralytics", [str(managed_python), "-m", "pip", "install", "ultralytics"], 55),
            (
                "Verifying Ultralytics",
                [str(managed_python), "-c", "import ultralytics; print(ultralytics.__version__)"],
                75,
            ),
            ("Checking PyTorch", [str(managed_python), "-c", "import torch; print(torch.__version__)"], 82),
            (
                "Checking CUDA",
                [str(managed_python), "-c", "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"],
                88,
            ),
            ("Verifying YOLO command", [str(managed_yolo), "version"], 92),
        ]

        target.parent.mkdir(parents=True, exist_ok=True)
        for step, command, progress in steps:
            if self._cancelled.is_set():
                self._cancelled_result()
                return
            self.step_changed.emit(step)
            self.progress.emit(progress)
            self.log.emit(f"\n[{step}]\n> {subprocess.list2cmdline(command)}\n")
            code = self._run_process(command)
            if self._cancelled.is_set():
                self._cancelled_result()
                return
            if code != 0:
                self.finished.emit(False, f"{step} failed with exit code {code}.")
                return

        if not managed_python.is_file() or not managed_yolo.is_file():
            self.finished.emit(False, "Managed runtime verification failed: python.exe or yolo.exe is missing.")
            return

        self.step_changed.emit("Saving runtime configuration")
        self.progress.emit(96)
        values = self.manager.save_managed_runtime()
        self.settings_changed.emit(values)
        self.log.emit(f"Managed runtime saved at: {target}\n")

        self.step_changed.emit("Completed")
        self.progress.emit(100)
        diagnostics = self.manager.run_diagnostics()
        self.diagnostics_ready.emit(diagnostics)
        self.finished.emit(True, "Managed YOLO environment created successfully.")

    def _run_process(self, command: list[str]) -> int:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            self.log.emit(f"ERROR: {exc}\n")
            return -1
        with self._process_lock:
            self._process = process
        try:
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    self.log.emit(line)
                    if self._cancelled.is_set() and process.poll() is None:
                        self.cancel()
            return process.wait()
        finally:
            with self._process_lock:
                self._process = None

    def _cancelled_result(self) -> None:
        self.step_changed.emit("Cancelled")
        self.log.emit("Runtime setup cancelled by user. Partial files may remain in the managed runtime folder.\n")
        self.finished.emit(False, "Cancelled")
