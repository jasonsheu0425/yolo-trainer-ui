"""QProcess JSONL transport for the persistent annotation inference worker."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Any

from PySide6.QtCore import QObject, QProcess, Signal

from core.annotation_worker_path import resolve_annotation_worker_path
from runtime_workers.annotation_inference_protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    validate_response,
)


class InferenceWorkerController(QObject):
    message_received = Signal(object)
    log_received = Signal(str)
    protocol_failed = Signal(str, str)
    exited = Signal(int, int)
    started = Signal()

    def __init__(self, worker_path: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker_path = worker_path
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.started.connect(self.started)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._stdout_buffer = b""

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @property
    def process_id(self) -> int:
        return int(self.process.processId())

    def start(self, program: str, prefix_args: list[str] | None = None) -> None:
        if self.running:
            raise RuntimeError("worker_already_running")
        worker = (self.worker_path or resolve_annotation_worker_path()).resolve()
        if not worker.is_file():
            raise FileNotFoundError(str(worker))
        self._stdout_buffer = b""
        self.process.setProgram(program)
        self.process.setArguments([*(prefix_args or []), "-u", str(worker)])
        self.process.start()

    def send(self, payload: dict[str, Any]) -> None:
        if not self.running:
            raise RuntimeError("worker_not_running")
        self.process.write(encode_message(payload))

    def request_shutdown(self) -> None:
        if self.running:
            self.process.closeWriteChannel()

    def terminate(self) -> None:
        if self.running:
            self.process.terminate()

    def kill(self) -> None:
        if self.running:
            pid = self.process_id
            if os.name == "nt" and pid:
                completed = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
                if completed.returncode == 0:
                    return
            self.process.kill()

    def wait_for_finished(self, milliseconds: int) -> bool:
        return self.process.waitForFinished(milliseconds)

    def _read_stdout(self) -> None:
        self._stdout_buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._stdout_buffer:
            raw, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
            if not raw.strip():
                continue
            try:
                payload = validate_response(decode_message(raw))
            except ProtocolError as exc:
                self.protocol_failed.emit(exc.code, str(exc))
                continue
            self.message_received.emit(payload)

    def _read_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
        if text:
            self.log_received.emit(text.rstrip())

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.exited.emit(exit_code, int(exit_status.value))

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self.log_received.emit(f"QProcess: {error.name}: {self.process.errorString()}")
