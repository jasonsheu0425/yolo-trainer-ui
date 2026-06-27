from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class TrainerProcess(QObject):
    output = Signal(str)
    started = Signal()
    finished = Signal(int, int)
    error = Signal(str)
    state_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @staticmethod
    def preview(program: str, args: list[str]) -> str:
        return subprocess.list2cmdline([program, *args])

    def start(self, program: str, args: list[str], working_directory: Path | None = None) -> None:
        if self.running:
            self.error.emit("已有一個程序正在執行。")
            return
        if working_directory:
            self.process.setWorkingDirectory(str(working_directory))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)
        self.output.emit(f"> {self.preview(program, args)}\n")
        self.process.start(program, args)

    def stop(self) -> None:
        if not self.running:
            return
        self.output.emit("\n正在停止程序…\n")
        self.process.terminate()
        QTimer.singleShot(3000, self._force_kill_if_needed)

    def _force_kill_if_needed(self) -> None:
        if self.running:
            self.output.emit("程序未正常結束，正在強制終止…\n")
            self.process.kill()

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput())
        self.output.emit(data.decode("utf-8", errors="replace"))

    def _on_started(self) -> None:
        self.state_changed.emit(True)
        self.started.emit()

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read_output()
        self.state_changed.emit(False)
        self.finished.emit(code, int(status.value))

    def _on_error(self, error: QProcess.ProcessError) -> None:
        self.error.emit(f"無法啟動或執行程序：{self.process.errorString()} ({error.name})")

