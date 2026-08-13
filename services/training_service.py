"""Training orchestration separated from TrainPage widget ownership."""
from __future__ import annotations

import shlex
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.config_manager import ConfigManager
from core.runtime_manager import RuntimeManager
from core.trainer_process import TrainerProcess
from domain.training import TrainingConfig
from services.errors import InvalidTrainingConfigError, RuntimeUnavailableError


class TrainingService(QObject):
    """Coordinates typed config, runtime resolution, and one train runner."""

    started = Signal()
    log = Signal(str)
    completed = Signal(int, int)
    failed = Signal(str)
    state_changed = Signal(bool)

    def __init__(self, config_store: ConfigManager, runtime: RuntimeManager | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config_store = config_store
        self.runtime = runtime or RuntimeManager(config_store)
        self.runner = TrainerProcess(self)
        self.runner.started.connect(self.started)
        self.runner.output.connect(self.log)
        self.runner.finished.connect(self.completed)
        self.runner.error.connect(self.failed)
        self.runner.state_changed.connect(self.state_changed)

    @property
    def running(self) -> bool:
        return self.runner.running

    @staticmethod
    def build_command(config: TrainingConfig) -> list[str]:
        """The single canonical TrainingConfig → YOLO argument conversion."""
        args = [
            config.task, "train", f"model={config.model}", f"data={config.data}",
            f"imgsz={config.imgsz}", f"epochs={config.epochs}", f"batch={config.batch}",
            f"device={config.device}", f"workers={config.workers}", f"project={config.project}",
            f"name={config.name}", f"resume={config.resume}", f"pretrained={config.pretrained}",
            f"cache={config.cache}", f"patience={config.patience}",
        ]
        if config.advanced.strip():
            args.extend(shlex.split(config.advanced.strip(), posix=False))
        return args

    def preview(self, config: TrainingConfig) -> str:
        return self.runner.preview(self.runtime.yolo_command_for_preview(), self.build_command(config))

    def start(self, config: TrainingConfig, working_directory: Path | None = None) -> None:
        """Resolve runtime then start non-blocking execution; no UI dialogs here."""
        if not config.data or not Path(config.data).is_file():
            raise InvalidTrainingConfigError("training_data_missing")
        if not config.model.strip():
            raise InvalidTrainingConfigError("training_model_missing")
        program = self.runtime.resolve_yolo_command()
        if not program:
            raise RuntimeUnavailableError("runtime_unavailable")
        self.runner.start(program, self.build_command(config), working_directory)

    def stop(self) -> None:
        self.runner.stop()

    @staticmethod
    def latest_run(config: TrainingConfig, base_directory: Path | None = None) -> Path | None:
        """Find the newest output folder using the established naming behavior."""
        project = Path(config.project).expanduser()
        if not project.is_absolute():
            project = (base_directory or Path.cwd()) / project
        if not project.is_dir():
            return None
        candidates = [path for path in project.glob(f"{config.name}*") if path.is_dir()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
