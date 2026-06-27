from __future__ import annotations

from core.trainer_process import TrainerProcess


class ValidatorProcess(TrainerProcess):
    """Independent non-blocking QProcess runner for YOLO validation."""
