from __future__ import annotations

from core.trainer_process import TrainerProcess


class PredictorProcess(TrainerProcess):
    """Independent non-blocking QProcess runner for YOLO predictions."""

