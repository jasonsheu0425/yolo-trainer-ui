"""Common state vocabulary for cancellable runners and workers."""
from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
