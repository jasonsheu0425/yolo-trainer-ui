"""Stable service-level errors mapped to localized UI messages by pages."""
from __future__ import annotations


class ApplicationServiceError(RuntimeError):
    """Base error for expected application workflow failures."""


class RuntimeUnavailableError(ApplicationServiceError):
    """Raised when a requested YOLO command cannot be resolved."""


class InvalidTrainingConfigError(ApplicationServiceError):
    """Raised when a typed training configuration cannot start safely."""
