"""Composition root for the incrementally introduced application services."""
from __future__ import annotations

from dataclasses import dataclass

from app.navigation import NavigationController
from core.config_manager import ConfigManager
from core.runtime_manager import RuntimeManager
from services.analysis_service import AnalysisService
from services.annotation_service import AnnotationService
from services.annotation_inference_service import AnnotationInferenceService
from services.training_service import TrainingService


@dataclass
class ApplicationServices:
    """Own shared workflow services while pages remain presentation objects."""

    runtime: RuntimeManager
    training: TrainingService
    analysis: AnalysisService
    annotation: AnnotationService
    annotation_inference: AnnotationInferenceService
    navigation: NavigationController

    @classmethod
    def create(cls, config: ConfigManager) -> "ApplicationServices":
        runtime = RuntimeManager(config)
        annotation = AnnotationService()
        return cls(
            runtime=runtime,
            training=TrainingService(config, runtime),
            analysis=AnalysisService(),
            annotation=annotation,
            annotation_inference=AnnotationInferenceService(runtime, annotation),
            navigation=NavigationController({
                "quick_start", "train", "dataset", "builder", "validate", "predict",
                "mining", "export", "monitor", "analysis", "annotation", "runtime", "settings",
            }),
        )
