"""Deterministic, evidence-first analysis of existing YOLO result artifacts.

This module only reads files already produced by YOLO.  It deliberately never
loads a model, starts training, validates a dataset, or infers scene-level
causes from aggregate metrics.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.results_reader import METRIC_ALIASES, extract_metrics, read_results, scan_run_folder
from persistence.training_analysis_store import TrainingAnalysisStore


ANALYSIS_SCHEMA_VERSION = 1
HEURISTIC_VERSION = 1
ANALYSIS_FILENAME = "training_analysis.json"
ANALYSIS_STORE = TrainingAnalysisStore()


@dataclass(frozen=True)
class AnalysisThresholds:
    """Centralized thresholds for conservative, reproducible heuristics."""

    precision_recall_gap: float = 0.10
    localization_gap: float = 0.18
    minimum_trend_epochs: int = 10
    trend_window_minimum: int = 5
    trend_window_maximum: int = 20
    plateau_delta: float = 0.01
    improving_delta: float = 0.01
    overfit_train_loss_drop: float = 0.03
    overfit_val_loss_rise: float = 0.03
    overfit_metric_drop: float = 0.01
    overfit_early_best_windows: int = 1
    excellent_map: float = 0.75
    good_map: float = 0.60
    fair_map: float = 0.40


@dataclass
class AnalysisFinding:
    finding_id: str
    severity: str
    confidence: str
    evidence: dict[str, float | int | str]
    recommendations: list[str] = field(default_factory=list)


@dataclass
class TrainingAnalysis:
    metrics: dict[str, float | None]
    rating: str
    best_epoch: int | None
    final_epoch: int | None
    findings: list[AnalysisFinding]
    source_files: dict[str, str]
    recommendations: list[str] = field(default_factory=list)
    cache_status: str = "fresh"
    persistence_error: str = ""


def persist_analysis(run_folder: str | Path, analysis: TrainingAnalysis) -> tuple[Path | None, str]:
    """Atomically persist derived data only; return an error instead of raising."""
    root = Path(run_folder)
    results = root / "results.csv"
    if not results.is_file():
        return None, "Analysis cache was not saved: results.csv not found."
    try:
        payload = _analysis_payload(results, analysis)
        return ANALYSIS_STORE.write_payload(root, payload), ""
    except (OSError, TypeError, ValueError) as exc:
        return None, f"Analysis cache was not saved: {exc}"


def load_cached_analysis(run_folder: str | Path) -> TrainingAnalysis | None:
    """Load a valid cache, or ``None`` when it must be regenerated.

    A cache is valid only when the supported schema/heuristic versions and the
    SHA-256 of the current ``results.csv`` match.  If the source was removed,
    an otherwise-valid cache remains usable as explicitly unverified reference
    data.
    """
    root = Path(run_folder)
    payload = ANALYSIS_STORE.read_payload(root)
    if payload is None:
        return None
    try:
        analysis = _analysis_from_payload(payload)
        source_status = ANALYSIS_STORE.cache_matches_source(root, payload)
        if source_status is None:
            analysis.cache_status = "unverified_cache"
            return analysis
        if not source_status:
            return None
        analysis.cache_status = "cache_hit"
        return analysis
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def analyze_or_load(run_folder: str | Path, *, force: bool = False) -> TrainingAnalysis:
    """Use a valid cache when possible; otherwise analyze and refresh it."""
    root = Path(run_folder)
    if not force:
        cached = load_cached_analysis(root)
        if cached is not None:
            return cached
    analysis = analyze_run(root)
    cache_path, error = persist_analysis(root, analysis)
    if error:
        analysis.persistence_error = error
    elif cache_path is not None:
        analysis.cache_status = "fresh"
    return analysis


def analyze_run(run_folder: str | Path, thresholds: AnalysisThresholds = AnalysisThresholds()) -> TrainingAnalysis:
    """Analyze a run folder safely and deterministically from ``results.csv``."""
    root = Path(run_folder)
    artifacts = scan_run_folder(root)
    source = {key: str(value) for key, value in artifacts.items() if value is not None}
    results = artifacts.get("results.csv")
    if results is None:
        return _insufficient_analysis(source, "results.csv missing")
    try:
        frame = read_results(results)
    except Exception as exc:  # pandas may raise several parser/value exceptions.
        return _insufficient_analysis(source, f"results.csv unavailable: {exc}")
    metrics = extract_metrics(frame)
    final_epoch = _row_epoch(frame, len(frame) - 1)
    metric_column = _metric_column(frame, "map50_95") or _metric_column(frame, "map50")
    best_epoch = _best_epoch(frame, metric_column)
    findings = _findings(metrics, frame, best_epoch, final_epoch, thresholds)
    return TrainingAnalysis(
        metrics=metrics,
        rating=_rating(metrics, thresholds),
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        findings=findings,
        source_files=source,
        recommendations=_dedupe_recommendations(findings),
    )


def _insufficient_analysis(source: dict[str, str], reason: str) -> TrainingAnalysis:
    return TrainingAnalysis(
        metrics={key: None for key in METRIC_ALIASES},
        rating="insufficient_data",
        best_epoch=None,
        final_epoch=None,
        findings=[AnalysisFinding("insufficient_metrics", "warning", "low", {"reason": reason})],
        source_files=source,
    )


def _analysis_payload(results: Path, analysis: TrainingAnalysis) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "heuristic_version": HEURISTIC_VERSION,
        "app_version": "0.11.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {"results_csv": str(results.resolve()), "sha256": ANALYSIS_STORE.source_hash(results)},
        "metrics": analysis.metrics,
        "rating": {"id": analysis.rating},
        "best_epoch": analysis.best_epoch,
        "final_epoch": analysis.final_epoch,
        "findings": [
            {
                "finding_id": item.finding_id,
                "severity": item.severity,
                "confidence": item.confidence,
                "evidence": item.evidence,
                "recommendations": item.recommendations,
            }
            for item in analysis.findings
        ],
        "recommendations": analysis.recommendations,
    }


def _analysis_from_payload(payload: object) -> TrainingAnalysis:
    if not isinstance(payload, dict):
        raise ValueError("analysis cache root must be an object")
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported analysis cache schema")
    if payload.get("heuristic_version") != HEURISTIC_VERSION:
        raise ValueError("analysis cache heuristics do not match")
    source = payload.get("source")
    metrics = payload.get("metrics")
    rating = payload.get("rating")
    findings = payload.get("findings")
    if not isinstance(source, dict) or not _is_sha256(source.get("sha256")):
        raise ValueError("analysis cache has no valid source hash")
    if not isinstance(metrics, dict) or not isinstance(rating, dict) or not isinstance(findings, list):
        raise ValueError("analysis cache has invalid canonical fields")
    if not isinstance(source.get("results_csv"), str) or any(key not in metrics for key in METRIC_ALIASES):
        raise ValueError("analysis cache lacks required canonical fields")
    parsed_metrics = {key: _safe_metric(metrics.get(key)) for key in METRIC_ALIASES}
    if not isinstance(rating.get("id"), str):
        raise ValueError("analysis cache has no rating id")
    parsed_findings: list[AnalysisFinding] = []
    for item in findings:
        if not isinstance(item, dict):
            raise ValueError("analysis cache has malformed finding")
        required = ("finding_id", "severity", "confidence", "evidence", "recommendations")
        if not all(key in item for key in required) or not isinstance(item["evidence"], dict):
            raise ValueError("analysis cache finding lacks canonical fields")
        if not all(isinstance(item[key], str) for key in ("finding_id", "severity", "confidence")):
            raise ValueError("analysis cache finding fields are invalid")
        if not isinstance(item["recommendations"], list) or not all(isinstance(value, str) for value in item["recommendations"]):
            raise ValueError("analysis cache recommendations are invalid")
        parsed_findings.append(AnalysisFinding(
            item["finding_id"], item["severity"], item["confidence"], item["evidence"], item["recommendations"]
        ))
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list) or not all(isinstance(value, str) for value in recommendations):
        recommendations = _dedupe_recommendations(parsed_findings)
    return TrainingAnalysis(
        metrics=parsed_metrics,
        rating=rating["id"],
        best_epoch=_safe_int(payload.get("best_epoch")),
        final_epoch=_safe_int(payload.get("final_epoch")),
        findings=parsed_findings,
        source_files={"results.csv": str(source.get("results_csv", ""))},
        recommendations=recommendations,
    )


def _rating(metrics: dict[str, float | None], t: AnalysisThresholds) -> str:
    precision, recall = metrics["precision"], metrics["recall"]
    score = metrics["map50_95"] if metrics["map50_95"] is not None else metrics["map50"]
    if any(value is None for value in (precision, recall, score)):
        return "insufficient_data"
    minimum = min(float(precision), float(recall))
    if float(score) >= t.excellent_map and minimum >= 0.85:
        return "excellent"
    if float(score) >= t.good_map and minimum >= 0.75:
        return "good"
    if float(score) >= t.fair_map and minimum >= 0.60:
        return "fair"
    return "needs_improvement"


def _findings(metrics, frame, best_epoch, final_epoch, t):
    findings: list[AnalysisFinding] = []
    precision, recall = metrics["precision"], metrics["recall"]
    if precision is not None and recall is not None:
        gap = float(precision) - float(recall)
        if gap >= t.precision_recall_gap:
            findings.append(AnalysisFinding(
                "precision_recall_gap_low_recall", "warning", "high",
                {"precision": precision, "recall": recall, "gap": gap, "threshold": t.precision_recall_gap},
                ["review_false_negatives"],
            ))
        elif -gap >= t.precision_recall_gap:
            findings.append(AnalysisFinding(
                "precision_recall_gap_low_precision", "warning", "high",
                {"precision": precision, "recall": recall, "gap": -gap, "threshold": t.precision_recall_gap},
                ["review_false_positives"],
            ))
    if metrics["map50"] is not None and metrics["map50_95"] is not None:
        gap = float(metrics["map50"]) - float(metrics["map50_95"])
        if gap >= t.localization_gap:
            findings.append(AnalysisFinding(
                "localization_gap", "info", "medium",
                {"map50": metrics["map50"], "map50_95": metrics["map50_95"], "gap": gap, "threshold": t.localization_gap},
                ["inspect_low_iou"],
            ))
    metric_column = _metric_column(frame, "map50_95") or _metric_column(frame, "map50")
    values = _clean_series(frame[metric_column]) if metric_column else []
    if len(frame) < t.minimum_trend_epochs or len(values) < t.minimum_trend_epochs:
        findings.append(AnalysisFinding(
            "insufficient_trend_epochs", "info", "high",
            {"epochs": len(values), "minimum": t.minimum_trend_epochs},
        ))
        return findings
    window = _trend_window(len(values), t)
    delta = values[-1] - values[-window]
    if abs(delta) <= t.plateau_delta:
        findings.append(AnalysisFinding(
            "training_plateau", "info", "medium",
            {"delta": delta, "window": window, "threshold": t.plateau_delta},
            ["consider_stopping_earlier"],
        ))
    elif delta > t.improving_delta:
        findings.append(AnalysisFinding(
            "still_improving", "info", "medium",
            {"delta": delta, "window": window, "threshold": t.improving_delta},
            ["consider_more_epochs"],
        ))
    overfit = _overfitting_evidence(frame, metric_column, best_epoch, final_epoch, t)
    signal_count = _overfit_signal_count(overfit)
    if signal_count >= 2:
        confidence = "high" if signal_count >= 3 else "medium"
        findings.append(AnalysisFinding(
            "possible_overfitting", "warning", confidence, overfit,
            ["inspect_best_epoch", "consider_early_stopping"],
        ))
    return findings


def _overfitting_evidence(frame, metric_column, best_epoch, final_epoch, t):
    train_column = _loss_column(frame, "train")
    val_column = _loss_column(frame, "val")
    evidence: dict[str, float | int | str] = {}
    window = _trend_window(len(frame), t)
    if train_column:
        values = _clean_series(frame[train_column])
        if len(values) >= window:
            delta = values[-1] - values[-window]
            if delta <= -t.overfit_train_loss_drop:
                evidence.update({"training_box_loss_start": values[-window], "training_box_loss_end": values[-1]})
    if val_column:
        values = _clean_series(frame[val_column])
        if len(values) >= window:
            delta = values[-1] - values[-window]
            if delta >= t.overfit_val_loss_rise:
                evidence.update({"validation_box_loss_start": values[-window], "validation_box_loss_end": values[-1]})
    if metric_column:
        values = _clean_series(frame[metric_column])
        if len(values) >= window and values[-1] - values[-window] <= -t.overfit_metric_drop:
            evidence.update({"map_start": values[-window], "map_end": values[-1]})
    if best_epoch is not None and final_epoch is not None and final_epoch - best_epoch >= window * t.overfit_early_best_windows:
        evidence.update({"best_epoch": best_epoch, "final_epoch": final_epoch})
    return evidence


def _overfit_signal_count(evidence: dict[str, float | int | str]) -> int:
    return sum(
        (
            "training_box_loss_start" in evidence,
            "validation_box_loss_start" in evidence,
            "map_start" in evidence,
            "best_epoch" in evidence,
        )
    )


def _metric_column(frame, metric):
    normalized = {"".join(char for char in str(column).lower() if char.isalnum()): column for column in frame.columns}
    for alias in METRIC_ALIASES[metric]:
        column = normalized.get("".join(char for char in alias.lower() if char.isalnum()))
        if column:
            return column
    return None


def _loss_column(frame, prefix: str):
    for column in frame.columns:
        normalized = "".join(char for char in str(column).lower() if char.isalnum())
        if normalized.startswith(f"{prefix}box") and normalized.endswith("loss"):
            return column
    return None


def _best_epoch(frame, column) -> int | None:
    if not column:
        return None
    candidates: list[tuple[int, float]] = []
    for index, value in enumerate(frame[column]):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            candidates.append((index, number))
    if not candidates:
        return None
    index = max(candidates, key=lambda item: item[1])[0]
    return _row_epoch(frame, index)


def _row_epoch(frame, index: int) -> int:
    value = frame.iloc[index].get("epoch", index)
    try:
        return int(value)
    except (TypeError, ValueError):
        return index


def _trend_window(length: int, t: AnalysisThresholds) -> int:
    return max(t.trend_window_minimum, min(t.trend_window_maximum, length // 10 or 1))


def _clean_series(series) -> list[float]:
    values: list[float] = []
    for value in series:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _dedupe_recommendations(findings: list[AnalysisFinding]) -> list[str]:
    return list(dict.fromkeys(value for finding in findings for value in finding.recommendations))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _safe_metric(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("metric is not numeric") from None
    if not math.isfinite(number):
        raise ValueError("metric is not finite")
    return number


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
