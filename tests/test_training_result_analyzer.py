from __future__ import annotations

import json

from core.training_result_analyzer import (
    ANALYSIS_FILENAME,
    analyze_or_load,
    analyze_run,
    load_cached_analysis,
    persist_analysis,
)


def write_csv(folder, rows):
    folder.mkdir()
    (folder / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        + "\n".join(rows),
        encoding="utf-8",
    )


def test_gap_localization_and_determinism(tmp_path):
    run = tmp_path / "run"
    write_csv(run, ["0,0.92,0.70,0.90,0.65"])
    first, second = analyze_run(run), analyze_run(run)
    assert first.rating == second.rating
    assert {item.finding_id for item in first.findings} == {
        "precision_recall_gap_low_recall",
        "localization_gap",
        "insufficient_trend_epochs",
    }


def test_one_epoch_and_missing_csv_are_safe(tmp_path):
    run = tmp_path / "one"
    write_csv(run, ["0,0.9,0.9,0.9,0.8"])
    assert any(
        item.finding_id == "insufficient_trend_epochs"
        for item in analyze_run(run).findings
    )
    assert analyze_run(tmp_path / "missing").rating == "insufficient_data"


def test_balanced_model_rating_and_no_scene_hallucinations(tmp_path):
    run = tmp_path / "healthy"
    write_csv(run, ["0,0.91,0.90,0.92,0.82"])
    analysis = analyze_run(run)
    assert analysis.rating == "excellent"
    forbidden = {"small_objects", "distant_targets", "bad_lighting", "class_imbalance", "bad_labels"}
    assert not forbidden.intersection(item.finding_id for item in analysis.findings)


def test_low_precision_and_missing_metrics_are_safe(tmp_path):
    run = tmp_path / "precision"
    write_csv(run, ["0,0.60,0.84,0.88,0.70"])
    assert "precision_recall_gap_low_precision" in {item.finding_id for item in analyze_run(run).findings}
    missing = tmp_path / "missing_metric"
    missing.mkdir()
    (missing / "results.csv").write_text("epoch,unknown\n0,1\n", encoding="utf-8")
    assert analyze_run(missing).rating == "insufficient_data"


def test_trend_plateau_and_improving_are_deterministic(tmp_path):
    plateau = tmp_path / "plateau"
    rows = [f"{epoch},0.80,0.80,0.80,{0.700 + epoch * 0.0005:.4f}" for epoch in range(10)]
    write_csv(plateau, rows)
    first, second = analyze_run(plateau), analyze_run(plateau)
    assert "training_plateau" in {item.finding_id for item in first.findings}
    assert first.metrics == second.metrics
    assert [(item.finding_id, item.evidence, item.recommendations) for item in first.findings] == [
        (item.finding_id, item.evidence, item.recommendations) for item in second.findings
    ]
    improving = tmp_path / "improving"
    rows = [f"{epoch},0.80,0.80,0.80,{0.50 + epoch * 0.02:.3f}" for epoch in range(10)]
    write_csv(improving, rows)
    assert "still_improving" in {item.finding_id for item in analyze_run(improving).findings}


def test_possible_overfitting_requires_multiple_evidence_signals(tmp_path):
    run = tmp_path / "overfit"
    run.mkdir()
    rows = [
        f"{epoch},0.80,0.80,0.85,{0.80 - epoch * 0.02:.3f},{0.50 - epoch * 0.02:.3f},{0.40 + epoch * 0.02:.3f}"
        for epoch in range(10)
    ]
    (run / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,val/box_loss\n"
        + "\n".join(rows), encoding="utf-8"
    )
    finding_ids = {item.finding_id for item in analyze_run(run).findings}
    assert "possible_overfitting" in finding_ids
    healthy = tmp_path / "losses_healthy"
    healthy.mkdir()
    rows = [
        f"{epoch},0.80,0.80,0.85,{0.50 + epoch * 0.02:.3f},{0.50 - epoch * 0.02:.3f},{0.40 - epoch * 0.02:.3f}"
        for epoch in range(10)
    ]
    (healthy / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,val/box_loss\n"
        + "\n".join(rows), encoding="utf-8"
    )
    assert "possible_overfitting" not in {item.finding_id for item in analyze_run(healthy).findings}


def test_cache_hit_stale_malformed_and_missing_source(tmp_path):
    run = tmp_path / "cache"
    write_csv(run, ["0,0.8,0.8,0.9,0.7"])
    analysis = analyze_run(run)
    path, error = persist_analysis(run, analysis)
    assert path and not error
    assert load_cached_analysis(run).cache_status == "cache_hit"  # type: ignore[union-attr]
    (run / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n0,0.7,0.7,0.8,0.6\n",
        encoding="utf-8",
    )
    assert load_cached_analysis(run) is None
    assert analyze_or_load(run).cache_status == "fresh"
    (run / ANALYSIS_FILENAME).write_text("{bad", encoding="utf-8")
    assert load_cached_analysis(run) is None
    persist_analysis(run, analyze_run(run))
    (run / "results.csv").unlink()
    cached = load_cached_analysis(run)
    assert cached and cached.cache_status == "unverified_cache"
    payload = json.loads((run / ANALYSIS_FILENAME).read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    (run / ANALYSIS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_analysis(run) is None
