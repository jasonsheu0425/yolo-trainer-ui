from __future__ import annotations

from services.analysis_service import AnalysisService


def test_analysis_service_preserves_analyzer_cache_behavior(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.8,0.8,0.9,0.7\n", encoding="utf-8"
    )
    service = AnalysisService()
    fresh = service.load_run(run)
    cached = service.load_run(run)
    assert fresh.rating == cached.rating
    assert cached.cache_status == "cache_hit"
