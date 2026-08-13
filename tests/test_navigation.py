from __future__ import annotations

from app.navigation import NavigationController


def test_navigation_controller_only_emits_registered_stable_ids(qtbot=None):
    controller = NavigationController({"train", "analysis"})
    requests = []
    controller.requested.connect(requests.append)
    assert controller.go_to("analysis", {"run_path": "example"})
    assert not controller.go_to("not-a-page")
    assert len(requests) == 1
    assert requests[0].target == "analysis"
