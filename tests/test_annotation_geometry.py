from __future__ import annotations

import pytest

from domain.annotation import BoundingBox, PixelBox, xyxy_to_yolo, yolo_to_xyxy


@pytest.mark.parametrize("size", [(640, 480), (1920, 1080), (2560, 1440), (3840, 2160)])
def test_yolo_pixel_round_trip_at_common_and_4k_sizes(size):
    original = BoundingBox(2, 0.5, 0.45, 0.25, 0.30)
    result = xyxy_to_yolo(yolo_to_xyxy(original, *size), *size, original.class_id)
    assert result == pytest.approx(original)


def test_edges_full_image_small_box_and_clamping():
    full = xyxy_to_yolo(PixelBox(0, 0, 640, 480), 640, 480, 0)
    assert full == BoundingBox(0, 0.5, 0.5, 1.0, 1.0)
    small = xyxy_to_yolo(PixelBox(10, 20, 13, 23), 640, 480, 0)
    assert small.is_valid(1)
    clamped = xyxy_to_yolo(PixelBox(-50, -10, 700, 500), 640, 480, 0)
    assert clamped == full


def test_invalid_image_and_zero_area_are_rejected():
    with pytest.raises(ValueError):
        xyxy_to_yolo(PixelBox(1, 1, 1, 1), 640, 480, 0)
    with pytest.raises(ValueError):
        yolo_to_xyxy(BoundingBox(0, 0.5, 0.5, 0.2, 0.2), 0, 480)
