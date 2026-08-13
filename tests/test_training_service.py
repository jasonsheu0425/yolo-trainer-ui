from __future__ import annotations

from domain.training import TrainingConfig
from services.training_service import TrainingService


def test_training_config_to_command_is_single_stable_path():
    config = TrainingConfig(
        task="detect", model="model with spaces.pt", data="data folder/data.yaml",
        imgsz=832, epochs=73, batch=11, device="cpu", workers=3,
        project="runs/detect", name="custom run", resume=True, pretrained=False,
        cache=True, patience=12, advanced="optimizer=auto cos_lr=True",
    )
    assert TrainingService.build_command(config) == [
        "detect", "train", "model=model with spaces.pt", "data=data folder/data.yaml",
        "imgsz=832", "epochs=73", "batch=11", "device=cpu", "workers=3",
        "project=runs/detect", "name=custom run", "resume=True", "pretrained=False",
        "cache=True", "patience=12", "optimizer=auto", "cos_lr=True",
    ]


def test_simple_profile_values_round_trip_through_training_config():
    config = TrainingConfig().with_updates(model="yolov8s.pt", epochs=100, imgsz=640, batch=16, patience=50)
    assert (config.model, config.epochs, config.imgsz, config.batch, config.patience) == (
        "yolov8s.pt", 100, 640, 16, 50
    )
