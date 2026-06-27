# YOLO Trainer UI

YOLO Trainer UI is a Windows desktop application built with Python and PySide6 for Ultralytics YOLO dataset checking, training, validation, prediction, error mining, hard-case review, and dataset version building.

Long-running YOLO commands use `QProcess`, while Dataset Builder uses a `QThread` worker so the interface remains responsive.

## Features

- **Train:** command preview, training presets, live logs, stop control, and last-run summary.
- **Dataset Check:** validates YAML paths, image/label pairing, class IDs, and YOLO label rows.
- **GPU Monitor:** displays PyTorch, CUDA, GPU utilization, memory, temperature, and power information.
- **Export:** exports `.pt` models to ONNX and other Ultralytics-supported formats.
- **Predict / Test:** runs inference on images, folders, or videos and can save TXT labels and confidence values.
- **Validate / Evaluate:** runs validation, displays plots and metrics, and persists standalone validation metrics.
- **Run Browser:** classifies and inspects train, predict, and validation run folders.
- **Error Mining:** compares predictions with ground truth using IoU-based hard-case classification.
- **Report Viewer:** filters hard-case reports, previews images, and opens prediction or ground-truth labels.
- **Dataset Builder:** builds a new dataset version from a base dataset and selected hard cases without changing the source dataset.

## Screenshots

Screenshots will be added later. The placeholder directory is [`docs/screenshots/`](docs/screenshots/).

## Install from source

Requirements: Windows 10/11 and Python 3.10–3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

See [Installation](docs/installation.md) for CPU and GPU environment guidance.

## GPU and CUDA

- Running the UI, Dataset Check, Report Viewer, or Dataset Builder does not require CUDA.
- GPU training, prediction, and validation require a compatible NVIDIA driver and a CUDA-enabled PyTorch build.
- If `torch.cuda.is_available()` returns `False`, YOLO may run on the CPU and training can be very slow.
- Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the local Python, driver, and compute environment. This project does not prescribe one CUDA wheel for every machine.

## YOLO dataset layout

```text
dataset/
├─ images/
│  ├─ train/
│  └─ val/
├─ labels/
│  ├─ train/
│  └─ val/
└─ data.yaml
```

A minimal `data.yaml` can use paths relative to its dataset root:

```yaml
path: .
train: images/train
val: images/val
names:
  0: object
```

Each non-empty detection label row uses:

```text
class_id x_center y_center width height
```

## Recommended workflow

```text
Dataset Check
→ Train
→ Validate
→ Predict with save_txt/save_conf
→ Error Mining with ground-truth comparison
→ Report Viewer
→ Dataset Builder
→ Train with the new data.yaml
```

More detail is available in [Usage](docs/usage.md) and [Troubleshooting](docs/troubleshooting.md).

## GitHub releases and portable ZIP

A future release is planned to provide a portable ZIP. v0.7 prepares the documentation and packaging scripts, but it does not claim that a tested portable binary is currently available. See [Packaging](docs/packaging.md).

## Known limitations

- Dataset Builder does not replace human review.
- Prediction labels may contain incorrect boxes.
- Error Mining uses greedy matching rather than a complete COCO evaluation implementation.
- Large dataset builds can be cancelled, but files copied before cancellation may remain.
- The packaged Windows distribution is not complete yet.

## Development status

Current version: **v0.7 GitHub Release Preparation** (`0.7.0`).

Development dependencies are listed in `requirements-dev.txt`. Before publishing a release, follow the [Release Checklist](docs/release_checklist.md).

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Packaging](docs/packaging.md)
- [Release checklist](docs/release_checklist.md)
- [Release notes template](docs/release_notes_template.md)

## License

This project is licensed under the [MIT License](LICENSE).
