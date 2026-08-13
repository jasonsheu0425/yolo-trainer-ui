# YOLO Trainer UI

## Languages

- 繁體中文 (default)
- English

Use **Language / 語言** on the Settings page to switch immediately. See
[localization documentation](docs/localization.md) for translation resources
and contributor guidance.

## Simple Mode

Simple Mode offers a guided Dataset Check → profile selection → training flow
while Advanced Mode retains all YOLO controls. See [Simple Mode](docs/simple_mode.md).

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
- **Runtime / Environment:** discovers Python and YOLO, diagnoses Ultralytics, PyTorch, CUDA, and GPU support, and can create a per-user managed runtime.

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

## Runtime setup

Open **Runtime / Environment** and choose **Run Diagnostics** to inspect the Python source, YOLO command, Ultralytics and PyTorch versions, CUDA status, and GPU name. Existing environments are resolved in this order: explicit Settings overrides, the managed runtime, Windows `py`, and commands on `PATH`.

If YOLO is missing, **Create Managed YOLO Environment** creates a virtual environment under the current user's Qt application-data folder and installs Ultralytics there. It does not need administrator rights and does not install CUDA Toolkit or force a particular PyTorch CUDA wheel. See [Runtime setup](docs/runtime_setup.md).

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

The v0.8.1 prerelease provides a PyInstaller **onedir** portable ZIP named `YOLO-Trainer-UI-v0.8.1-windows-portable.zip`. Download it from [GitHub Releases](https://github.com/jasonsheu0425/yolo-trainer-ui/releases), extract the entire archive, and run `YOLO-Trainer-UI.exe` beside its `_internal` folder.

The portable app contains its UI runtime, while YOLO is resolved separately from Settings, a per-user managed environment, or `PATH`. The first launch can be slower while Windows loads files or Windows Defender scans the extracted folder. GPU workflows require a compatible NVIDIA driver and PyTorch/CUDA build in the selected YOLO environment. See [Packaging](docs/packaging.md).

## Known limitations

- Dataset Builder does not replace human review.
- Prediction labels may contain incorrect boxes.
- Error Mining uses greedy matching rather than a complete COCO evaluation implementation.
- Large dataset builds can be cancelled, but files copied before cancellation may remain.
- The portable build is a prerelease and should be tested on additional clean Windows systems.

## Development status

Current version: **v0.8.1 Runtime Setup & Diagnostics** (`0.8.1`).

Development dependencies are listed in `requirements-dev.txt`. Before publishing a release, follow the [Release Checklist](docs/release_checklist.md).

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Packaging](docs/packaging.md)
- [Runtime setup](docs/runtime_setup.md)
- [Release checklist](docs/release_checklist.md)
- [Release notes template](docs/release_notes_template.md)

## License

This project is licensed under the [MIT License](LICENSE).
