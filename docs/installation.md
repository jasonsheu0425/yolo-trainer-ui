# Installation

## Source code

YOLO Trainer UI targets Windows 10/11 and Python 3.10–3.12. From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The application calls the Ultralytics `yolo` command. Verify it with:

```powershell
yolo checks
```

## CPU-only environments

For a machine that does not need CUDA, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to install CPU builds of `torch` and `torchvision`, then run:

```powershell
pip install -r requirements-cpu.txt
```

## GPU and CUDA

The UI, Dataset Check, Report Viewer, and Dataset Builder can run without CUDA. GPU training, prediction, and validation require a compatible NVIDIA driver and a CUDA-enabled PyTorch build. Choose the PyTorch command for the actual driver and Python environment instead of assuming one fixed CUDA version.

Check availability with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
