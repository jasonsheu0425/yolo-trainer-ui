# Runtime Setup & Diagnostics

YOLO Trainer UI keeps the portable application and the YOLO execution environment separate. Dataset Check, Dataset Builder, Report Viewer, and other UI-only tools remain usable when YOLO is missing.

## Existing environment

Open **Settings** to browse to an existing `python.exe` and `yolo.exe`, save the values, then open **Runtime / Environment** and select **Run Diagnostics**. The page reports where Python and YOLO were found, package versions, CUDA availability, and the detected GPU.

Auto detection uses this order:

1. Explicit executable configured in Settings.
2. Per-user managed runtime.
3. Windows `py` launcher for Python.
4. `python.exe` and `python3.exe` on `PATH`.
5. `yolo.exe` on `PATH` for YOLO.

In a frozen build, `YOLO-Trainer-UI.exe` is the application itself and is never accepted as a Python interpreter.

## Managed environment

Open **Runtime / Environment** and choose **Create Managed YOLO Environment**. The background worker:

1. Finds a usable installed Python.
2. Creates a virtual environment below `QStandardPaths.AppLocalDataLocation/runtime/.venv`.
3. Upgrades pip and installs `ultralytics`.
4. Verifies Ultralytics, PyTorch, CUDA, and `yolo.exe`.
5. Saves the managed Python and YOLO executables to local application settings.

The operation runs per user and does not require Administrator privileges. Progress and process output appear on the page. **Cancel** terminates the active step cooperatively; partially installed files may remain and can be reused or manually removed later.

## GPU setup

Managed setup intentionally does not force a particular CUDA wheel. GPU compatibility depends on the physical GPU, NVIDIA driver, Python version, and PyTorch build. If diagnostics says `CUDA Available: No`, open the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), choose the correct command, and run it using the Python executable shown for the desired environment.

Installing NVIDIA CUDA Toolkit is not automatically required for PyTorch wheels, and the application does not install or change NVIDIA drivers.

## CPU mode

When YOLO and PyTorch are available but CUDA is not, the page reports **Partial / Ready for CPU only**. Training and inference can still work on CPU, but they may be significantly slower. Set the YOLO device option appropriately for the intended operation.

## Portable build

The portable ZIP contains the application UI but not a managed YOLO environment. Managed runtime files are stored in the user's local application-data folder, outside the extracted ZIP, and are excluded from Git and release archives. This lets a replacement portable build reuse the same configured runtime.
