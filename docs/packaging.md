# Packaging

## Portable ZIP

The v0.8 prerelease uses a PyInstaller **onedir** build. Download `YOLO-Trainer-UI-v0.8.0-windows-portable.zip` from GitHub Releases, extract the complete ZIP, and run `YOLO-Trainer-UI.exe`. Keep `_internal/` beside the executable.

The package is large because it includes Python, Qt, plotting, and required ML/data-processing dependencies. First launch may take longer while Windows loads the dependency tree or Windows Defender scans the extracted files.

The portable package does not include runs, datasets, local settings, model weights, or a separate YOLO runtime. Train, Predict, Validate, and Export use the `yolo` executable configured in Settings. GPU operations therefore depend on that external environment having compatible NVIDIA driver and PyTorch/CUDA support. A CUDA-unavailable environment may fall back to slow CPU execution.

## Building locally

Install development dependencies, clean previous output, build the onedir application, and create the ZIP:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
scripts\clean_build.bat
scripts\build_pyinstaller_onedir.bat
scripts\make_portable_zip.bat
```

The build script uses `--onedir --windowed`, adds application documentation and example configuration, and creates `dist\YOLO-Trainer-UI`. It intentionally does not use `--onefile`, which is harder to inspect and troubleshoot with Qt and ML dependencies.

The ZIP script creates:

```text
release_artifacts\YOLO-Trainer-UI-v0.8.0-windows-portable.zip
```

`build/`, `dist/`, `release_artifacts/`, generated spec files, and ZIP archives are local artifacts and must not be committed. An installer is not currently provided.
