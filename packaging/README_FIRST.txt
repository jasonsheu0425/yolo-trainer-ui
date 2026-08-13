YOLO Trainer UI - Portable Windows Build
========================================

This is the portable Windows build of YOLO Trainer UI.

1. Extract the entire ZIP before running the application.
2. Start YOLO-Trainer-UI.exe from the extracted folder.
3. The first launch may take longer while Windows loads and scans dependencies.
4. If Windows Defender is scanning the files, please wait for it to finish.
5. Open Runtime / Environment and choose Run Diagnostics.
6. Use an existing Python/YOLO environment, or create a managed per-user runtime.
   The managed runtime is stored outside this portable folder and needs no administrator rights.
7. Managed setup does not force a CUDA wheel. GPU training requires a compatible NVIDIA
   driver and supported PyTorch/CUDA build; use the official PyTorch installation guide.
8. If CUDA is unavailable, YOLO can run on CPU but may be significantly slower.
9. runs/, datasets/, model weights, local settings, and managed runtimes are not included.
10. Model-assisted annotation accepts trusted local YOLO Detection .pt models only.
    It runs locally through the configured runtime; images are not uploaded.
    Model loading is not a security sandbox, so do not open untrusted model files.
11. Report problems at:
   https://github.com/jasonsheu0425/yolo-trainer-ui/issues

Keep the _internal folder beside YOLO-Trainer-UI.exe. Do not run the executable
directly from inside the ZIP archive.
