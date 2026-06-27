# Packaging

Packaging is planned but is not a completed release asset in v0.7.

The preferred first target is PyInstaller **onedir** because it is easier to inspect and troubleshoot than **onefile**, particularly with Qt, Ultralytics, and model runtime dependencies. Use `scripts\build_pyinstaller_onedir.bat` as a preparation script.

A portable ZIP is simply the generated onedir folder compressed for manual extraction. An installer would additionally manage shortcuts, installation location, upgrades, and removal; no installer is currently provided.

PyTorch and CUDA dependencies can make a packaged application very large. A production package should decide whether CPU and GPU runtimes are separate downloads and must be tested on a clean Windows machine before release.
