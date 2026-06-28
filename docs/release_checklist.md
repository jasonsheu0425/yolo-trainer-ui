# Release Checklist

- [ ] `git status` is clean.
- [ ] Regression tests pass.
- [ ] README and documentation match the release.
- [ ] Version constants and release tag are correct.
- [ ] Portable ZIP was built and tested, when included.
- [ ] Extracted portable ZIP launches and closes normally.
- [ ] Portable asset follows `YOLO-Trainer-UI-vX.X.X-windows-portable.zip` naming.
- [ ] Release notes and checksums are prepared.
- [ ] Tracked files exclude `runs/`, `datasets/`, `weights/`, model files, build output, and local settings.
- [ ] `build/`, `dist/`, and `release_artifacts/` remain untracked.
- [ ] Release assets open and run on a clean Windows test environment.
