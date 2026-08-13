# Portable Packaging

Run `scripts\build_pyinstaller_onedir.bat`, then
`scripts\make_portable_zip.bat`. The ZIP is self-contained and should be
tested only after extraction to a folder outside the repository. It contains
the executable, `_internal`, i18n resources, documentation, and the settings
example; it intentionally excludes local settings, runtimes, models, and runs.
# Training Analysis packaging check

The frozen build must include `core/training_result_analyzer.py`,
`ui/training_analysis_page.py`, and both `i18n` JSON files. Verify that the
portable app builds the Training Analysis page and can open a run independently
of the source repository.

For v0.12, also verify that `ui.annotation`, `services.annotation_service`,
`domain.annotation`, `persistence.yolo_annotation_store`, and Qt image format
plugins are present. The frozen smoke must construct the Annotation page and
load a temporary YOLO dataset without importing anything from the source tree.

For v0.13, `annotation_inference_worker.py` and its protocol module must be
physical files under `_internal/runtime_workers`. Resolve and launch them with
the external configured runtime in both the build folder and a clean extracted
ZIP. The package must not include PyTorch, Ultralytics, a managed runtime,
models, or user annotation metadata/reports.
