# Portable Packaging

Run `scripts\build_pyinstaller_onedir.bat`, then
`scripts\make_portable_zip.bat`. The ZIP is self-contained and should be
tested only after extraction to a folder outside the repository. It contains
the executable, `_internal`, i18n resources, documentation, and the settings
example; it intentionally excludes local settings, runtimes, models, and runs.
