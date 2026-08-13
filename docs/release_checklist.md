# Release Checklist

The v0.10.1 release validates source checks, Qt behavioral tests, translation
resources, a clean PyInstaller onedir build, portable ZIP content, clean ZIP
extraction, and matching SHA-256 hashes for the uploaded release asset.

The Windows portable build is a prerelease. It includes the UI and diagnostics,
but not a managed runtime, trained weights, user settings, or dataset outputs.
# Training Result Analysis release check

- Run analyzer, UI, cache, translation, and source regression tests.
- Build the PyInstaller onedir application and perform a frozen-process smoke.
- Create a portable ZIP, extract it outside the repository, and verify its
  SHA-256 before publishing.
