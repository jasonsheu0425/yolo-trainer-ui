# YOLO Trainer UI v0.12.0 Annotation Editor MVP (Alpha)

## Status

**Public Alpha / GitHub Prerelease.** This build should not be marked Latest Stable.

## Highlights

- Built-in YOLO detection annotation editor in Simple and Advanced Mode
- Bounding-box creation, selection, move, four-corner resize, delete, and class changes
- Copy/paste and current-image undo/redo
- Autosave, negative-image support, zoom, pan, Fit Image, and 100%
- Dataset YAML and train/validation/test split integration
- Traditional Chinese and English interface

## Data safety

- Label saves use a flushed temporary file and atomic replacement.
- Untouched unlabeled images are never converted into empty label files.
- Saving after deleting the final box creates an intentional empty negative label.
- Malformed labels are not silently overwritten; explicit repair creates a per-user backup first.

## Scope

v0.12 supports detection bounding boxes only. It does not include segmentation, pose, OBB, tracking, video annotation, or AI-assisted annotation.

## Validation

The release gate includes annotation geometry and round-trip tests, label/store data-loss tests, service and undo tests, Qt interaction tests, Dataset Check compatibility, architecture boundaries, translation validation, clean PyInstaller onedir packaging, frozen launch, and clean-extraction smoke tests.

## Known issues

- Undo/redo is scoped to the current image.
- There is no crash-recovery journal for unsaved memory-only changes.
- Image lists are intentionally thumbnail-free for large-dataset responsiveness.
- Public Alpha coverage is programmatic/frozen; additional physical Windows and DPI feedback is welcome.
