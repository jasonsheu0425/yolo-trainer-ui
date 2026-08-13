# YOLO Trainer UI v0.13.0 Model-Assisted Annotation (Beta)

## Status

**Public Beta / GitHub Prerelease.** This build is not Latest Stable.

## Highlights

- Trusted local YOLO Detection `.pt` model assistance
- Persistent external runtime worker; model remains loaded across images
- Predict Current Image with editable confidence-labelled proposals
- One-step Add/Replace undo
- Sequential split auto-annotation with progress and cancellation
- CPU/CUDA selection, class compatibility validation, and provenance
- Traditional Chinese and English shared Simple/Advanced Annotation page

## Data safety and privacy

- Batch annotation never overwrites existing labels or empty negatives.
- No detection does not create an empty label.
- Current predictions remain unsaved until Save/autosave navigation.
- YOLO format is unchanged; provenance metadata is separate.
- Label writes remain atomic.
- Inference is local and images are not uploaded.

## Scope

Detection `.pt` only. ONNX, TensorRT, segmentation, pose, OBB, video, tracking, review queues, and active learning are not included. Model loading is not a security sandbox.

## Validation

Validated with JSONL/worker/QProcess lifecycle tests, persistent-model reuse,
crash/timeout recovery, metadata and batch protection, Qt UI, Dataset Check,
architecture boundaries, a real `best.pt` CPU worker smoke on a temporary image,
PyInstaller, frozen bundled-worker launch, and clean extraction.

## Known issues

- Release validation runtime was CPU-only; the CUDA request/error path is automated, but physical GPU inference was unavailable.
- Cancellation is cooperative and the current inference may finish first.
- Class-name override is session-only; automatic class remapping is absent.
- Only trusted `.pt` files should be loaded because deserialization is not sandboxed.
