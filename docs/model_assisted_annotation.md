# Model-Assisted Annotation

v0.13 is a Public Beta for local YOLO **Detection `.pt`** assistance. The model proposes boxes; a person remains responsible for reviewing, editing, and saving them.

## Runtime and model

Model assistance uses the Python selected by the existing Runtime Manager: configured runtime first, managed per-user runtime second, then approved launcher/PATH fallbacks. The portable GUI does not bundle PyTorch or Ultralytics. If no runtime is available, manual annotation remains fully usable.

Choose only a trusted local `.pt` file and select **Load Model**. Model files can contain executable serialized content; the external worker is process isolation for reliability, not a security sandbox. URLs and automatic model downloads are not supported. The worker verifies that the model task is `detect` and returns its class names. Exact dataset/model class ID and name matches are enabled automatically. A name mismatch requires an explicit session-only override; a class-count/ID mismatch remains blocked and predictions with invalid dataset class IDs are rejected.

Device choices use stable IDs: `auto` selects CUDA GPU 0 when available and otherwise CPU; `0` requires GPU 0; and `cpu` forces CPU. The worker reports the actual device. CPU inference is supported but may be substantially slower.

## Predict the current image

Set confidence from 0.01 through 1.00 and select **Predict Current Image**. The persistent worker keeps the model in RAM/VRAM across images. Predictions are memory-only and dirty the document; they are written only by Save or autosave navigation. On an already annotated image, choose Add, Replace, or Cancel. Add and Replace each create one undo step. Zero detections do not clear boxes, create a label, or mark an image negative.

AI boxes display class, confidence, and an `AI` text indicator. Moving, resizing, deleting, or changing an AI box hides stale confidence and changes provenance to `model_assisted`.

## Provenance metadata

YOLO labels remain unchanged. Confidence, model identity, source, and outcomes use schema version 1 under `%LOCALAPPDATA%\YOLO-Trainer-UI\annotation_metadata\<dataset-id>.json`. Dataset identity hashes canonical YAML and root paths. Sources are `manual`, `model_generated`, `model_assisted`, and `unknown`. Existing labels without metadata remain unknown. Corrupt metadata is quarantined and never prevents training or label editing.

## Auto Annotate Current Split

Only images with no label file are eligible. Processing is sequential and reuses one loaded model. Existing labeled, empty-negative, and invalid label files are never overwritten. Detections are saved atomically; no detection leaves no `.txt`. Single-image errors continue, while fatal worker/protocol failures stop the job. Cancel stops sending new requests, preserves completed labels, and may wait for the current inference.

Audit reports contain relative paths and outcomes—not image data—under `%LOCALAPPDATA%\YOLO-Trainer-UI\annotation_reports\<dataset-id>\`.

## Privacy and limitations

Inference is local; images are not uploaded. v0.13 excludes ONNX, TensorRT, segmentation, pose, OBB, video, tracking, SAM, review queues, active learning, class remapping, parallel GPU inference, and automatic retraining.
