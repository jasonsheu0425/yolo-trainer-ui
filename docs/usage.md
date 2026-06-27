# Usage

## Pages

- **Train:** configure training parameters, apply presets, start or stop training, and inspect the last run.
- **Dataset Check:** validate `data.yaml`, image/label pairing, and YOLO label rows.
- **Dataset Builder:** combine a base dataset with selected hard cases in a background worker.
- **Validate / Evaluate:** run model validation and preserve validation metrics.
- **Predict / Test:** run image, folder, or video inference and optionally save TXT labels and confidence values.
- **Error Mining:** compare predictions with ground truth and export hard cases.
- **Report Viewer:** filter hard-case reports and inspect related images and labels.
- **Export:** export PyTorch weights to ONNX and other supported Ultralytics formats.
- **Monitor / Results:** inspect GPU state, training charts, and run folders.
- **Settings:** configure local Python, YOLO command, runs folder, model, and device defaults.

## Recommended workflow

```text
Dataset Check
→ Train
→ Validate
→ Predict with save_txt/save_conf
→ Error Mining with ground-truth comparison
→ Report Viewer
→ Dataset Builder
→ Train with the new data.yaml
```

Use **Preview Build** before building a dataset. A cancelled build may leave already copied files, so use a new empty output folder for large jobs.
