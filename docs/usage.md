# Usage

## Language

The default UI language is Traditional Chinese. Open **Language / 語言** in
Settings to change to English without restarting; the preference is saved
locally.

## Simple Mode

Use Quick Start for the guided workflow, or select Advanced Mode in Settings
for the complete pages and parameters. Dataset Check is required before Simple
Mode training can start.

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
# Training Result Analysis

After a training run completes, use **Analyze Results** on the Train page, or
scan `runs/detect` in Run Browser and select **Analyze Run**. The analysis page
uses existing `results.csv` only; it never retrains or validates. It caches
derived output as `training_analysis.json` and previews available curves and
confusion matrices. See [Training Analysis](training_analysis.md) for the
heuristic and cache behavior.
