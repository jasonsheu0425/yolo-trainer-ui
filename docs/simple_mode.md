# Simple Mode

Simple Mode is a guided entry point over the same backend as Advanced Mode.
Choose a `data.yaml`, run Dataset Check, select a model profile and a training
profile, then start training. Dataset Check must pass before Simple Mode enables
training. Changing the dataset invalidates the previous check.

Profiles map to existing Train settings: Fast → `yolov8n.pt`, Balanced →
`yolov8s.pt`, High Accuracy → `yolov8m.pt`; Quick Test → 1 epoch / batch 4,
Standard → 100 epochs / batch 16, Extended → 150 epochs / batch 8.

Switch to Advanced Mode at any time to inspect or adjust the actual values.
Custom Advanced settings are preserved until a Simple Mode profile is explicitly
selected. Interface mode is saved locally in `app_settings.json`.
# Results analysis

After a successful Simple Mode run, **View Results** opens Training Analysis
with that completed run already selected. The page uses the same deterministic
analysis as Advanced Mode; it does not start another YOLO operation.
