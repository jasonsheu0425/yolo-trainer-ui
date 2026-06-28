# Troubleshooting

## CUDA is not available

Run `python -c "import torch; print(torch.cuda.is_available())"`. If it returns `False`, verify the NVIDIA driver and install a CUDA-enabled PyTorch build selected for the local environment. CPU execution remains possible but may be much slower.

## `yolo` command not found

Install `requirements.txt`, run `yolo checks`, and set **Settings → YOLO command** to `yolo` or the appropriate executable inside the virtual environment.

## Torch installation errors

Do not assume one CUDA wheel works everywhere. Use the official PyTorch installation selector for the installed Python version, operating system, and desired compute platform.

## PySide6 startup failure

Activate the same virtual environment used for installation and reinstall with `pip install -r requirements.txt`. Run `python main.py` from the repository root to retain expected relative paths.

## Dataset Check path errors

Confirm that `data.yaml` exists and that its `path`, `train`, `val`, and optional `test` entries resolve to real folders or supported image lists.

## Roboflow relative YAML paths

Some exports use paths such as `../train/images`. YOLO Trainer UI includes compatibility handling, but the exported folder structure must remain intact. Prefer keeping `data.yaml` at the dataset root when reorganizing files.

## Predict produces no labels

Enable **Save TXT labels** and **Save confidence values** before prediction. Error Mining needs those TXT results for precise comparison.

## Dataset Builder output already exists

Choose an empty output folder, or explicitly enable overwrite after confirming the target is disposable. With overwrite disabled, the builder safely refuses to replace existing content.

## Portable executable starts slowly

The onedir package contains Python, Qt, and ML dependencies. First launch may take longer while Windows loads many files or Windows Defender scans the extracted folder. Extract the entire ZIP first and keep `_internal/` beside `YOLO-Trainer-UI.exe`.

## Portable executable does not open

Confirm the ZIP was fully extracted, `_internal/` is present, and Windows is not still scanning the folder. Run the source version to distinguish packaging problems from local GPU or dependency problems, then report reproducible details through GitHub Issues.

## GPU is unavailable in the portable build

Packaging does not make incompatible GPU environments compatible. Verify the NVIDIA driver and the runtime requirements for the included PyTorch build. UI-only features remain usable without CUDA, while training may fall back to the CPU.
