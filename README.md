# YOLO Trainer UI

Windows 桌面版 Ultralytics YOLO 訓練器，使用 PySide6 製作。訓練與匯出都透過 `QProcess` 呼叫 YOLO CLI，因此長時間工作不會凍結 UI，stdout/stderr 會即時顯示在畫面中。

## 功能

- Train：選擇 `data.yaml` 與預訓練模型，設定 epochs、imgsz、batch、device 等參數；支援自訂 CLI 參數、停止程序與定位 `best.pt` / `last.pt`。
- Dataset Check：檢查 YAML 結構、影像與 label 配對、每行 YOLO label 格式、座標範圍、class ID、空標籤、類別數量及 train/val 重複影像。
- Export：匯出 ONNX、TensorRT engine、OpenVINO、CoreML 或 TFLite。
- Monitor / Results：每 2 秒顯示 Torch、CUDA、GPU 使用率、VRAM、溫度及功耗；讀取 `results.csv` 顯示 loss、mAP、precision、recall 曲線。
- Settings：保存 Python、YOLO command、runs folder、model 與 device 預設值。

## 環境需求

- Windows 10/11
- Python 3.10～3.12
- NVIDIA GPU 為選用；若使用 NVIDIA GPU，請安裝與顯示卡/CUDA 相容的 PyTorch 版本。

## 安裝

在 PowerShell 開啟本資料夾：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch 的 CUDA wheel 會依 CUDA 版本而異。若一般安裝沒有 CUDA 支援，請使用 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 提供的命令重新安裝 `torch` 與 `torchvision`。

確認 YOLO CLI：

```powershell
yolo checks
```

## 啟動

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

也可直接雙擊 `start_yolo_trainer.bat`。它會優先使用 `.venv`，若不存在則使用系統 PATH 中的 `python`。

## YOLO Dataset 格式

範例 `data.yaml`：

```yaml
path: datasets/my_dataset
train: images/train
val: images/val
names:
  0: person
  1: helmet
```

對應目錄：

```text
my_dataset/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

每張影像的 label 為同名 `.txt`，每行格式如下，座標皆為 0～1 的相對值：

```text
class_id x_center y_center width height
```

## 使用方式

1. 先到 Settings 確認 `YOLO command`。一般透過 requirements 安裝後填 `yolo` 即可；也可填完整的 `yolo.exe` 路徑。
2. 在 Dataset Check 選取資料集 YAML，先排除錯誤與重要警告。
3. 到 Train 選擇相同 YAML 與 model，設定參數後按 **Start Training**。
4. 完成時畫面會顯示 weights 路徑；若有 `results.csv`，結果頁會自動載入。
5. 到 Export 選擇 `best.pt`，再選取輸出格式。

## 常見問題

- `yolo` 找不到：啟用安裝 ultralytics 的虛擬環境，或在 Settings 填入 `.venv\Scripts\yolo.exe` 的完整路徑。
- CUDA unavailable：先執行 `python -c "import torch; print(torch.cuda.is_available())"`。若為 `False`，通常需要安裝正確的 NVIDIA 驅動與 CUDA 版 PyTorch。
- Windows 路徑包含空白：本程式以參數列表啟動程序，無須自行加引號；Command Preview 會自動顯示必要引號。
- Dataset Check 為避免大型資料集等待過久，每個 split 最多深入檢查前 500 張影像，但仍會統計該 split 的完整影像數。

## Version 2 additions

- **Last Run Summary**：Train 頁會記住最後一次 run folder，並列出 `best.pt`、`last.pt`、`results.csv`、`results.png` 與 `confusion_matrix.png`。可直接開啟檔案或其資料夾；不存在的產物會顯示 `Not found`。
- **Predict / Test**：選擇 `.pt` / `.onnx` 模型與圖片、圖片資料夾或影片，設定 imgsz、conf、IoU、device 後執行非阻塞預測。完成後可直接開啟輸出資料夾。
- **Training Presets**：Train 頁提供 Smoke Test、Small Dataset Conservative、Standard YOLOv8n 與 Higher Accuracy YOLOv8s。選取 preset 只會填入欄位，不會自動開始訓練。

## Version 3 additions

- **Validate / Evaluate**：使用 `.pt` 或 `.onnx` 模型對 `val` / `test` split 執行非阻塞 validation，顯示 metrics、輸出資料夾與常見 validation plots。
- **Validation metrics fallback**：優先讀取 `results.csv`；standalone validation 沒有 CSV 時，會顯示提示並嘗試解析 Ultralytics log summary。
- **Run Browser**：Monitor / Results 內可掃描 runs root，辨識 train、predict、val 或 unknown run，並顯示 weights、results.csv、最後 metrics 與時間。

### v0.3.1 validation metrics persistence

Completed validations save `validation_metrics.json` in their output folder. Run Browser reads `results.csv` first and falls back to this UTF-8 JSON file, so standalone validation metrics remain available after restarting the application.

## Version 4: Error Mining / Hard Cases

Error Mining organizes uncertain or incomplete prediction outputs into review folders that can be used for relabeling and dataset expansion.

1. In **Predict / Test**, enable both **Save TXT labels** and **Save confidence values**. This adds `save_txt=True` and `save_conf=True` to the exact YOLO CLI command.
2. Run prediction, then open **Error Mining** and select the resulting predict run folder.
3. Optionally select the original source folder and a YOLO labels folder. Original source images are preferred when exporting.
4. Set the hard-cases output folder and confidence threshold, select **Scan**, then **Export Hard Cases**.

The export creates category folders plus `hard_cases_report.csv` and `hard_cases_summary.json`. CSV fields are:

- `image_name`: source image filename.
- `category`: `low_confidence`, `no_detection`, `no_label_file`, or `unknown`.
- `image_path` / `label_path`: selected source and label paths.
- `min_confidence`: minimum confidence found in the prediction label.
- `detection_count`: number of prediction label rows.
- `copied_to`: exported image path.
- `notes`: classification details or limitations.

Run Browser shows whether a run-local hard-cases summary exists and displays its primary counts. v0.4 classifies cases mainly from prediction labels and confidence values; it does not yet calculate full ground-truth IoU false-positive or false-negative metrics.
