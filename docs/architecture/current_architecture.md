# Current Architecture Audit (v0.11.1 baseline)

| Area | Actual responsibility and dependency hotspot |
| --- | --- |
| MainWindow | Creates all pages, global mode/language state, and cross-page signal routing. |
| TrainPage | Largest interactive hotspot: widgets, config conversion, command preview, run summary, and process events. |
| SimpleModePage | Guided facade historically reused TrainPage widgets; now uses public typed config conversion. |
| Predict/Validate/Export | Each owns its page-specific input/preview and a thin QProcess runner wrapper. |
| RuntimePage | QThread orchestration over RuntimeManager and RuntimeWorker. |
| Dataset Builder | QThread worker plus file/report workflow; intentionally not moved in this release. |
| Training Analysis | Analyzer, cache, page, artifact preview, and navigation integration. |
| ConfigManager | JSON settings storage; now uses the shared atomic JSON utility. |

Largest Python files are `core/dataset_builder.py`, `core/error_miner.py`,
`ui/error_mining_page.py`, `ui/train_page.py`, and
`core/training_result_analyzer.py`. Existing process duplication is intentionally
thin: Predict, Validate, and Export subclass the common QProcess runner.

Persistence is currently JSON/CSV/YAML filesystem based: settings, validation
metrics, analysis cache, dataset-builder reports, and error-mining reports.
No database is required. Remaining page-level direct dependencies are recorded
in the v0.11.1 refactor report.
