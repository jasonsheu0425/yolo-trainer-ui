# Training Result Analysis

Training Result Analysis reads existing YOLO training artifacts, primarily
`results.csv`, and produces deterministic, evidence-first guidance. It does
not use an LLM, cloud service, GPU, model loading, retraining, or validation.
The reference rating is guidance only and does not guarantee real-world
deployment performance.

Open **Training Analysis** in Advanced Mode, select a training run folder, and
choose **Analyze Run**. Simple Mode sends its completed run to the same page.
The Run Browser exposes **Analyze Run** for train-type rows, and the Advanced
Train page exposes **Analyze Results** after a successful run has `results.csv`.

## What is evaluated

- Precision / Recall gap: a difference of at least 10 percentage points offers
  a review action for false negatives or false positives.
- Localization gap: a large mAP50 to mAP50-95 difference offers a Low-IoU
  review action.
- Trend: with at least ten usable epochs, the analyzer can report a plateau or
  continued improvement.
- Possible overfitting signal: this is intentionally conservative. It needs at
  least two independent late-training signals among training loss decreasing,
  validation loss increasing, mAP falling, and a clearly earlier best epoch.

All thresholds live in `AnalysisThresholds`. Findings retain stable English IDs
in the JSON cache; translated UI text never changes the stored schema.

## Cache and artifacts

The page writes `training_analysis.json` atomically beside `results.csv`. It
contains schema and heuristic versions, an ISO timestamp, source CSV SHA-256,
metrics, rating, findings, and recommendation IDs. A matching SHA-256 is a
verified cache hit. If `results.csv` changed, malformed cache data is ignored
and regenerated. If the CSV was deleted, a valid cache is shown only as an
explicitly **unverified** reference. **Re-analyze** bypasses the cache.

The page can safely open available artifacts and previews `results.png` plus a
normalized confusion matrix (or the standard confusion matrix fallback). No
per-class metric is invented when the run does not contain a reliable source.

## Limitations

Aggregate metrics cannot reveal specific scene-level causes without supporting
evidence. The analyzer therefore does not claim that a dataset needs more
distant targets, different lighting, specific augmentation, class balancing,
or label fixes. Its review actions navigate to existing Error Mining workflows
where available evidence can be inspected.
