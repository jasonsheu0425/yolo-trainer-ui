# Annotation Inference JSONL Protocol

Protocol version: **1**. Every stdin/stdout line is one complete UTF-8 JSON object. Pickle, `eval`, shell command strings, temporary-file polling, and network servers are not used.

The worker first emits `hello`. Requests carry unique `request_id` values and use `load_model`, `predict`, `unload_model`, `ping`, or `shutdown`. Responses are `model_loaded`, `prediction_result`, `model_unloaded`, `pong`, `error`, and `shutdown_ack`. Every non-hello response echoes its request ID.

Stdout is protocol-only. Model loading and prediction redirect library stdout to stderr and use `verbose=False`; diagnostics and tracebacks remain on stderr. The controller validates protocol framing/version, and the service validates request ID, expected image identity, finite confidence, dataset class ID, and normalized geometry.

The controller starts configured Python with an argument list and never uses `shell=True`. Centralized model-load/prediction timeouts stop hung workers. App close sends shutdown, waits briefly, then terminate/kill fallback prevents a hung close. Model changes restart the worker so RAM/VRAM is reliably released.
