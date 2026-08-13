# ADR-003: Worker and Runner Model

**Decision:** retain QProcess for YOLO CLI execution and QThread/QObject for
long filesystem/runtime work. Widgets are updated only through Qt signals on
the GUI thread.

Runners use the shared vocabulary `idle`, `running`, `cancelling`,
`completed`, `failed`, and `cancelled`. The established boolean runner signal
is retained for compatibility while new consumers can use the stable state ID.
No universal runner with mode booleans is introduced; training, prediction,
validation, and export may keep their thin semantic wrappers over a common
QProcess implementation.
