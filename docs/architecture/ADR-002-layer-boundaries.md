# ADR-002: Layer Boundaries

**Decision:** preserve current folders where practical, while enforcing
dependency direction: UI → services → domain/core/persistence/runtime.

Domain, persistence, and reusable core algorithms must not import PySide6
widgets or concrete UI pages. UI pages may use services and public domain
models. MainWindow is the application shell and navigation composition point;
it does not implement YOLO commands, analysis heuristics, or subprocess logic.

Compatibility shims may remain temporarily where established pages expose
public behavior, but they must delegate to the single service implementation.
