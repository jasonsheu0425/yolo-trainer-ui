# ADR-001: Language Strategy

**Decision:** retain Python + PySide6 as the primary application language and
framework. Do not perform a wholesale rewrite solely because the project has
grown.

The current codebase benefits from direct access to Ultralytics, PyTorch,
NumPy, OpenCV, and the Python ML/CV ecosystem. Existing investment and stable
Windows packaging also favor incremental modularization.

If profiling later proves a specific hot path (for example million-file
hashing or image similarity) needs native performance, a narrow Rust, C++, or
native-library component may be introduced behind a language-neutral service
API. It is not a reason to rewrite the application.
