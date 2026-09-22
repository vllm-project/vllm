# vLLM gRPC API

Protocol definitions for the optional gRPC server exposed by the vLLM Rust frontend.

- `Inference` provides unary and streaming generation.
- `Control` provides discovery and request control.

The `nightly` label is updated daily from vLLM's `main` branch. The `main` label tracks the latest vLLM release, and version labels match tags such as `v0.27.0`. Pin a Buf commit or generated SDK version for reproducible builds.
