# vLLM gRPC API

Protocol definitions for the optional gRPC server exposed by the vLLM Rust frontend.

- `Inference` provides unary and streaming generation.
- `Control` provides server and model discovery, request aborts, and LoRA management.
- `KvTransfer` provides KV event sources, KV transfer info, and NIXL handshake metadata.
- `RlControl` provides pause/resume, sleep/wake, and weight updates.

`Control` still declares the RPCs that moved to `KvTransfer` and `RlControl`, marked deprecated, for one release.

The `nightly` label is updated daily from vLLM's `main` branch. The `main` label tracks the latest vLLM release, and version labels match tags such as `v0.27.0`. Pin a Buf commit or generated SDK version for reproducible builds.
