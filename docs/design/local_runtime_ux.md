# Local Runtime UX Design Note

## Architecture Overview

The local-runtime work is intentionally layered on top of existing vLLM entrypoints instead of replacing them.

The shape is:

- lightweight launcher for fast help and metadata commands
- CLI-local model alias and local-state helpers
- CLI-local backend diagnostics and selection helpers
- existing vLLM `LLM` and OpenAI-compatible `serve` paths underneath

This keeps local UX improvements separate from the engine, scheduler, batching logic, and production serving internals.

## Why the Local UX Layer Stays Thin

vLLM's core identity is still:

- high-throughput inference
- continuous batching
- production-grade serving
- backend scalability
- observability

The local UX additions are wrappers, defaults, and diagnostics. They do not replace vLLM's execution model with a toy runner.

## Apple Support Strategy

Apple support is handled in a plugin-aware way.

The CLI:

- detects Apple Silicon hosts
- looks for out-of-tree platform plugins
- prefers an Apple GPU path when such a plugin is present
- falls back to CPU with an explicit explanation otherwise

This matches upstream vLLM's plugin architecture and avoids spreading Apple-specific runtime assumptions through core platform logic.

## Performance and Scalability Preservation

The work is designed to preserve the core performance path by:

- keeping fast metadata commands on a lightweight launcher path
- lazy-loading heavier CLI paths only when needed
- preserving the existing OpenAI-compatible serve path
- using small local profiles only as defaulting layers
- exposing diagnostics instead of hiding backend decisions

## TensorRT-LLM Scope

TensorRT-LLM is treated as optional NVIDIA interoperability, not as a replacement for native vLLM CUDA execution.

The current scope is:

- inspect environment relevance
- report likely eligibility
- surface FlashInfer / TensorRT-LLM-related signals in diagnostics

Future work can add export/staging flows separately.

## Known Limitations

- Apple GPU acceleration still depends on an out-of-tree plugin rather than built-in core support.
- Model preflight is heuristic and conservative when exact model metadata is unavailable.
- TensorRT-LLM diagnostics are intentionally environment-focused rather than a full export pipeline.

## Recommended PR Split

1. Local CLI hardening and correctness fixes
2. Backend diagnostics, doctor/status, and preflight
3. Apple plugin-aware auto-selection and docs
4. Performance profiles and local observability polish
5. TensorRT-LLM interoperability enhancements
