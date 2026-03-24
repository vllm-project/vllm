# Apple Silicon Quickstart

Apple Silicon local usage is supported through a plugin-aware workflow.

## Design Direction

The local CLI does not pretend that core vLLM has a built-in Apple GPU backend. Instead, it follows vLLM's hardware-plugin architecture:

- if an Apple GPU plugin such as `vllm-metal` is installed and detected, the local CLI prefers that backend
- if no Apple GPU plugin is available, the local CLI falls back to CPU and explains why

This keeps Apple support aligned with upstream vLLM architecture instead of spreading Apple-specific logic through the core runtime.

## Recommended Flow

1. Install `uv` using Astral's official instructions.
2. Install this checkout with `./scripts/install.sh`.
3. Run diagnostics:

```bash
vllm doctor
vllm doctor deepseek-r1:8b
```

## What `doctor` Tells You

On Apple Silicon, the report explains:

- whether the machine is `Darwin/arm64`
- whether an Apple GPU plugin is installed
- whether the CLI selected `apple-metal` or fell back to `cpu`
- the fallback reason if CPU is being used
- a rough fit estimate for the requested model

## Current Expectations

- Apple GPU acceleration is plugin-based, not hardwired into core CLI logic.
- CPU fallback remains available when no Apple GPU plugin is present.
- The local runtime can still be used for model management, service management, and diagnostics even when the backend is CPU-only.

## Follow-up Direction

Future work should continue to prefer:

- MLX / Metal-oriented plugin paths
- plugin-provided capability reporting
- explicit backend selection diagnostics

instead of making raw Torch MPS the primary design center.
