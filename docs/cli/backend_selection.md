# Backend Selection

The local-runtime CLI adds a thin backend-selection layer for local UX. It does not replace vLLM's platform or plugin architecture.

## Goals

- keep backend choice explicit
- surface fallback reasons clearly
- remain compatible with vLLM's out-of-tree hardware plugin approach
- avoid hidden "magic" that makes performance debugging harder

## Selection Order

With `--backend auto`, the local CLI currently prefers:

1. `apple-metal` on Apple Silicon when an Apple GPU plugin is detected
2. `cuda`
3. `rocm`
4. `xpu`
5. `cpu`

You can override this with:

```bash
vllm doctor --backend cuda
vllm run deepseek-r1:8b --backend cpu
vllm serve qwen2.5:7b-instruct --backend apple-metal
```

If the requested backend is unavailable, the CLI reports the reason instead of silently pretending it worked.

## Profiles

The local CLI also supports small defaulting profiles:

- `balanced`
- `throughput`
- `low-memory`

Profiles only fill in defaults that the user did not set explicitly.

## Diagnostics

Use these commands to inspect the decision:

```bash
vllm doctor
vllm doctor deepseek-r1:8b
vllm preflight deepseek-r1:8b --profile low-memory
```

The report includes:

- selected backend
- available backends
- fallback reason
- profile
- preflight estimate
- TensorRT-LLM relevance for NVIDIA environments
