# Using Railway

[Railway](https://railway.com) builds a service from a Git repository and
runs it in a container without a GPU. vLLM therefore runs with the CPU
backend there, using the pre-built `vllm/vllm-openai-cpu` image rather than
compiling from source (a from-source build exceeds Railway's build limits).

The repository ships the Railway configuration:

- `railway.json` selects `docker/Dockerfile.railway` and sets a `/health`
  check with a long timeout to cover model download and load.
- `docker/Dockerfile.railway` starts from the CPU release image and installs
  `docker/entrypoints/railway-entrypoint.sh`.

## Service variables

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `VLLM_MODEL` | `Qwen/Qwen3-0.6B` | Model to serve. Pick one that fits the plan's RAM. |
| `VLLM_ARGS` | empty | Extra `vllm serve` flags, e.g. `--max-model-len 4096 --dtype bfloat16`. |
| `VLLM_CPU_KVCACHE_SPACE` | `2` | KV cache size in GiB. |
| `OMP_NUM_THREADS` | `nproc` | OpenMP threads used by the CPU backend. |
| `HF_TOKEN` | unset | Required for gated models. |
| `VLLM_API_KEY` | unset | Enables API key authentication. |

`PORT` is injected by Railway and must not be set manually.

## Notes

- Railway hosts are AMD EPYC; AVX512 is required for full performance and
  `bfloat16`. On AVX2-only hosts pass `--dtype float32` via `VLLM_ARGS`.
- Model weights are re-downloaded on every deploy unless a
  [volume](https://docs.railway.com/reference/volumes) is mounted at
  `/root/.cache/huggingface`.
- Memory usage is roughly model weights plus `VLLM_CPU_KVCACHE_SPACE`; size
  both against the service's memory limit.
