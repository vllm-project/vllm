# Local Runtime Troubleshooting

## `./scripts/install.sh` says `uv` is missing

Install `uv` manually using Astral's official instructions:

- <https://docs.astral.sh/uv/getting-started/installation/>

The installer intentionally does not use `curl | sh`.

## `vllm doctor` selects CPU on Apple Silicon

That means the CLI did not detect an Apple GPU plugin.

Run:

```bash
vllm doctor
```

and check:

- `selected_backend`
- `fallback_reason`
- the discovered platform plugin list

## `vllm serve` changed my port

If you pass either:

- `--port 8100`
- `--port=8100`

the managed background-service path should honor it exactly.

## A model does not fit locally

Use:

```bash
vllm preflight <model> --profile low-memory
```

Then try:

- a smaller model
- lower-memory profile
- explicit quantization
- a backend with more memory

## TensorRT-LLM looks unavailable

That is expected outside NVIDIA environments. The local diagnostics surface TensorRT-LLM only as an optional NVIDIA interoperability path.
