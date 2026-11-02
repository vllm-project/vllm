# Gated DeltaNet ReplaySSM with Model Runner V2

This path depends on the GDN standard-decode implementation in
[PR #62004](https://github.com/vllm-project/vllm/pull/62004).
It supports Qwen3.5 standard decoding with the Triton Mamba backend.

For compiled execution, explicitly disable Breakable CUDA graphs and select
`VLLM_COMPILE`. Breakable graphs select a different, uncompiled execution path.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
vllm serve Qwen/Qwen3.5-4B \
    --dtype bfloat16 --tensor-parallel-size 1 \
    --mamba-backend triton --use-replayssm --replayssm-buffer-len 16 \
    --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching \
    --compilation-config '{"mode": 3}'
```

The runner forwards each request's prefill endpoint, gathered in the current
batch order, to the existing GDN metadata builder. The endpoint is refreshed
when a preempted request resumes. The builder uses it to derive ring positions
and flush flags without reading GPU state back to the CPU. Capture padding is
zero-filled, and the existing builder owns the persistent device buffers.

Validation covers single-device GB10, BF16 Qwen3.5-4B, FP32 checkpoint state,
buffer length 16, standard decode, and prefix caching disabled. Other models,
distributed execution, speculative decode, and V2 align-mode prefix caching
are outside this validation. This change does not introduce a new kernel or
change the default compilation settings.
