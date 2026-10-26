# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

```python
--8<-- "vllm/envs.py:env-vars-definition"
```

For the measured DeepSeek V4 DEP configuration and its limits, see
[DeepSeek V4 DEP on MI355X](deepseek_v4_dep.md).

## DeepSeek V4 sparse MLA on AMD MI355X

`VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA` enables AITER's native-layout ASM attention
for DeepSeek V4 decode on MI355X (`gfx950`). It is disabled by default. To enable
it, set the variable before starting your DeepSeek V4 server:

```bash
export VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA=1
```

This path requires tensor parallelism of one, 128 local attention heads, and
FP8 MLA KV caches. Use `--tensor-parallel-size 1 --kv-cache-dtype fp8`; DeepSeek
V4 resolves the cache dtype to `fp8_ds_mla`. For an eight-GPU deployment, combine
this with `--data-parallel-size 8 --enable-expert-parallel` in your serving
configuration.

Decode uses FP8 queries and a native KV layout with a separate BF16 RoPE pool.
Prefill uses the default attention path with the same cached values. The cache
planner accounts for the additional pool, so fewer tokens fit at a fixed GPU
memory budget. Assess model quality on your workload when changing query
precision.

Larger compressed decode batches use single-pass CSA or persistent HCA kernels.
Short rows receive an FP32 softmax/value correction. The persistent workspace is
allocated before GPU memory profiling and reduces the memory available for KV
caches.

KV transfer connectors and CPU KV offload are unsupported while this variable
is enabled. The sparse indexer's KV dtype remains a separate setting under
`--attention-config`.

## DeepSeek V4 FP8 block scales on AMD MI355X

`VLLM_ROCM_USE_AITER_DSV4_FP8_MXSCALE=1` enables native UE8M0 activation
and weight scales for compatible attention and shared-expert projections,
including the DSpark draft model and its input projection.
It requires AITER on `gfx950`, tensor parallelism of one, and FP8 weights
with group-128 block scales that can be represented exactly as UE8M0.
BF16 activations are required. The ASM sparse attention path also uses a fused
query RMSNorm and UE8M0 quantizer for the attention and indexer query projections.
Both consumers must have compatible scales before this producer is enabled.
Other layers retain their configured quantization path.

The option is disabled by default. It quantizes activation scales to powers
of two and selects AITER's native scaled GEMMs. Evaluate model quality when
changing activation quantization. Checkpoint weight scales are preserved
losslessly; unsupported weights retain their FP32 scale path.

## DeepSeek V4 indexer on AMD MI355X

The MXFP4 indexer uses AITER's 1024-key top-k on `gfx950` with a
262144-column logits plane, including small decode batches. This is the
layout for DeepSeek V4's 1M token context with compression by four.
The dispatch uses the device and tensor shape and supports changing row lengths
inside a captured graph.
