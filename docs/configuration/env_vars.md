# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## ROCm FlyDSL FP8 MLA prefill

Set `VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=1` to opt in to direct
FP8 K/V projection and cached-context attention on MI355X (gfx950).
The default is `0`.

This path requires `ROCM_AITER_MLA` with `ROCM_AITER_FA` prefill,
decode context parallelism greater than one, BF16 activations, an FP8 E4M3
KV cache, and MLA dimensions latent/NoPE/RoPE/value = 512/128/64/128.
Select native per-channel FP8 quantization for `kv_b_proj` explicitly;
the attention flag does not change weight quantization.

For example:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=1
vllm serve moonshotai/Kimi-K3 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --attention-backend ROCM_AITER_MLA \
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}' \
    --quantization-config '{"targets":{"*.self_attn.kv_b_proj":"fp8_per_channel"}}'
```

AITER must expose `gather_kv_b_proj_flydsl` with FP8 output descales and
`flydsl_flash_attn_fp8_func` (available in AITER v0.1.26), with its compatible
FlyDSL package installed. Unsupported model dimensions, cache formats, weight
layouts, or missing kernel APIs fail during weight initialization.
Batch invariance is unsupported. Other prefill backends do not use this path.

Cached chunks use FP8 FMHA when the maximum query length is at least 512 and
the maximum context length is at least 8192. Smaller chunks retain BF16
attention. Calls exceeding the kernels' 32-bit indexing limits use the
standard path. Opting in also uses BF16 attention for new-token causal prefill,
bypassing ASM prefill and its metadata allocation on the target model.
Speculative draft models retain their existing attention routing.

The context projection consumes compressed FP8 cache and FP8 weights directly,
with FP32 accumulation, and writes FP8 K/V without a BF16 intermediate.
Q uses dynamic per-tensor quantization. K/V output descales are computed from
the current new-token K/V maxima with a factor of two headroom and a positive
scale floor. Reuse is limited to the current layer's forward; distinct Q slices
are quantized independently. FMHA returns BF16 output and natural-log LSE for
the existing context/new-token merge.

This is an experimental numerical path: cached values can exceed the
new-token-derived range and saturate. Validate model accuracy and performance
on the intended workload before enabling it in production.

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
