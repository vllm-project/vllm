# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

```python
--8<-- "vllm/envs.py:env-vars-definition"
```

## Experimental Kimi-K3.1 AttnRes norm cache on ROCm

`VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE=1` enables reuse of historical AttnRes
reciprocal RMS values within one native model forward. It defaults to `0`.
The initial block is recomputed by each consumer; later blocks publish a norm
only when their live prefix is written. Layer-specific norm and projection
weights and output normalization still run normally.

The current admission is limited to gfx950, BF16 residuals, 93 layers,
AttnRes block size 12, hidden size 7168, epsilon `1e-5`, TP8/DCP8, PP1/DP1,
and 280 or 336 input rows. Other shapes and configurations retain the native
path. Microbatching, DBO, LoRA, batch-invariant execution and specialized decode
are excluded. Set the environment variable before constructing the model.

The FP32 cache occupies eight values per input row (8,960 or 10,752 bytes).
It is a forward-local tensor passed explicitly between layers, not a model or
workspace resource retained across requests. During graph capture its storage
follows the graph's ordinary intermediate-tensor lifetime; each replay rewrites
every reusable slot before reading it. The existing residual-bank allocation is
unchanged.

This is an experimental option. Kernel sequence tests cover writer/consumer
ordering, BF16 and fused FP8 outputs, and shared-pool replay with changed inputs
and alternating shapes. Native V2 memory-profiling-to-serving capture, compiled
FULL/PIECEWISE replay, model quality and serving performance still require
qualification before production use. No request-latency improvement is claimed.
