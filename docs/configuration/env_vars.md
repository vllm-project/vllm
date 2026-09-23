# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

```python
--8<-- "vllm/envs.py:env-vars-definition"
```

## Rust frontend

`VLLM_RS_ITL_FLUSH_INTERVAL_TOKENS` sets the Rust frontend's ITL publication
interval in generated tokens per request. It accepts a positive `u32` (1 through
4294967295) and defaults to **32**. Invalid values warn and fall back to 32.
The Rust frontend reads the setting once during initialization; set it before
startup. Use `1` to publish on each eligible output update.

See [Rust frontend ITL publication](../design/metrics.md#rust-frontend-itl-publication)
for the time-based trigger, termination behavior, and visibility limits.
