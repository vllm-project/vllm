# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

!!! note
    vLLM boolean environment variables are parsed uniformly: valid values are `true`, `false`, `1`, `0`, `yes`, `no`, `on` and `off` (**case-insensitive**, leading/trailing whitespace ignored, and an empty value is treated as unset). Any other value is rejected with an error naming the variable — for example `VLLM_BATCH_INVARIANT=2` fails loudly instead of being silently interpreted as enabled. The two exceptions are the de-facto standard flags `NO_COLOR` ([no-color.org](https://no-color.org): any non-empty value disables colored output) and `FORCE_COLOR` ([force-color.org](https://force-color.org): any value other than `0` forces colored output, including level values like `2`), which are read during logger initialization and therefore never raise.

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
