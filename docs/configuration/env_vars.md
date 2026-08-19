# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## Inter-process communication spin tuning

When vLLM runs across multiple processes (for example, tensor-parallel inference), the processes exchange data through shared memory. A reader process that checks for new data has two options: *spin* (keep checking on the CPU, for the lowest possible wake latency) or *park* (sleep until the writer notifies it, freeing the CPU). vLLM spins for a short grace period after each read and only parks once the grace expires.

The `VLLM_SHM_BROADCAST_ADAPTIVE_*` variables tune that grace. By default the grace is *adaptive*: vLLM measures how quickly new data usually arrives and adjusts automatically. During a burst of traffic it spins a bit longer (new data is almost certainly about to arrive, so staying awake is cheap and fast); when traffic slows down it parks sooner, so idle deployments no longer keep burning CPU cores just waiting. The bounds (`MIN_GRACE` / `MAX_GRACE`) and the pivot (`BUDGET`) clamp that behavior; most deployments can keep the defaults. The tuning is fully automatic and needs no configuration in typical use.

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
