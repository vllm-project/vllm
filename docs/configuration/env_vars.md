# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    All environment variables used by vLLM are prefixed with `VLLM_`. **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## vLLM Environment Variables

```python
--8<-- "vllm/envs.py:env-vars-definition"
```

## Third-Party Environment Variables

vLLM depends on several third-party libraries that have their own environment variables. These are not part of vLLM but can affect its behavior.

### FlashInfer

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is used for efficient attention and MoE kernels. It uses JIT compilation and caches compiled kernels. The following environment variables can be used to configure FlashInfer:

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASHINFER_WORKSPACE` | Base directory for FlashInfer workspace (JIT compilation files) | `~/.cache/flashinfer` |
| `FLASHINFER_CACHE_DIR` | Directory for compiled kernel cache. Can be set separately from workspace | Same as `FLASHINFER_WORKSPACE` |
| `FLASHINFER_JIT_THREADS` | Number of threads for JIT compilation | Number of CPU cores |
| `FLASHINFER_CUBINS_REPOSITORY` | URL of the repository to download FlashInfer cubin files. Useful for restricted network environments | NVIDIA public repository |

!!! tip "Containerized Deployments"
    In containerized environments, you may want to:

    - Set `FLASHINFER_WORKSPACE` to a persistent volume to avoid recompilation on container restart
    - Pre-compile kernels during image build to reduce cold-start time
    - Use `VLLM_HAS_FLASHINFER_CUBIN=1` if you've pre-downloaded cubin files

For more details, see the [FlashInfer JIT documentation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/env.py).
