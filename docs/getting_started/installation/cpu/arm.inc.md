# --8<-- [start:installation]

vLLM has been adapted to work on ARM64 CPUs with NEON support, leveraging the CPU backend initially developed for the x86 platform.

ARM CPU backend currently supports Float32, FP16 and BFloat16 datatypes.

!!! warning
    There are no pre-built wheels or images for this device, so you must build vLLM from source.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- OS: Linux
- Compiler: `gcc/g++ >= 12.3.0` (optional, recommended)
- Instruction Set Architecture (ISA): NEON support is required

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

--8<-- "docs/getting_started/installation/cpu/build.inc.md"

Testing has been conducted on AWS Graviton3 instances for compatibility.

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]
```bash
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-env .

# Launching OpenAI server
docker run --rm \
            --privileged=true \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-env \
            --model=meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```
# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]
