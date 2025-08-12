# CPU

vLLM is a Python library that supports the following CPU variants. Select your CPU type to see vendor specific instructions:

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:installation"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:installation"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu/apple.inc.md:installation"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:installation"

## Requirements

- Python: 3.9 -- 3.12

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:requirements"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:requirements"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu/apple.inc.md:requirements"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:requirements"

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### Pre-built wheels

Currently, there are no pre-built CPU wheels.

### Build wheel from source

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:build-wheel-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:build-wheel-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu/apple.inc.md:build-wheel-from-source"

=== "IBM Z (s390x)"

    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:pre-built-images"

### Build image from source

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu/x86.inc.md:build-image-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:build-image-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu/arm.inc.md:build-image-from-source"

=== "IBM Z (S390X)"
    --8<-- "docs/getting_started/installation/cpu/s390x.inc.md:build-image-from-source"

## Related runtime environment variables

- `VLLM_CPU_KVCACHE_SPACE`: specify the KV Cache size (e.g, `VLLM_CPU_KVCACHE_SPACE=40` means 40 GiB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users. Default value is `0`.
- `VLLM_CPU_OMP_THREADS_BIND`: specify the CPU cores dedicated to the OpenMP threads. For example, `VLLM_CPU_OMP_THREADS_BIND=0-31` means there will be 32 OpenMP threads bound on 0-31 CPU cores. `VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` means there will be 2 tensor parallel processes, 32 OpenMP threads of rank0 are bound on 0-31 CPU cores, and the OpenMP threads of rank1 are bound on 32-63 CPU cores. By setting to `auto`, the OpenMP threads of each rank are bound to the CPU cores in each NUMA node. By setting to `all`, the OpenMP threads of each rank uses all CPU cores available on the system. Default value is `auto`.
- `VLLM_CPU_NUM_OF_RESERVED_CPU`: specify the number of CPU cores which are not dedicated to the OpenMP threads for each rank. The variable only takes effect when VLLM_CPU_OMP_THREADS_BIND is set to `auto`. Default value is `0`.
- `VLLM_CPU_MOE_PREPACK` (x86 only): whether to use prepack for MoE layer. This will be passed to `ipex.llm.modules.GatedMLPMOE`. Default is `1` (True). On unsupported CPUs, you might need to set this to `0` (False).
- `VLLM_CPU_SGL_KERNEL` (x86 only, Experimental): whether to use small-batch optimized kernels for linear layer and MoE layer, especially for low-latency requirements like online serving. The kernels require AMX instruction set, BFloat16 weight type and weight shapes divisible by 32. Default is `0` (False).

## FAQ

### Which `dtype` should be used?

- Currently vLLM CPU uses model default settings as `dtype`. However, due to unstable float16 support in torch CPU, it is recommended to explicitly set `dtype=bfloat16` if there are any performance or accuracy problem.  

### How to launch a vLLM service on CPU?

- When using the online serving, it is recommended to reserve 1-2 CPU cores for the serving framework to avoid CPU oversubscription. For example, on a platform with 32 physical CPU cores, reserving CPU 31 for the framework and using CPU 0-30 for inference threads:

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve facebook/opt-125m --dtype=bfloat16
```

 or using default auto thread binding:

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=bfloat16
```

### How to decide `VLLM_CPU_OMP_THREADS_BIND`?

- Bind each OpenMP thread to a dedicated physical CPU core respectively, or use auto thread binding feature by default. On a hyper-threading enabled platform with 16 logical CPU cores / 8 physical CPU cores:

??? console "Commands"

    ```console
    $ lscpu -e # check the mapping between logical CPU cores and physical CPU cores

    # The "CPU" column means the logical CPU core IDs, and the "CORE" column means the physical core IDs. On this platform, two logical cores are sharing one physical core.
    CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
    0    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    1    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    2    0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    3    0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    4    0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    5    0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    6    0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    7    0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000
    8    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    9    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    10   0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    11   0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    12   0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    13   0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    14   0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    15   0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000

    # On this platform, it is recommend to only bind openMP threads on logical CPU cores 0-7 or 8-15
    $ export VLLM_CPU_OMP_THREADS_BIND=0-7
    $ python examples/offline_inference/basic/basic.py
    ```

- When deploy vLLM CPU backend on a multi-socket machine with NUMA and enable tensor parallel or pipeline parallel, each NUMA node is treated as a TP/PP rank. So be aware to set CPU cores of a single rank on a same NUMA node to avoid cross NUMA node memory access.

### How to decide `VLLM_CPU_KVCACHE_SPACE`?

  - This value is 4GB by default. Larger space can support more concurrent requests, longer context length. However, users should take care of memory capacity of each NUMA node. The memory usage of each TP rank is the sum of `weight shard size` and `VLLM_CPU_KVCACHE_SPACE`, if it exceeds the capacity of a single NUMA node, the TP worker will be killed with `exitcode 9` due to out-of-memory.

### Which quantization configs does vLLM CPU support?

  - vLLM CPU supports quantizations:
    - AWQ (x86 only)
    - GPTQ (x86 only)
    - compressed-tensor INT8 W8A8 (x86, s390x)

### (x86 only) What is the purpose of `VLLM_CPU_MOE_PREPACK` and `VLLM_CPU_SGL_KERNEL`?

  - Both of them requires `amx` CPU flag.
    - `VLLM_CPU_MOE_PREPACK` can provides better performance for MoE models
    - `VLLM_CPU_SGL_KERNEL` can provides better performance for MoE models and small-batch scenarios.
