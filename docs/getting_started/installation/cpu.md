# CPU

vLLM is a Python library that supports the following CPU variants. Select your CPU type to see vendor specific instructions:

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:installation"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:installation"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:installation"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:installation"

## Technical Discussions

The main discussions happen in the `#sig-cpu` channel of [vLLM Slack](https://slack.vllm.ai/).

When open a Github issue about the CPU backend, please add `[CPU Backend]` in the title and it will be labeled with `cpu` for better awareness.

## Requirements

- Python: 3.10 -- 3.13

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:requirements"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:requirements"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:requirements"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:requirements"

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### Pre-built wheels

When specifying the index URL, please make sure to use the `cpu` variant subdirectory.
For example, the nightly build index is: `https://wheels.vllm.ai/nightly/cpu/`.

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-wheels"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:pre-built-wheels"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:pre-built-wheels"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:pre-built-wheels"

### Build wheel from source

#### Set up using Python-only build (without compilation) {#python-only-build}

Please refer to the instructions for [Python-only build on GPU](./gpu.md#python-only-build), and replace the build commands with:

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu uv pip install --editable .
```

#### Full build (with compilation) {#full-build}

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-wheel-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-wheel-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-wheel-from-source"

=== "IBM Z (s390x)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-images"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:pre-built-images"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:pre-built-images"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:pre-built-images"

### Build image from source

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-image-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-image-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-image-from-source"

=== "IBM Z (S390X)"
    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-image-from-source"

## Related runtime environment variables

- `VLLM_CPU_KVCACHE_SPACE`: specify the KV Cache size (e.g, `VLLM_CPU_KVCACHE_SPACE=40` means 40 GiB space for KV cache), larger setting will allow vLLM to run more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users. Default value is `0`.
- `VLLM_CPU_OMP_THREADS_BIND`: specify the CPU cores dedicated to the OpenMP threads, can be set as CPU id lists, `auto` (by default), or `nobind` (to disable binding to individual CPU cores and to inherit user-defined OpenMP variables). For example, `VLLM_CPU_OMP_THREADS_BIND=0-31` means there will be 32 OpenMP threads bound on 0-31 CPU cores. `VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` means there will be 2 tensor parallel processes, 32 OpenMP threads of rank0 are bound on 0-31 CPU cores, and the OpenMP threads of rank1 are bound on 32-63 CPU cores. By setting to `auto`, the OpenMP threads of each rank are bound to the CPU cores in each NUMA node respectively. If set to `nobind`, the number of OpenMP threads is determined by the standard `OMP_NUM_THREADS` environment variable.
- `VLLM_CPU_NUM_OF_RESERVED_CPU`: specify the number of CPU cores which are not dedicated to the OpenMP threads for each rank. The variable only takes effect when VLLM_CPU_OMP_THREADS_BIND is set to `auto`. Default value is `None`. If the value is not set and use `auto` thread binding, no CPU will be reserved for `world_size == 1`, 1 CPU per rank will be reserved for `world_size > 1`.
- `CPU_VISIBLE_MEMORY_NODES`: specify visible NUMA memory nodes for vLLM CPU workers, similar to ```CUDA_VISIBLE_DEVICES```. The variable only takes effect when VLLM_CPU_OMP_THREADS_BIND is set to `auto`. The variable provides more control for the auto thread-binding feature, such as masking nodes and changing nodes binding sequence.
- `VLLM_CPU_SGL_KERNEL` (x86 only, Experimental): whether to use small-batch optimized kernels for linear layer and MoE layer, especially for low-latency requirements like online serving. The kernels require AMX instruction set, BFloat16 weight type and weight shapes divisible by 32. Default is `0` (False).

## FAQ

### Which `dtype` should be used?

- Currently, vLLM CPU uses model default settings as `dtype`. However, due to unstable float16 support in torch CPU, it is recommended to explicitly set `dtype=bfloat16` if there are any performance or accuracy problem.  

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

Note, it is recommended to manually reserve 1 CPU for vLLM front-end process when `world_size == 1`.

### What are supported models on CPU?

For the full and up-to-date list of models validated on CPU platforms, please see the official documentation: [Supported Models on CPU](../../models/hardware_supported_models/cpu.md)

### How to find benchmark configuration examples for supported CPU models?

For any model listed under [Supported Models on CPU](../../models/hardware_supported_models/cpu.md), optimized runtime configurations are provided in the vLLM Benchmark Suite’s CPU test cases, defined in [cpu test cases](../../../.buildkite/performance-benchmarks/tests/serving-tests-cpu.json)
For details on how these optimized configurations are determined, see: [performance-benchmark-details](../../../.buildkite/performance-benchmarks/README.md#performance-benchmark-details).
To benchmark the supported models using these optimized settings, follow the steps in [running vLLM Benchmark Suite manually](../../benchmarking/dashboard.md#manually-trigger-the-benchmark) and run the Benchmark Suite on a CPU environment.  

Below is an example command to benchmark all CPU-supported models using optimized configurations.

```bash
ON_CPU=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

The benchmark results will be saved in `./benchmark/results/`.
In the directory, the generated `.commands` files contain all example commands for the benchmark.

We recommend configuring tensor-parallel-size to match the number of NUMA nodes on your system. Note that the current release does not support tensor-parallel-size=6.
To determine the number of NUMA nodes available, use the following command:

```bash
lscpu | grep "NUMA node(s):" | awk '{print $3}'
```

For performance reference, users may also consult the [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm&deviceName=cpu)
, which publishes default-model CPU results produced using the same Benchmark Suite.

### How to decide `VLLM_CPU_OMP_THREADS_BIND`?

- Default `auto` thread-binding is recommended for most cases. Ideally, each OpenMP thread will be bound to a dedicated physical core respectively, threads of each rank will be bound to the same NUMA node respectively, and 1 CPU per rank will be reserved for other vLLM components when `world_size > 1`. If you have any performance problems or unexpected binding behaviours, please try to bind threads as following.

- On a hyper-threading enabled platform with 16 logical CPU cores / 8 physical CPU cores:

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

    # On this platform, it is recommended to only bind openMP threads on logical CPU cores 0-7 or 8-15
    $ export VLLM_CPU_OMP_THREADS_BIND=0-7
    $ python examples/offline_inference/basic/basic.py
    ```

- When deploying vLLM CPU backend on a multi-socket machine with NUMA and enable tensor parallel or pipeline parallel, each NUMA node is treated as a TP/PP rank. So be aware to set CPU cores of a single rank on the same NUMA node to avoid cross NUMA node memory access.

### How to decide `VLLM_CPU_KVCACHE_SPACE`?

This value is 4GB by default. Larger space can support more concurrent requests, longer context length. However, users should take care of memory capacity of each NUMA node. The memory usage of each TP rank is the sum of `weight shard size` and `VLLM_CPU_KVCACHE_SPACE`, if it exceeds the capacity of a single NUMA node, the TP worker will be killed with `exitcode 9` due to out-of-memory.

### How to do performance tuning for vLLM CPU?

First of all, please make sure the thread-binding and KV cache space are properly set and take effect. You can check the thread-binding by running a vLLM benchmark and observing CPU cores usage via `htop`.

Use multiples of 32 as `--block-size`, which is 128 by default.

Inference batch size is an important parameter for the performance. A larger batch usually provides higher throughput, a smaller batch provides lower latency. Tuning the max batch size starting from the default value to balance throughput and latency is an effective way to improve vLLM CPU performance on specific platforms. There are two important related parameters in vLLM:

- `--max-num-batched-tokens`, defines the limit of token numbers in a single batch, has more impacts on the first token performance. The default value is set as:
    - Offline Inference: `4096 * world_size`
    - Online Serving: `2048 * world_size`
- `--max-num-seqs`, defines the limit of sequence numbers in a single batch, has more impacts on the output token performance.
    - Offline Inference: `256 * world_size`
    - Online Serving: `128 * world_size`

vLLM CPU supports data parallel (DP), tensor parallel (TP) and pipeline parallel (PP) to leverage multiple CPU sockets and memory nodes. For more details of tuning DP, TP and PP, please refer to [Optimization and Tuning](../../configuration/optimization.md). For vLLM CPU, it is recommended to use DP, TP and PP together if there are enough CPU sockets and memory nodes.

### Which quantization configs does vLLM CPU support?

- vLLM CPU supports quantizations:
    - AWQ (x86 only)
    - GPTQ (x86 only)
    - compressed-tensor INT8 W8A8 (x86, s390x)

### Why do I see `get_mempolicy: Operation not permitted` when running in Docker?

In some container environments (like Docker), NUMA-related syscalls used by vLLM (e.g., `get_mempolicy`, `migrate_pages`) are blocked/denied in the runtime's default seccomp/capabilities settings. This may lead to warnings like `get_mempolicy: Operation not permitted`. Functionality is not affected, but NUMA memory binding/migration optimizations may not take effect and performance can be suboptimal.

To enable these optimizations inside Docker with the least privilege, you can follow below tips:

```bash
docker run ... --cap-add SYS_NICE --security-opt seccomp=unconfined  ...

# 1) `--cap-add SYS_NICE` is to address `get_mempolicy` EPERM issue.

# 2) `--security-opt seccomp=unconfined` is to enable `migrate_pages` for `numa_migrate_pages()`.
# Actually, `seccomp=unconfined` bypasses the seccomp for container,
# if it's unacceptable, you can customize your own seccomp profile,
# based on docker/runtime default.json and add `migrate_pages` to `SCMP_ACT_ALLOW` list.

# reference : https://docs.docker.com/engine/security/seccomp/
```

Alternatively, running with `--privileged=true` also works but is broader and not generally recommended.

In K8S, the following configuration can be added to workload yaml to achieve the same effect as above:

```yaml
securityContext:
  seccompProfile:
    type: Unconfined
  capabilities:
    add:
    - SYS_NICE
```
