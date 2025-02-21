# CPU

vLLM is a Python library that supports the following CPU variants. Select your CPU type to see vendor specific instructions:

:::::{tab-set}
:sync-group: device

::::{tab-item} Intel/AMD x86
:selected:
:sync: x86

:::{include} x86.inc.md
:start-after: "# Installation"
:end-before: "## Requirements"
:::

::::

::::{tab-item} ARM AArch64
:sync: arm

:::{include} arm.inc.md
:start-after: "# Installation"
:end-before: "## Requirements"
:::

::::

::::{tab-item} Apple silicon
:sync: apple

:::{include} apple.inc.md
:start-after: "# Installation"
:end-before: "## Requirements"
:::

::::

:::::

## Requirements

- Python: 3.9 -- 3.12

:::::{tab-set}
:sync-group: device

::::{tab-item} Intel/AMD x86
:sync: x86

:::{include} x86.inc.md
:start-after: "## Requirements"
:end-before: "## Set up using Python"
:::

::::

::::{tab-item} ARM AArch64
:sync: arm

:::{include} arm.inc.md
:start-after: "## Requirements"
:end-before: "## Set up using Python"
:::

::::

::::{tab-item} Apple silicon
:sync: apple

:::{include} apple.inc.md
:start-after: "## Requirements"
:end-before: "## Set up using Python"
:::

::::

:::::

## Set up using Python

### Create a new Python environment

:::{include} ../python_env_setup.inc.md
:::

### Pre-built wheels

Currently, there are no pre-built CPU wheels.

### Build wheel from source

:::::{tab-set}
:sync-group: device

::::{tab-item} Intel/AMD x86
:sync: x86

:::{include} x86.inc.md
:start-after: "### Build wheel from source"
:end-before: "## Set up using Docker"
:::

::::

::::{tab-item} ARM AArch64
:sync: arm

:::{include} arm.inc.md
:start-after: "### Build wheel from source"
:end-before: "## Set up using Docker"
:::

::::

::::{tab-item} Apple silicon
:sync: apple

:::{include} apple.inc.md
:start-after: "### Build wheel from source"
:end-before: "## Set up using Docker"
:::

::::

:::::

## Set up using Docker

### Pre-built images

Currently, there are no pre-build CPU images.

### Build image from source

```console
$ docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
$ docker run -it \
             --rm \
             --network=host \
             --cpuset-cpus=<cpu-id-list, optional> \
             --cpuset-mems=<memory-node, optional> \
             vllm-cpu-env
```

::::{tip}
For ARM or Apple silicon, use `Dockerfile.arm`
::::

## Supported features

vLLM CPU backend supports the following vLLM features:

- Tensor Parallel
- Model Quantization (`INT8 W8A8, AWQ, GPTQ`)
- Chunked-prefill
- Prefix-caching
- FP8-E5M2 KV-Caching (TODO)

## Related runtime environment variables

- `VLLM_CPU_KVCACHE_SPACE`: specify the KV Cache size (e.g, `VLLM_CPU_KVCACHE_SPACE=40` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.
- `VLLM_CPU_OMP_THREADS_BIND`: specify the CPU cores dedicated to the OpenMP threads. For example, `VLLM_CPU_OMP_THREADS_BIND=0-31` means there will be 32 OpenMP threads bound on 0-31 CPU cores. `VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` means there will be 2 tensor parallel processes, 32 OpenMP threads of rank0 are bound on 0-31 CPU cores, and the OpenMP threads of rank1 are bound on 32-63 CPU cores.

## Performance tips

- We highly recommend to use TCMalloc for high performance memory allocation and better cache locality. For example, on Ubuntu 22.4, you can run:

```console
sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
find / -name *libtcmalloc* # find the dynamic link library path
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
python examples/offline_inference/basic/basic.py # run vLLM
```

- When using the online serving, it is recommended to reserve 1-2 CPU cores for the serving framework to avoid CPU oversubscription. For example, on a platform with 32 physical CPU cores, reserving CPU 30 and 31 for the framework and using CPU 0-29 for OpenMP:

```console
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-29
vllm serve facebook/opt-125m
```

- If using vLLM CPU backend on a machine with hyper-threading, it is recommended to bind only one OpenMP thread on each physical CPU core using `VLLM_CPU_OMP_THREADS_BIND`. On a hyper-threading enabled platform with 16 logical CPU cores / 8 physical CPU cores:

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

- If using vLLM CPU backend on a multi-socket machine with NUMA, be aware to set CPU cores using `VLLM_CPU_OMP_THREADS_BIND` to avoid cross NUMA node memory access.

## Other considerations

- The CPU backend significantly differs from the GPU backend since the vLLM architecture was originally optimized for GPU use. A number of optimizations are needed to enhance its performance.

- Decouple the HTTP serving components from the inference components. In a GPU backend configuration, the HTTP serving and tokenization tasks operate on the CPU, while inference runs on the GPU, which typically does not pose a problem. However, in a CPU-based setup, the HTTP serving and tokenization can cause significant context switching and reduced cache efficiency. Therefore, it is strongly recommended to segregate these two components for improved performance.

- On CPU based setup with NUMA enabled, the memory access performance may be largely impacted by the [topology](https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/performance_tuning/tuning_guide.inc.md#non-uniform-memory-access-numa). For NUMA architecture, two optimizations are to recommended: Tensor Parallel or Data Parallel.

  - Using Tensor Parallel for a latency constraints deployment: following GPU backend design, a Megatron-LM's parallel algorithm will be used to shard the model, based on the number of NUMA nodes (e.g. TP = 2 for a two NUMA node system). With [TP feature on CPU](gh-pr:6125) merged, Tensor Parallel is supported for serving and offline inferencing. In general each NUMA node is treated as one GPU card. Below is the example script to enable Tensor Parallel = 2 for serving:

    ```console
    VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-31|32-63" vllm serve meta-llama/Llama-2-7b-chat-hf -tp=2 --distributed-executor-backend mp
    ```

  - Using Data Parallel for maximum throughput: to launch an LLM serving endpoint on each NUMA node along with one additional load balancer to dispatch the requests to those endpoints. Common solutions like [Nginx](#nginxloadbalancer) or HAProxy are recommended. Anyscale Ray project provides the feature on LLM [serving](https://docs.ray.io/en/latest/serve/index.html). Here is the example to setup a scalable LLM serving with [Ray Serve](https://github.com/intel/llm-on-ray/blob/main/docs/setup.inc.md).
