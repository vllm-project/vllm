.. _installation_cpu:

Installation with CPU
========================

vLLM initially supports basic model inferencing and serving on x86 CPU platform, with data types FP32, FP16 and BF16. vLLM CPU backend supports the following vLLM features:

- Tensor Parallel 
- Model Quantization (``INT8 W8A8, AWQ``)
- Chunked-prefill
- Prefix-caching
- FP8-E5M2 KV-Caching (TODO)

Table of contents:

#. :ref:`Requirements <cpu_backend_requirements>`
#. :ref:`Quick start using Dockerfile <cpu_backend_quick_start_dockerfile>`
#. :ref:`Build from source <build_cpu_backend_from_source>`
#. :ref:`Related runtime environment variables <env_intro>`
#. :ref:`Intel Extension for PyTorch <ipex_guidance>`
#. :ref:`Performance tips <cpu_backend_performance_tips>`

.. _cpu_backend_requirements:

Requirements
------------

* OS: Linux
* Compiler: gcc/g++>=12.3.0 (optional, recommended)
* Instruction set architecture (ISA) requirement: AVX512 (optional, recommended)

.. _cpu_backend_quick_start_dockerfile:

Quick start using Dockerfile
----------------------------

.. code-block:: console

    $ docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
    $ docker run -it \
                 --rm \
                 --network=host \
                 --cpuset-cpus=<cpu-id-list, optional> \
                 --cpuset-mems=<memory-node, optional> \
                 vllm-cpu-env

.. _build_cpu_backend_from_source:

Build from source
-----------------

- First, install recommended compiler. We recommend to use ``gcc/g++ >= 12.3.0`` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

.. code-block:: console

    $ sudo apt-get update  -y
    $ sudo apt-get install -y gcc-12 g++-12 libnuma-dev
    $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

- Second, install Python packages for vLLM CPU backend building:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install cmake>=3.26 wheel packaging ninja "setuptools-scm>=8" numpy
    $ pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

- Finally, build and install vLLM CPU backend: 

.. code-block:: console

    $ VLLM_TARGET_DEVICE=cpu python setup.py install

.. note::
    - AVX512_BF16 is an extension ISA provides native BF16 data type conversion and vector product instructions, will brings some performance improvement compared with pure AVX512. The CPU backend build script will check the host CPU flags to determine whether to enable AVX512_BF16. 
    
    - If you want to force enable AVX512_BF16 for the cross-compilation, please set environment variable VLLM_CPU_AVX512BF16=1 before the building.    

.. _env_intro:

Related runtime environment variables
-------------------------------------

- ``VLLM_CPU_KVCACHE_SPACE``: specify the KV Cache size (e.g, ``VLLM_CPU_KVCACHE_SPACE=40`` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.

- ``VLLM_CPU_OMP_THREADS_BIND``: specify the CPU cores dedicated to the OpenMP threads. For example, ``VLLM_CPU_OMP_THREADS_BIND=0-31`` means there will be 32 OpenMP threads bound on 0-31 CPU cores. ``VLLM_CPU_OMP_THREADS_BIND=0-31|32-63`` means there will be 2 tensor parallel processes, 32 OpenMP threads of rank0 are bound on 0-31 CPU cores, and the OpenMP threads of rank1 are bound on 32-63 CPU cores.

.. _ipex_guidance:

Intel Extension for PyTorch
---------------------------

- `Intel Extension for PyTorch (IPEX) <https://github.com/intel/intel-extension-for-pytorch>`_ extends PyTorch with up-to-date features optimizations for an extra performance boost on Intel hardware.

.. _cpu_backend_performance_tips:

Performance tips
-----------------

- We highly recommend to use TCMalloc for high performance memory allocation and better cache locality. For example, on Ubuntu 22.4, you can run:

.. code-block:: console

    $ sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
    $ find / -name *libtcmalloc* # find the dynamic link library path
    $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
    $ python examples/offline_inference.py # run vLLM

- When using the online serving, it is recommended to reserve 1-2 CPU cores for the serving framework to avoid CPU oversubscription. For example, on a platform with 32 physical CPU cores, reserving CPU 30 and 31 for the framework and using CPU 0-29 for OpenMP:

.. code-block:: console

    $ export VLLM_CPU_KVCACHE_SPACE=40
    $ export VLLM_CPU_OMP_THREADS_BIND=0-29 
    $ vllm serve facebook/opt-125m

- If using vLLM CPU backend on a machine with hyper-threading, it is recommended to bind only one OpenMP thread on each physical CPU core using ``VLLM_CPU_OMP_THREADS_BIND``. On a hyper-threading enabled platform with 16 logical CPU cores / 8 physical CPU cores:

.. code-block:: console

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
    $ python examples/offline_inference.py

- If using vLLM CPU backend on a multi-socket machine with NUMA, be aware to set CPU cores using ``VLLM_CPU_OMP_THREADS_BIND`` to avoid cross NUMA node memory access.

CPU Backend Considerations
--------------------------

- The CPU backend significantly differs from the GPU backend since the vLLM architecture was originally optimized for GPU use. A number of optimizations are needed to enhance its performance.

- Decouple the HTTP serving components from the inference components. In a GPU backend configuration, the HTTP serving and tokenization tasks operate on the CPU, while inference runs on the GPU, which typically does not pose a problem. However, in a CPU-based setup, the HTTP serving and tokenization can cause significant context switching and reduced cache efficiency. Therefore, it is strongly recommended to segregate these two components for improved performance.

- On CPU based setup with NUMA enabled, the memory access performance may be largely impacted by the `topology <https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/performance_tuning/tuning_guide.md#non-uniform-memory-access-numa>`_. For NUMA architecture, two optimizations are to recommended: Tensor Parallel or Data Parallel.  

  * Using Tensor Parallel for a latency constraints deployment: following GPU backend design, a Megatron-LM's parallel algorithm will be used to shard the model, based on the number of NUMA nodes (e.g. TP = 2 for a two NUMA node system). With `TP feature on CPU <https://github.com/vllm-project/vllm/pull/6125>`_ merged, Tensor Parallel is supported for serving and offline inferencing. In general each NUMA node is treated as one GPU card. Below is the example script to enable Tensor Parallel = 2 for serving:

    .. code-block:: console

         $ VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-31|32-63" vllm serve meta-llama/Llama-2-7b-chat-hf -tp=2 --distributed-executor-backend mp


  * Using Data Parallel for maximum throughput: to launch an LLM serving endpoint on each NUMA node along with one additional load balancer to dispatch the requests to those endpoints. Common solutions like `Nginx <../serving/deploying_with_nginx.html>`_ or HAProxy are recommended. Anyscale Ray project provides the feature on LLM `serving <https://docs.ray.io/en/latest/serve/index.html>`_. Here is the example to setup a scalable LLM serving with `Ray Serve <https://github.com/intel/llm-on-ray/blob/main/docs/setup.md>`_.