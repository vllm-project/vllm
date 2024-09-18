.. _installation_cpu:

Installation with CPU
========================

vLLM initially supports basic model inferencing and serving on x86 CPU platform, with data types FP32 and BF16.

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
    $ pip install wheel packaging ninja "setuptools>=49.4.0" numpy
    $ pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

- Third, build and install oneDNN library from source:

.. code-block:: console

    $ git clone -b rls-v3.5 https://github.com/oneapi-src/oneDNN.git
    $ cmake -B ./oneDNN/build -S ./oneDNN -G Ninja -DONEDNN_LIBRARY_TYPE=STATIC \ 
        -DONEDNN_BUILD_DOC=OFF \ 
        -DONEDNN_BUILD_EXAMPLES=OFF \ 
        -DONEDNN_BUILD_TESTS=OFF \ 
        -DONEDNN_BUILD_GRAPH=OFF \ 
        -DONEDNN_ENABLE_WORKLOAD=INFERENCE \ 
        -DONEDNN_ENABLE_PRIMITIVE=MATMUL
    $ cmake --build ./oneDNN/build --target install --config Release

- Finally, build and install vLLM CPU backend: 

.. code-block:: console

    $ VLLM_TARGET_DEVICE=cpu python setup.py install

.. note::
    - BF16 is the default data type in the current CPU backend (that means the backend will cast FP16 to BF16), and is compatible will all CPUs with AVX512 ISA support. 

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



