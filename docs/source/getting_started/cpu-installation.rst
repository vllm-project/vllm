.. _installation_cpu:

Installation with CPU
========================

vLLM initially supports basic model inferencing and serving on x86 CPU platform, with data types FP32 and BF16.

Table of contents:

#. :ref:`Requirements <cpu_backend_requirements>`
#. :ref:`Quick start using Dockerfile <cpu_backend_quick_start_dockerfile>`
#. :ref:`Build from source <build_cpu_backend_from_source>`
#. :ref:`Intel Extension for PyTorch <ipex_guidance>`
#. :ref:`Performance tips <cpu_backend_performance_tips>`

.. _cpu_backend_requirements:

Requirements
------------

* OS: Linux
* Compiler: gcc/g++>=12.3.0 (optional, recommended)
* Instruction set architecture (ISA) requirement: AVX512 is required.

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
    $ sudo apt-get install -y gcc-12 g++-12
    $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

- Second, install Python packages for vLLM CPU backend building:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install wheel packaging ninja "setuptools>=49.4.0" numpy
    $ pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

- Finally, build and install vLLM CPU backend: 

.. code-block:: console

    $ VLLM_TARGET_DEVICE=cpu python setup.py install

.. note::
    - BF16 is the default data type in the current CPU backend (that means the backend will cast FP16 to BF16), and is compatible will all CPUs with AVX512 ISA support. 

    - AVX512_BF16 is an extension ISA provides native BF16 data type conversion and vector product instructions, will brings some performance improvement compared with pure AVX512. The CPU backend build script will check the host CPU flags to determine whether to enable AVX512_BF16. 
    
    - If you want to force enable AVX512_BF16 for the cross-compilation, please set environment variable VLLM_CPU_AVX512BF16=1 before the building.    

.. _ipex_guidance:

Intel Extension for PyTorch
---------------------------

- `Intel Extension for PyTorch (IPEX) <https://github.com/intel/intel-extension-for-pytorch>`_ extends PyTorch with up-to-date features optimizations for an extra performance boost on Intel hardware.

- IPEX after the ``2.3.0`` can be enabled in the CPU backend by default if it is installed.

.. _cpu_backend_performance_tips:

Performance tips
-----------------

- vLLM CPU backend uses environment variable ``VLLM_CPU_KVCACHE_SPACE`` to specify the KV Cache size (e.g, ``VLLM_CPU_KVCACHE_SPACE=40`` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.

- We highly recommend to use TCMalloc for high performance memory allocation and better cache locality. For example, on Ubuntu 22.4, you can run:

.. code-block:: console

    $ sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
    $ find / -name *libtcmalloc* # find the dynamic link library path
    $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
    $ python examples/offline_inference.py # run vLLM

- vLLM CPU backend uses OpenMP for thread-parallel computation. If you want the best performance on CPU, it will be very critical to isolate CPU cores for OpenMP threads with other thread pools (like web-service event-loop), to avoid CPU oversubscription. 

- If using vLLM CPU backend on a bare-metal machine, it is recommended to disable the hyper-threading.

- If using vLLM CPU backend on a multi-socket machine with NUMA, be aware to set CPU cores and memory nodes, to avoid the remote memory node access. ``numactl`` is an useful tool for CPU core and memory binding on NUMA platform. Besides, ``--cpuset-cpus`` and ``--cpuset-mems`` arguments of ``docker run`` are also useful.



