.. _installation_cpu:

Installation with CPU
========================

vLLM initially supports basic model inferencing and serving on x86 CPU platform, with data types FP32 and BF16.

Table of contents:

#. :ref:`Requirements <requirements>`
#. :ref:`Quick start using Dockerfile <quick_start_dockerfile>`
#. :ref:`Build from source <build_from_source>`
#. :ref:`Performance tips <performance_tips>`

.. _requirements:

Requirements
------------

* OS: Linux
* Compiler: gcc/g++>=12.3.0 (recommended)
* Instruction set architecture (ISA) requirement: AVX512 is required.

.. _quick_start_dockerfile:

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

.. _build_from_source:

Build from source
-----------------

0. Install compiler

We recommend to use ``gcc/g++ >= 12.3.0`` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

.. code-block:: console

    $ sudo apt-get update  -y
    $ sudo apt-get install -y gcc-12 g++-12
    $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

1. Install packages

Install Python packages for vLLM CPU backend building:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install wheel packaging ninja setuptools>=49.4.0 numpy
    $ pip install -v -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu

2. Build and install vLLM CPU backend 

.. code-block:: console

    $ VLLM_TARGET_DEVICE=cpu python setup.py install

.. _performance_tips:

Performance tips
-----------------

0. vLLM CPU backend uses ``swap_space`` parameter to specify the KV Cache size (e.g, ``--swap-space=40`` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.

1. vLLM CPU backend uses OpenMP for thread-parallel computation. If you want the best performance on CPU, it will be very critical to isolate CPU cores for OpenMP threads with other thread pools (like web-service event-loop), to avoid CPU oversubscription. 

2. If using vLLM CPU backend on a bare-metal machine, it is recommended to disable the hyper-threading.

3. If using vLLM CPU backend on a multi-socket machine with NUMA, be aware to set CPU cores and memory nodes, to avoid the remote memory node access. ``numactl`` is an useful tool for CPU core and memory binding on NUMA platform. Besides, ``--cpuset-cpus`` and ``--cpuset-mems`` arguments of ``docker run`` are also useful.



