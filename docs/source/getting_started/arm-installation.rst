.. _installation_arm:

Installation with ARM CPU
=========================

Based on the CPU backend developed on x86 CPU platform, with the basic vector types implemented on ARM64 with NEON, vLLM is enabled on ARM CPUs.
This document is just about installation, refer to the x86 platform write-up for more information about:

* CPU backend inferencing support
* Runtime environment variables
* Performance tips

Table of contents:

#. :ref:`Requirements <arm_backend_requirements>`
#. :ref:`Quick start using Dockerfile <arm_backend_quick_start_dockerfile>`
#. :ref:`Build from source <build_arm_backend_from_source>`

.. _arm_backend_requirements:

Requirements
------------

* OS: Linux / MacOS
* Docker Desktop if on MacOS
* Compiler: gcc/g++>=12.3.0 (optional, recommended)
* Instruction set architecture (ISA) requirement: Neon (basic requirement)

.. _arm_backend_quick_start_dockerfile:

Quick start using Dockerfile
----------------------------

.. code-block:: console

    $ docker build -f Dockerfile.arm -t vllm-cpu-env --shm-size=4g .
    $ docker run -it \
                 --rm \
                 --network=host \
                 --cpuset-cpus=<cpu-id-list, optional> \
                 --cpuset-mems=<memory-node, optional> \
                 vllm-cpu-env

.. _build_arm_backend_from_source:

Build from source
-----------------

With Ubuntu 22.04/Linux, building from source should be no difference from building docker image.

Only tested on MacOS with M2 Pro chip, build on bare-metal will be added soon.
