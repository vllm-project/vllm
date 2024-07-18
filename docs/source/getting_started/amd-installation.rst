.. _installation_rocm:

Installation with ROCm
======================

vLLM supports AMD GPUs with ROCm 6.1.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11
* GPU: MI200s (gfx90a), MI300 (gfx942), Radeon RX 7900 series (gfx1100)
* ROCm 6.1

Installation options:

#. :ref:`Build from source with docker <build_from_source_docker_rocm>`
#. :ref:`Build from source <build_from_source_rocm>`

.. _build_from_source_docker_rocm:

Option 1: Build from source with docker (recommended)
-----------------------------------------------------

You can build and install vLLM from source.

First, build a docker image from `Dockerfile.rocm <https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm>`_ and launch a docker container from the image.

`Dockerfile.rocm <https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm>`_ uses ROCm 6.1 by default, but also supports ROCm 5.7 and 6.0 in older vLLM branches.
It provides flexibility to customize the build of docker image using the following arguments:

* `BASE_IMAGE`: specifies the base image used when running ``docker build``, specifically the PyTorch on ROCm base image.
* `BUILD_FA`: specifies whether to build CK flash-attention. The default is 1. For `Radeon RX 7900 series (gfx1100) <https://rocm.docs.amd.com/projects/radeon/en/latest/index.html>`_, this should be set to 0 before flash-attention supports this target.
* `FX_GFX_ARCHS`: specifies the GFX architecture that is used to build CK flash-attention, for example, `gfx90a;gfx942` for MI200 and MI300. The default is `gfx90a;gfx942`
* `FA_BRANCH`: specifies the branch used to build the CK flash-attention in `ROCm's flash-attention repo <https://github.com/ROCmSoftwarePlatform/flash-attention>`_. The default is `ae7928c`
* `BUILD_TRITON`: specifies whether to build triton flash-attention. The default value is 1. 

Their values can be passed in when running ``docker build`` with ``--build-arg`` options.


To build vllm on ROCm 6.1 for MI200 and MI300 series, you can use the default:

.. code-block:: console

    $ DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .

To build vllm on ROCm 6.1 for Radeon RX7900 series (gfx1100), you should specify ``BUILD_FA`` as below:

.. code-block:: console

    $ DOCKER_BUILDKIT=1 docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .

To run the above docker image ``vllm-rocm``, use the below command:

.. code-block:: console

    $ docker run -it \
       --network=host \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --device /dev/kfd \
       --device /dev/dri \
       -v <path/to/model>:/app/model \
       vllm-rocm \
       bash

Where the `<path/to/model>` is the location where the model is stored, for example, the weights for llama2 or llama3 models.


.. _build_from_source_rocm:

Option 2: Build from source
---------------------------

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):

- `ROCm <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_
- `PyTorch <https://pytorch.org/>`_
- `hipBLAS <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/install.html>`_

For installing PyTorch, you can start from a fresh docker image, e.g, `rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging`, `rocm/pytorch-nightly`.

Alternatively, you can install PyTorch using PyTorch wheels. You can check PyTorch installation guild in PyTorch `Getting Started <https://pytorch.org/get-started/locally/>`_


1. Install `Triton flash attention for ROCm <https://github.com/ROCm/triton>`_

Install ROCm's Triton flash attention (the default triton-mlir branch) following the instructions from `ROCm/triton <https://github.com/ROCm/triton/blob/triton-mlir/README.md>`_

2. Optionally, if you choose to use CK flash attention, you can install `flash attention for ROCm <https://github.com/ROCm/flash-attention/tree/flash_attention_for_rocm>`_

Install ROCm's flash attention (v2.0.4) following the instructions from `ROCm/flash-attention <https://github.com/ROCm/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support>`_

.. note::
    - ROCm's Flash-attention-2 (v2.0.4) does not support sliding windows attention.
    - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

3. Build vLLM.

.. code-block:: console

    $ cd vllm
    $ pip install -U -r requirements-rocm.txt
    $ python setup.py develop # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation


.. tip::

    - Triton flash attention is used by default. For benchmarking purposes, it is recommended to run a warm up step before collecting perf numbers.
    - To use CK flash-attention or PyTorch naive attention, please use this flag ``export VLLM_USE_TRITON_FLASH_ATTN=0`` to turn off triton flash attention. 
    - The ROCm version of PyTorch, ideally, should match the ROCm driver version.
