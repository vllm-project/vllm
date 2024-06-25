.. _installation_rocm:

Installation with ROCm
======================

vLLM supports AMD GPUs with ROCm 5.7 and 6.0.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11
* GPU: MI200s (gfx90a), MI300 (gfx942), Radeon RX 7900 series (gfx1100)
* ROCm 6.0 and ROCm 5.7

Installation options:

#. :ref:`Build from source with docker <build_from_source_docker_rocm>`
#. :ref:`Build from source <build_from_source_rocm>`

.. _build_from_source_docker_rocm:

Option 1: Build from source with docker (recommended)
-----------------------------------------------------

You can build and install vLLM from source.

First, build a docker image from `Dockerfile.rocm <https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm>`_ and launch a docker container from the image.

`Dockerfile.rocm <https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm>`_ uses ROCm 6.0 by default, but also supports ROCm 5.7.
It provides flexibility to customize the build of docker image using the following arguments:

* `BASE_IMAGE`: specifies the base image used when running ``docker build``, specifically the PyTorch on ROCm base image. We have tested ROCm 5.7 and ROCm 6.0. The default is `rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1`
* `BUILD_FA`: specifies whether to build CK flash-attention. The default is 1. For `Radeon RX 7900 series (gfx1100) <https://rocm.docs.amd.com/projects/radeon/en/latest/index.html>`_, this should be set to 0 before flash-attention supports this target.
* `FX_GFX_ARCHS`: specifies the GFX architecture that is used to build CK flash-attention, for example, `gfx90a;gfx942` for MI200 and MI300. The default is `gfx90a;gfx942`
* `FA_BRANCH`: specifies the branch used to build the CK flash-attention in `ROCm's flash-attention repo <https://github.com/ROCmSoftwarePlatform/flash-attention>`_. The default is `ae7928c`
* `BUILD_TRITON`: specifies whether to build triton flash-attention. The default value is 1. 

Their values can be passed in when running ``docker build`` with ``--build-arg`` options.


To build vllm on ROCm 6.0 for MI200 and MI300 series, you can use the default:

.. code-block:: console

    $ docker build -f Dockerfile.rocm -t vllm-rocm .

To build vllm on ROCm 6.0 for Radeon RX7900 series (gfx1100), you should specify ``BUILD_FA`` as below:

.. code-block:: console

    $ docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .

To build docker image for vllm on ROCm 5.7, you can specify ``BASE_IMAGE`` as below:

.. code-block:: console

    $ docker build --build-arg BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1" \
       -f Dockerfile.rocm -t vllm-rocm . 

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
- `Pytorch <https://pytorch.org/>`_
- `hipBLAS <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/install.html>`_

For installing PyTorch, you can start from a fresh docker image, e.g, `rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging`, `rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1`, `rocm/pytorch-nightly`.

Alternatively, you can install pytorch using pytorch wheels. You can check Pytorch installation guild in Pytorch `Getting Started <https://pytorch.org/get-started/locally/>`_

For rocm6.0:

.. code-block:: console

    $ pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.0


For rocm5.7:

.. code-block:: console

    $ pip install torch --index-url https://download.pytorch.org/whl/rocm5.7


1. Install `Triton flash attention for ROCm <https://github.com/ROCm/triton>`_

Install ROCm's Triton flash attention (the default triton-mlir branch) following the instructions from `ROCm/triton <https://github.com/ROCm/triton/blob/triton-mlir/README.md>`_

2. Optionally, if you choose to use CK flash attention, you can install `flash attention for ROCm <https://github.com/ROCm/flash-attention/tree/flash_attention_for_rocm>`_

Install ROCm's flash attention (v2.0.4) following the instructions from `ROCm/flash-attention <https://github.com/ROCm/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support>`_

.. note::
    - If you are using rocm5.7 with pytorch 2.1.0 onwards, you don't need to apply the `hipify_python.patch`. You can build the ROCm flash attention directly.
    - If you fail to install `ROCm/flash-attention`, try cloning from the commit `6fd2f8e572805681cd67ef8596c7e2ce521ed3c6`.
    - ROCm's Flash-attention-2 (v2.0.4) does not support sliding windows attention.
    - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

3. Build vLLM.

.. code-block:: console

    $ cd vllm
    $ pip install -U -r requirements-rocm.txt
    $ python setup.py develop # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation


.. tip::

    - You may need to turn on the ``--enforce-eager`` flag if you experience process hang when running the `benchmark_thoughput.py` script to test your installation.
    - Triton flash attention is used by default. For benchmarking purposes, it is recommended to run a warm up step before collecting perf numbers.
    - To use CK flash-attention, please use this flag ``export VLLM_USE_TRITON_FLASH_ATTN=0`` to turn off triton flash attention. 
    - The ROCm version of pytorch, ideally, should match the ROCm driver version.
