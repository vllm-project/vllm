.. _installation_rocm:

Installation with ROCm
======================

vLLM 0.2.4 onwards supports model inferencing and serving on AMD GPUs with ROCm.
At the moment AWQ quantization is not supported in ROCm, but SqueezeLLM quantization has been ported.
Data types currently supported in ROCm are FP16 and BF16.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11
* GPU: MI200s (gfx90a), MI300 (gfx942), Radeon RX 7900 series (gfx1100)
* Pytorch 2.0.1/2.1.1/2.2
* ROCm 5.7 (Verified on python 3.10) or ROCm 6.0 (Verified on python 3.9)

Installation options:

#. :ref:`(Recommended) Quick start with vLLM pre-installed in Docker Image <quick_start_docker_rocm>`
#. :ref:`Build from source <build_from_source_rocm>`
#. :ref:`Build from source with docker <build_from_source_docker_rocm>`

.. _quick_start_docker_rocm:

(Recommended) Option 1: Quick start with vLLM pre-installed in Docker Image
---------------------------------------------------------------------------

This option is for ROCm 5.7 only:

.. code-block:: console

    $ docker pull embeddedllminfo/vllm-rocm:vllm-v0.2.4
    $ docker run -it \
       --network=host \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --device /dev/kfd \
       --device /dev/dri \
       -v <path/to/model>:/app/model \
       embeddedllminfo/vllm-rocm \
       bash


.. _build_from_source_rocm:

Option 2: Build from source
---------------------------

You can build and install vLLM from source:

Below instruction is for ROCm 5.7 only. 
At the time of this documentation update, PyTorch on ROCm 6.0 wheel is not yet available on the PyTorch website.

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):

- `ROCm <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_
- `Pytorch <https://pytorch.org/>`_

    .. code-block:: console

        $ pip install torch==2.2.0.dev20231206+rocm5.7 --index-url https://download.pytorch.org/whl/nightly/rocm5.7 # tested version


1. Install `flash attention for ROCm <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm>`_

    Install ROCm's flash attention (v2.0.4) following the instructions from `ROCmSoftwarePlatform/flash-attention <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support>`_

.. note::
    - If you are using rocm5.7 with pytorch 2.1.0 onwards, you don't need to apply the `hipify_python.patch`. You can build the ROCm flash attention directly.
    - If you fail to install `ROCmSoftwarePlatform/flash-attention`, try cloning from the commit `6fd2f8e572805681cd67ef8596c7e2ce521ed3c6`.
    - ROCm's Flash-attention-2 (v2.0.4) does not support sliding windows attention.
    - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

2. Setup `xformers==0.0.23` without dependencies, and apply patches to adapt for ROCm flash attention

    .. code-block:: console

        $ pip install xformers==0.0.23 --no-deps
        $ bash patch_xformers.rocm.sh

3. Build vLLM.

    .. code-block:: console

        $ cd vllm
        $ pip install -U -r requirements-rocm.txt
        $ python setup.py install # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation


.. _build_from_source_docker_rocm:

Option 3: Build from source with docker
-----------------------------------------------------

You can build and install vLLM from source:

Build a docker image from `Dockerfile.rocm`, and launch a docker container.

The `Dokerfile.rocm` is designed to support both ROCm 5.7 and ROCm 6.0 and later versions. It provides flexibility to customize the build of docker image using the following arguments:

* `BASE_IMAGE`: specifies the base image used when running ``docker build``, specifically the PyTorch on ROCm base image. We have tested ROCm 5.7 and ROCm 6.0. The default is `rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1`
* `FX_GFX_ARCHS`: specifies the GFX architecture that is used to build flash-attention, for example, `gfx90a;gfx942` for MI200 and MI300. The default is `gfx90a;gfx942`
* `FA_BRANCH`: specifies the branch used to build the flash-attention in `ROCmSoftwarePlatform's flash-attention repo <https://github.com/ROCmSoftwarePlatform/flash-attention>`_. The default is `3d2b6f5`
* `BUILD_FA`: specifies whether to build flash-attention. For `Radeon RX 7900 series (gfx1100) <https://rocm.docs.amd.com/projects/radeon/en/latest/index.html>`_, this should be set to 0 before flash-attention supports this target.

Their values can be passed in when running ``docker build`` with ``--build-arg`` options.

For example, to build docker image for vllm on ROCm 5.7, you can run:

.. code-block:: console

    $ docker build --build-arg BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1" \
       -f Dockerfile.rocm -t vllm-rocm . 

To build vllm on ROCm 6.0, you can use the default:

.. code-block:: console

    $ docker build -f Dockerfile.rocm -t vllm-rocm . 
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

Alternatively, if you plan to install vLLM-ROCm on a local machine or start from a fresh docker image (e.g. rocm/pytorch), you can follow the steps below:

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):

- `ROCm <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_
- `Pytorch <https://pytorch.org/>`_
- `hipBLAS <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/install.html>`_

1. Install `flash attention for ROCm <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm>`_

    Install ROCm's flash attention (v2.0.4) following the instructions from `ROCmSoftwarePlatform/flash-attention <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support>`_

.. note::
    - If you are using rocm5.7 with pytorch 2.1.0 onwards, you don't need to apply the `hipify_python.patch`. You can build the ROCm flash attention directly.
    - If you fail to install `ROCmSoftwarePlatform/flash-attention`, try cloning from the commit `6fd2f8e572805681cd67ef8596c7e2ce521ed3c6`.
    - ROCm's Flash-attention-2 (v2.0.4) does not support sliding windows attention.
    - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

2. Setup `xformers==0.0.23` without dependencies, and apply patches to adapt for ROCm flash attention

    .. code-block:: console

        $ pip install xformers==0.0.23 --no-deps
        $ bash patch_xformers.rocm.sh

3. Build vLLM.

    .. code-block:: console

        $ cd vllm
        $ pip install -U -r requirements-rocm.txt
        $ python setup.py install # This may take 5-10 minutes.

.. note::

    - You may need to turn on the ``--enforce-eager`` flag if you experience process hang when running the `benchmark_thoughput.py` script to test your installation.

