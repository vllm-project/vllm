.. _installation:

Installation with ROCm
============

vLLM 0.2.x onwards supports model inferencing and serving on AMD GPUs with ROCm. At the moment AWQ quantization is not supported, but SqueezeLLM has been incorporated.
The ROCm version is tested and verified on LLaMA-2 models.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11 (Verified on 3.10)
* GPU: MI200s
* Pytorch 2.0.1/2.1.1
* ROCm >= 5.7.0


.. _build_from_source:

Build from source with docker
-----------------

You can also build and install vLLM from source:

Build a docker image from `Dockerfile.rocm`, and launch a docker container.

.. code-block:: console

    $ docker build -f Dockerfile.rocm -t vllm-rocm . 
    $ docker run -it \
       --network=host \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --shm-size 8G \
       --device /dev/kfd \
       --device /dev/dri \
       -v <path/to/model>:/app/model \
       vllm-rocm \
       bash

If you plan to install vLLM-ROCm on a local machine or start from a fresh docker image (e.g. pytorch+rocm5.7), you can follow the steps below:

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):
    - `ROCm <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_ and
    - `Pytorch <https://pytorch.org/>`_

1. Install flash attention for ROCm

    If you are using Pytorch-2.0.1+rocm5.7.

    Install flash-attention-2 (v2.0.4) following the instruction from `ROCmSoftwarePlatform/flash-attention <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm>`_


    If you are using Pytorch-2.1.x+rocm5.7 or Pytorch-2.2.x+rocm5.7, you don't need to apply the `hipify_python.patch`.
    You can directly build the flash-attention-2.

    .. code-block:: console

        $ bash patch_torch211_flash_attn2.rocm.sh

    .. note::
        - ROCm's Flash-attention-2 (v2.0.4) does not support sliding windows attention.
        - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

2. Setup xformers==0.0.22.post7 without dependencies, and apply patches to adapt for ROCm flash attention

    .. code-block:: console

        $ pip install xformers==0.0.22.post7 --no-deps
        $ bash patch_xformers-0.0.22.post7.rocm.sh

3. Build vllm.

    .. code-block:: console
        $ cd vllm
        $ pip install -U -r requirements-rocm.txt
        $ python setup.py install # This may take 5-10 minutes.

