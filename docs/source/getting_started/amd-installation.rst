.. _installation:

Installation with ROCm
============

vLLM-ROCm is here! Currently it is supporting llama-2.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11 (Recommended 3.10 as this is the version that has been tested on.)
* GPU: MI210
* Pytorch 2.0.1/2.1.1
* ROCm 5.7


Install with pip
----------------

You can install vLLM using pip:

.. code-block:: console

    $ # (Optional) Create a new conda environment.
    $ conda create -n myenv python=3.8 -y
    $ conda activate myenv

    $ # Install vLLM with CUDA 12.1.
    $ pip install vllm

.. note::

    As of now, vLLM's binaries are compiled on CUDA 12.1 by default.
    However, you can install vLLM with CUDA 11.8 by running:

    .. code-block:: console

        $ # Install vLLM with CUDA 11.8.
        $ # Replace `cp310` with your Python version (e.g., `cp38`, `cp39`, `cp311`).
        $ pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-cp310-cp310-manylinux1_x86_64.whl

        $ # Re-install PyTorch with CUDA 11.8.
        $ pip uninstall torch -y
        $ pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118


.. _build_from_source:

Build from source with docker
-----------------

You can also build and install vLLM from source:

Build a docker image from `rocm.Dockerfile`, and launch a docker container.

.. code-block:: console

    $ docker build -f rocm.Dockerfile -t vllm-rocm . 
    $ docker run -it \
       --network=host \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --shm-size 8G \
       --device /dev/kfd \
       --device /dev/dri \
       -v <path/to/model>:/app/hf_model \
       vllm-rocm \
       bash

If you are going to setup on new pytorch+rocm5.7 docker container, you can follow the following steps.

1. Install flash-attention-2-rocm

    If you are using Pytorch-2.0.1+rocm5.7.

    Install flash-attention-2 (v2.0.4) following the instruction from [ROCmSoftwarePlatform/flash-attention](https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm)


    If you are using Pytorch-2.1.x+rocm5.7 or Pytorch-2.2.x+rocm5.7, you don't need to apply the `hipify_python.patch`.
    You can directly build the flash-attention-2.

    .. code-block:: console

        $ bash patch_torch211_flash_attn2.rocm.sh

    .. note::
        - Flash-attention-2 (v2.0.4) does not support sliding windows attention.
        - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

2. Setup xformers==0.0.22.post7 without dependencies, and apply patches

    .. code-block:: console

        $ pip install xformers==0.0.22.post7 --no-deps
        $ bash patch_xformers-0.0.22.post7.rocm.sh

3. Build vllm.

    .. code-block:: console
        $ cd vllm
        $ pip install -U -r requirements-rocm.txt
        $ python setup.py install # This may take 5-10 minutes.
